import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from convert_yolo_to_tracknet_csv import round_half_up


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "clips"
    / "ball_challenge_002.mp4"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "evaluation_001"
    / "manifest.csv"
)

DEFAULT_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)

DEFAULT_REFERENCE_ROOT = Path(
    r"C:\GitHub\TrackNetV3-reference"
)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_training"
    / "pilot_v2_transfer_lr1e-4_bs2_seed13_fresh_dataset"
    / "checkpoints"
    / "epoch_003.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_sparse_test"
    / "epoch_003_thresholds"
)

EXPECTED_REFERENCE_COMMIT = (
    "77c123ad4dd449b7d275f16cc43f316ba5b54042"
)

HEATMAP_WIDTH = 512
HEATMAP_HEIGHT = 288

FLOOR_HOTSPOT_X_RANGE = range(224, 240)
FLOOR_HOTSPOT_Y_RANGE = range(216, 232)

HAND_HOTSPOT_X_RANGE = range(216, 224)
HAND_HOTSPOT_Y_RANGE = range(88, 96)


def parse_args() -> argparse.Namespace:
    """疎なtestラベルを評価する条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "連続動画へTrackNetV3を適用し、"
            "アノテーション済みtestフレームだけを評価する"
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="連続620フレームを含むtest動画",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="test画像と元動画フレーム番号の対応表",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Label Studioから出力したYOLOラベル",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="固定したTrackNetV3公式実装",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="評価するTrackNet checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="評価結果の新規出力先",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.40,0.50",
        help="カンマ区切りのヒートマップしきい値",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="動画推論のbatch size",
    )
    parser.add_argument(
        "--max-sample-num",
        type=int,
        default=100,
        help="背景中央値を作る最大サンプル数",
    )

    return parser.parse_args()

def get_git_commit(
    repository_root: Path,
) -> str:
    """参照リポジトリの現在commitを取得する。"""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()

def calculate_sha256(path: Path) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def parse_thresholds(source: str) -> list[float]:
    """カンマ区切りのしきい値を昇順にする。"""
    thresholds = sorted(
        {
            float(value.strip())
            for value in source.split(",")
            if value.strip()
        }
    )

    if not thresholds:
        raise ValueError(
            "しきい値が指定されていません"
        )

    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError(
                "しきい値は0より大きく"
                "1未満にしてください: "
                f"{threshold}"
            )

    return thresholds


def validate_args(
    args: argparse.Namespace,
) -> list[float]:
    """入力、出力、数値条件を検証する。"""
    args.video = args.video.resolve()
    args.manifest = args.manifest.resolve()
    args.label_dir = args.label_dir.resolve()
    args.reference_root = (
        args.reference_root.resolve()
    )
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()

    required_files = (
        args.video,
        args.manifest,
        args.checkpoint,
        args.reference_root / "dataset.py",
        args.reference_root / "model.py",
        args.reference_root / "test.py",
    )

    for required_file in required_files:
        if not required_file.is_file():
            raise FileNotFoundError(
                f"必要なファイルがありません: "
                f"{required_file}"
            )

    if not args.label_dir.is_dir():
        raise FileNotFoundError(
            f"ラベルフォルダがありません: "
            f"{args.label_dir}"
        )

    if args.output_dir.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{args.output_dir}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-sizeには1以上を指定してください"
        )

    if args.max_sample_num <= 0:
        raise ValueError(
            "--max-sample-numには1以上を"
            "指定してください"
        )

    return parse_thresholds(args.thresholds)


def import_official_modules(
    reference_root: Path,
):
    """固定した公式実装から必要な機能を読む。"""
    reference_text = str(reference_root)

    if reference_text not in sys.path:
        sys.path.insert(0, reference_text)

    from dataset import Video_IterableDataset
    from test import (
        get_ensemble_weight,
        predict_location,
    )
    from utils.general import (
        get_model,
        to_img,
    )
    from utils.metric import get_metric

    return (
        Video_IterableDataset,
        get_ensemble_weight,
        predict_location,
        get_model,
        to_img,
        get_metric,
    )


def read_video_metadata(
    video_path: Path,
) -> dict[str, int | float]:
    """test動画のサイズ、FPS、フレーム数を読む。"""
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"動画を開けません: {video_path}"
        )

    metadata = {
        "width": int(
            video.get(cv2.CAP_PROP_FRAME_WIDTH)
        ),
        "height": int(
            video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ),
        "fps": float(
            video.get(cv2.CAP_PROP_FPS)
        ),
        "frame_count": int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        ),
    }

    video.release()

    if metadata["frame_count"] <= 0:
        raise ValueError(
            "動画フレーム数を取得できません"
        )

    return metadata


def build_label_map(
    label_dir: Path,
) -> dict[str, Path]:
    """元画像stemとLabel Studioラベルを対応させる。"""
    label_map = {}

    for label_path in label_dir.glob("*.txt"):
        match = re.search(
            r"(frame_\d{6})$",
            label_path.stem,
        )

        if match is None:
            raise ValueError(
                "ラベル名から元画像名を"
                "取得できません: "
                f"{label_path.name}"
            )

        image_stem = match.group(1)

        if image_stem in label_map:
            raise ValueError(
                "同じ元画像に複数ラベルがあります: "
                f"{image_stem}"
            )

        label_map[image_stem] = label_path

    return label_map


def read_yolo_label(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    """YOLOラベルを元画像ピクセル座標へ戻す。"""
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8-sig",
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        return {
            "visibility": 0,
            "x_pixel": 0,
            "y_pixel": 0,
        }

    if len(lines) != 1:
        raise ValueError(
            "1フレームに複数ラベルがあります: "
            f"{label_path}"
        )

    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(
            "YOLOラベルが5列ではありません: "
            f"{label_path}"
        )

    if parts[0] != "0":
        raise ValueError(
            "ball以外のクラスがあります: "
            f"{label_path}"
        )

    x_center = float(parts[1])
    y_center = float(parts[2])

    if not (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
    ):
        raise ValueError(
            "YOLO中心座標が範囲外です: "
            f"{label_path}"
        )

    x_pixel = round_half_up(
        x_center * image_width
    )
    y_pixel = round_half_up(
        y_center * image_height
    )

    return {
        "visibility": 1,
        "x_pixel": x_pixel,
        "y_pixel": y_pixel,
    }


def read_test_labels(
    manifest_path: Path,
    label_dir: Path,
    video_metadata: dict[str, int | float],
) -> dict[int, dict[str, int | str]]:
    """manifest順に疎なtest正解を作る。"""
    label_map = build_label_map(label_dir)
    test_labels = {}

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        for manifest_row in reader:
            file_name = manifest_row["file_name"]
            image_stem = Path(file_name).stem
            frame_index = int(
                manifest_row["frame_index"]
            )

            if image_stem not in label_map:
                raise FileNotFoundError(
                    "対応するラベルがありません: "
                    f"{file_name}"
                )

            if frame_index in test_labels:
                raise ValueError(
                    "manifestのフレーム番号が"
                    "重複しています: "
                    f"{frame_index}"
                )

            label = read_yolo_label(
                label_map[image_stem],
                int(video_metadata["width"]),
                int(video_metadata["height"]),
            )

            width_scaler = (
                int(video_metadata["width"])
                / HEATMAP_WIDTH
            )
            height_scaler = (
                int(video_metadata["height"])
                / HEATMAP_HEIGHT
            )

            test_labels[frame_index] = {
                "file_name": file_name,
                "visibility": label["visibility"],
                "x_original": label["x_pixel"],
                "y_original": label["y_pixel"],
                "x_heatmap": int(
                    label["x_pixel"] / width_scaler
                ),
                "y_heatmap": int(
                    label["y_pixel"] / height_scaler
                ),
            }

    manifest_image_stems = {
        Path(label["file_name"]).stem
        for label in test_labels.values()
    }
    unexpected_label_stems = sorted(
        set(label_map) - manifest_image_stems
    )

    if unexpected_label_stems:
        raise ValueError(
            "manifestに存在しない余分なラベルがあります: "
            f"{unexpected_label_stems}"
        )


    frame_count = int(
        video_metadata["frame_count"]
    )

    for frame_index in test_labels:
        if not 0 <= frame_index < frame_count:
            raise ValueError(
                "testフレーム番号が動画範囲外です: "
                f"{frame_index}"
            )

    return test_labels


def create_model(
    get_model,
    checkpoint_path: Path,
    device: torch.device,
):
    """checkpointからTrackNetモデルを復元する。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model" not in checkpoint
        or "param_dict" not in checkpoint
    ):
        raise KeyError(
            "checkpointにmodelまたは"
            "param_dictがありません"
        )

    parameters = checkpoint["param_dict"]
    model_name = parameters.get(
        "model_name",
        "TrackNet",
    )
    sequence_length = int(
        parameters.get("seq_len", 8)
    )
    background_mode = parameters.get(
        "bg_mode",
        "concat",
    )

    if model_name != "TrackNet":
        raise ValueError(
            "TrackNet checkpointではありません: "
            f"{model_name}"
        )

    if sequence_length != 8:
        raise ValueError(
            "系列長が想定と異なります: "
            f"{sequence_length}"
        )

    model = get_model(
        model_name,
        seq_len=sequence_length,
        bg_mode=background_mode,
    ).to(device)

    load_result = model.load_state_dict(
        checkpoint["model"]
    )

    if load_result.missing_keys:
        raise RuntimeError(
            "重み読込にmissing keysがあります: "
            f"{load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "重み読込にunexpected keysがあります: "
            f"{load_result.unexpected_keys}"
        )

    model.eval()

    return (
        model,
        sequence_length,
        background_mode,
    )


def expected_prediction_count(
    frame_index: int,
    frame_count: int,
    sequence_length: int,
) -> int:
    """先頭・末尾を含む予測重複数を返す。"""
    return min(
        sequence_length,
        frame_index + 1,
        frame_count - frame_index,
    )


def run_temporal_ensemble(
    model,
    Video_IterableDataset,
    get_ensemble_weight,
    video_path: Path,
    test_labels: dict[int, dict[str, int | str]],
    frame_count: int,
    sequence_length: int,
    background_mode: str,
    batch_size: int,
    max_sample_num: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """公式weight仕様でmanifest対象フレームだけ統合する。"""
    dataset = Video_IterableDataset(
        str(video_path),
        seq_len=sequence_length,
        sliding_step=1,
        bg_mode=background_mode,
        max_sample_num=max_sample_num,
    )

    if dataset.video_len != frame_count:
        raise ValueError(
            "Datasetと動画のフレーム数が"
            "一致しません"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    ensemble_weight = get_ensemble_weight(
        sequence_length,
        "weight",
    )

    accumulated_heatmaps = {
        frame_index: torch.zeros(
            (
                HEATMAP_HEIGHT,
                HEATMAP_WIDTH,
            ),
            dtype=torch.float32,
        )
        for frame_index in test_labels
    }

    valid_sequence_count = (
        frame_count - sequence_length + 1
    )
    processed_sequence_count = 0

    model.eval()

    with torch.inference_mode():
        for indices, input_tensor in data_loader:
            input_tensor = input_tensor.float().to(
                device
            )
            output = model(input_tensor).detach().cpu()

            for batch_index in range(
                output.shape[0]
            ):
                sequence_start = int(
                    indices[
                        batch_index,
                        0,
                        1,
                    ].item()
                )

                if sequence_start >= (
                    valid_sequence_count
                ):
                    continue

                processed_sequence_count += 1

                for sequence_index in range(
                    sequence_length
                ):
                    frame_index = (
                        sequence_start
                        + sequence_index
                    )

                    if frame_index not in test_labels:
                        continue

                    overlap_count = (
                        expected_prediction_count(
                            frame_index,
                            frame_count,
                            sequence_length,
                        )
                    )

                    if overlap_count == sequence_length:
                        factor = float(
                            ensemble_weight[
                                sequence_index
                            ].item()
                        )
                    else:
                        factor = 1.0 / overlap_count

                    accumulated_heatmaps[
                        frame_index
                    ] += (
                        output[
                            batch_index,
                            sequence_index,
                        ]
                        * factor
                    )

    if processed_sequence_count != (
        valid_sequence_count
    ):
        raise RuntimeError(
            "有効系列数が想定と一致しません: "
            f"{processed_sequence_count}"
        )

    for frame_index, heatmap in (
        accumulated_heatmaps.items()
    ):
        if not torch.isfinite(heatmap).all():
            raise RuntimeError(
                "非有限値のヒートマップがあります: "
                f"{frame_index}"
            )

    return accumulated_heatmaps


def classify_prediction(
    ground_truth_visible: int,
    ground_truth_x: int,
    ground_truth_y: int,
    predicted_visible: int,
    predicted_x: int,
    predicted_y: int,
    tolerance: float,
) -> tuple[str, float | None]:
    """公式TP・TN・FP1・FP2・FNへ分類する。"""
    if (
        ground_truth_visible == 0
        and predicted_visible == 0
    ):
        return "TN", None

    if (
        ground_truth_visible == 0
        and predicted_visible == 1
    ):
        return "FP2", None

    if (
        ground_truth_visible == 1
        and predicted_visible == 0
    ):
        return "FN", None

    distance = float(
        np.hypot(
            predicted_x - ground_truth_x,
            predicted_y - ground_truth_y,
        )
    )

    if distance > tolerance:
        return "FP1", distance

    return "TP", distance


def evaluate_thresholds(
    heatmaps: dict[int, torch.Tensor],
    test_labels: dict[int, dict[str, int | str]],
    thresholds: list[float],
    predict_location,
    to_img,
    get_metric,
    tolerance: float,
) -> tuple[list[dict], list[dict]]:
    """各しきい値でmanifest対象フレームを評価する。"""
    per_frame_rows = []
    summary_rows = []

    for threshold in thresholds:
        counts = {
            "TP": 0,
            "TN": 0,
            "FP1": 0,
            "FP2": 0,
            "FN": 0,
        }

        for frame_index in sorted(test_labels):
            label = test_labels[frame_index]
            heatmap = heatmaps[frame_index]

            raw_max = float(
                heatmap.max().item()
            )
            flat_index = int(
                heatmap.argmax().item()
            )
            raw_peak_y, raw_peak_x = divmod(
                flat_index,
                HEATMAP_WIDTH,
            )

            binary_heatmap = (
                heatmap > threshold
            ).numpy()
            heatmap_image = to_img(
                binary_heatmap
            )

            x, y, width, height = predict_location(
                heatmap_image
            )

            predicted_x = int(
                x + width / 2
            )
            predicted_y = int(
                y + height / 2
            )
            predicted_visible = int(
                not (
                    predicted_x == 0
                    and predicted_y == 0
                )
            )

            classification, distance = (
                classify_prediction(
                    int(label["visibility"]),
                    int(label["x_heatmap"]),
                    int(label["y_heatmap"]),
                    predicted_visible,
                    predicted_x,
                    predicted_y,
                    tolerance,
                )
            )

            counts[classification] += 1

            raw_peak_distance = None

            if int(label["visibility"]) == 1:
                raw_peak_distance = float(
                    np.hypot(
                        raw_peak_x
                        - int(label["x_heatmap"]),
                        raw_peak_y
                        - int(label["y_heatmap"]),
                    )
                )

            floor_hotspot = int(
                raw_peak_x
                in FLOOR_HOTSPOT_X_RANGE
                and raw_peak_y
                in FLOOR_HOTSPOT_Y_RANGE
            )
            hand_hotspot = int(
                raw_peak_x
                in HAND_HOTSPOT_X_RANGE
                and raw_peak_y
                in HAND_HOTSPOT_Y_RANGE
            )

            per_frame_rows.append(
                {
                    "threshold": threshold,
                    "frame_index": frame_index,
                    "file_name": label["file_name"],
                    "ground_truth_visible": int(
                        label["visibility"]
                    ),
                    "ground_truth_x": int(
                        label["x_heatmap"]
                    ),
                    "ground_truth_y": int(
                        label["y_heatmap"]
                    ),
                    "predicted_visible": (
                        predicted_visible
                    ),
                    "predicted_x": predicted_x,
                    "predicted_y": predicted_y,
                    "distance": distance,
                    "classification": classification,
                    "raw_max": raw_max,
                    "raw_peak_x": raw_peak_x,
                    "raw_peak_y": raw_peak_y,
                    "raw_peak_distance": (
                        raw_peak_distance
                    ),
                    "floor_hotspot": floor_hotspot,
                    "hand_hotspot": hand_hotspot,
                }
            )

        (
            accuracy,
            precision,
            recall,
            f1,
            miss_rate,
        ) = get_metric(
            counts["TP"],
            counts["TN"],
            counts["FP1"],
            counts["FP2"],
            counts["FN"],
        )

        threshold_rows = [
            row
            for row in per_frame_rows
            if row["threshold"] == threshold
        ]

        summary_rows.append(
            {
                "threshold": threshold,
                **counts,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "miss_rate": float(miss_rate),
                "floor_hotspot_count": sum(
                    row["floor_hotspot"]
                    for row in threshold_rows
                ),
                "hand_hotspot_count": sum(
                    row["hand_hotspot"]
                    for row in threshold_rows
                ),
            }
        )

    return per_frame_rows, summary_rows


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """辞書行をUTF-8 CSVへ保存する。"""
    if not rows:
        raise ValueError(
            f"CSVへ保存する行がありません: "
            f"{output_path}"
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """連続動画推論と疎なtest評価を実行する。"""
    args = parse_args()
    thresholds = validate_args(args)
    reference_commit = get_git_commit(
        args.reference_root
    )

    if reference_commit != (
        EXPECTED_REFERENCE_COMMIT
    ):
        raise RuntimeError(
            "公式参照commitが固定値と異なります: "
            f"{reference_commit}"
        )

    (
        Video_IterableDataset,
        get_ensemble_weight,
        predict_location,
        get_model,
        to_img,
        get_metric,
    ) = import_official_modules(
        args.reference_root
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "この評価はCUDA GPUを前提とします"
        )

    video_metadata = read_video_metadata(
        args.video
    )
    test_labels = read_test_labels(
        args.manifest,
        args.label_dir,
        video_metadata,
    )

    (
        model,
        sequence_length,
        background_mode,
    ) = create_model(
        get_model,
        args.checkpoint,
        device,
    )

    checkpoint_hash = calculate_sha256(
        args.checkpoint
    )
    script_hash = calculate_sha256(
        Path(__file__).resolve()
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = perf_counter()

    heatmaps = run_temporal_ensemble(
        model,
        Video_IterableDataset,
        get_ensemble_weight,
        args.video,
        test_labels,
        int(video_metadata["frame_count"]),
        sequence_length,
        background_mode,
        args.batch_size,
        args.max_sample_num,
        device,
    )

    torch.cuda.synchronize()
    elapsed_seconds = (
        perf_counter() - start_time
    )

    per_frame_rows, summary_rows = (
        evaluate_thresholds(
            heatmaps,
            test_labels,
            thresholds,
            predict_location,
            to_img,
            get_metric,
            tolerance=4.0,
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        args.output_dir / "per_frame.csv",
        per_frame_rows,
    )
    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
    )

    analysis = {
        "video": str(args.video),
        "manifest": str(args.manifest),
        "label_dir": str(args.label_dir),
        "reference_root": str(
            args.reference_root
        ),
        "reference_commit": reference_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "script_sha256": script_hash,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "video_metadata": video_metadata,
        "test_frame_count": len(test_labels),
        "positive_count": sum(
            label["visibility"]
            for label in test_labels.values()
        ),
        "negative_count": sum(
            1 - label["visibility"]
            for label in test_labels.values()
        ),
        "sequence_length": sequence_length,
        "background_mode": background_mode,
        "batch_size": args.batch_size,
        "max_sample_num": (
            args.max_sample_num
        ),
        "ensemble_mode": "weight",
        "thresholds": thresholds,
        "tolerance": 4.0,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024**2
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024**2
        ),
        "summary": summary_rows,
    }

    with (
        args.output_dir / "analysis.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            analysis,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "TrackNetV3の疎なtest評価が"
        "完了しました"
    )
    print(f"デバイス: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "公式参照commit: "
        f"{reference_commit}"
    )
    print(
        "checkpoint SHA-256: "
        f"{checkpoint_hash}"
    )
    print(f"評価ラベル数: {len(test_labels)}")
    print(
        "正例: "
        f"{analysis['positive_count']}"
    )
    print(
        "負例: "
        f"{analysis['negative_count']}"
    )
    print(
        "連続動画フレーム数: "
        f"{video_metadata['frame_count']}"
    )
    print(
        "有効8フレーム系列数: "
        f"{int(video_metadata['frame_count']) - 7}"
    )

    print("=== しきい値別結果 ===")

    for row in summary_rows:
        print(
            f"threshold={row['threshold']:.2f}, "
            f"TP={row['TP']}, "
            f"TN={row['TN']}, "
            f"FP1={row['FP1']}, "
            f"FP2={row['FP2']}, "
            f"FN={row['FN']}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"f1={row['f1']:.4f}, "
            f"floor_hotspot="
            f"{row['floor_hotspot_count']}, "
            f"hand_hotspot="
            f"{row['hand_hotspot_count']}"
        )

    print(
        f"評価時間: {elapsed_seconds:.4f}秒"
    )
    print(
        "ピーク割当VRAM: "
        f"{analysis['peak_allocated_vram_mib']:.1f} MiB"
    )
    print(
        "ピーク予約VRAM: "
        f"{analysis['peak_reserved_vram_mib']:.1f} MiB"
    )
    print(f"出力先: {args.output_dir}")


if __name__ == "__main__":
    main()