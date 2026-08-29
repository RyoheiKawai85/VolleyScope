import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tracknet_official_pilot_v2"
)

DEFAULT_MAPPING_CSV = (
    DEFAULT_DATASET_ROOT
    / "frame_mapping.csv"
)

DEFAULT_REFERENCE_ROOT = Path(
    r"C:\GitHub\TrackNetV3-reference"
)

DEFAULT_CHECKPOINT = (
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
    / "tracknet_heatmap_analysis"
    / "epoch_003_val_thresholds"
)

EXPECTED_REFERENCE_COMMIT = (
    "77c123ad4dd449b7d275f16cc43f316ba5b54042"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "8715203B396FBABE82CB99E11AD3F2C6137E892C66BF664A8C45836040E7C56D"
)

EXPECTED_MODEL_NAME = "TrackNet"
EXPECTED_SEQUENCE_LENGTH = 8
EXPECTED_BACKGROUND_MODE = "concat"
EXPECTED_TOLERANCE = 4
OFFICIAL_THRESHOLD = 0.5

EXPECTED_VALIDATION_SEQUENCES = 15
EXPECTED_VALIDATION_FRAMES = 120
EXPECTED_VISIBLE_FRAMES = 118
EXPECTED_INVISIBLE_FRAMES = 2


def parse_thresholds(
    value: str,
) -> list[float]:
    """カンマ区切りのしきい値を検証して返す。"""
    try:
        thresholds = [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "しきい値を数値へ変換できません"
        ) from error

    if not thresholds:
        raise argparse.ArgumentTypeError(
            "しきい値を1つ以上指定してください"
        )

    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise argparse.ArgumentTypeError(
                "しきい値は0より大きく"
                "1より小さくしてください"
            )

    unique_thresholds = sorted(
        set(thresholds)
    )

    if OFFICIAL_THRESHOLD not in unique_thresholds:
        raise argparse.ArgumentTypeError(
            "公式しきい値0.5を含めてください"
        )

    return unique_thresholds


def parse_args() -> argparse.Namespace:
    """ヒートマップ分析の実行条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3 Epoch 3の生ヒートマップと"
            "しきい値別分類を分析する"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="公式互換データセットのルート",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="ローカル番号と元フレームの対応表",
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
        default=DEFAULT_CHECKPOINT,
        help="分析するEpoch 3 checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="分析結果の新規出力先",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds(
            "0.05,0.10,0.20,0.30,0.40,0.50"
        ),
        help=(
            "カンマ区切りのしきい値。"
            "公式値0.5を必ず含める"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="推論時のbatch size",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """入力、出力先、CUDAを検証する。"""
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            "データセットが見つかりません: "
            f"{args.dataset_root}"
        )

    if not args.mapping_csv.is_file():
        raise FileNotFoundError(
            "対応表が見つかりません: "
            f"{args.mapping_csv}"
        )

    if not args.reference_root.is_dir():
        raise FileNotFoundError(
            "公式参照実装が見つかりません: "
            f"{args.reference_root}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "checkpointが見つかりません: "
            f"{args.checkpoint}"
        )

    if args.output_dir.exists():
        raise FileExistsError(
            "分析結果の上書きを防ぐため停止します: "
            f"{args.output_dir}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-sizeには1以上を指定してください"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを使用できません"
        )


def import_official_modules(
    reference_root: Path,
):
    """固定した公式実装から必要な機能を読み込む。"""
    reference_path = str(
        reference_root.resolve()
    )

    if reference_path not in sys.path:
        sys.path.insert(
            0,
            reference_path,
        )

    from dataset import (
        Shuttlecock_Trajectory_Dataset,
    )
    from test import predict_location
    from utils.general import (
        get_model,
        to_img,
    )
    from utils.metric import get_metric

    return (
        Shuttlecock_Trajectory_Dataset,
        get_model,
        get_metric,
        predict_location,
        to_img,
    )


def calculate_sha256(
    file_path: Path,
) -> str:
    """ファイル内容からSHA-256を計算する。"""
    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while True:
            chunk = input_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def get_git_commit(
    repository_root: Path,
) -> str:
    """指定したGitリポジトリのcommitを取得する。"""
    completed_process = subprocess.run(
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
        encoding="utf-8",
    )

    return completed_process.stdout.strip()


def load_val_mapping(
    mapping_csv: Path,
) -> dict[int, dict[str, str]]:
    """valのローカル番号と元フレームを読み込む。"""
    mapping = {}

    with mapping_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        required_columns = {
            "split",
            "local_frame",
            "source_frame",
            "source_image",
            "official_image",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "対応表にヘッダーがありません"
            )

        if not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "対応表に必要な列がありません: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            if row["split"] != "val":
                continue

            local_frame = int(
                row["local_frame"]
            )

            if local_frame in mapping:
                raise ValueError(
                    "val対応表に重複があります: "
                    f"{local_frame}"
                )

            mapping[local_frame] = row

    expected_keys = set(
        range(EXPECTED_VALIDATION_FRAMES)
    )

    if set(mapping.keys()) != expected_keys:
        missing = sorted(
            expected_keys - set(mapping.keys())
        )
        extra = sorted(
            set(mapping.keys()) - expected_keys
        )

        raise ValueError(
            "val対応表の番号が不正です: "
            f"missing={missing}, extra={extra}"
        )

    return mapping


def get_bbox_center(
    heatmap: np.ndarray,
    predict_location,
    to_img,
) -> tuple[int, int]:
    """公式処理でヒートマップの中心座標を得る。"""
    bbox = predict_location(
        to_img(heatmap)
    )

    center_x = int(
        bbox[0] + bbox[2] / 2
    )
    center_y = int(
        bbox[1] + bbox[3] / 2
    )

    return center_x, center_y


def classify_prediction(
    ground_truth_visible: bool,
    ground_truth_x: int,
    ground_truth_y: int,
    predicted_x: int,
    predicted_y: int,
    tolerance: float,
) -> tuple[str, float | None]:
    """公式定義と同じ5分類を返す。"""
    predicted_visible = not (
        predicted_x == 0
        and predicted_y == 0
    )

    if (
        not ground_truth_visible
        and not predicted_visible
    ):
        return "TN", None

    if (
        not ground_truth_visible
        and predicted_visible
    ):
        return "FP2", None

    if (
        ground_truth_visible
        and not predicted_visible
    ):
        return "FN", None

    distance = math.hypot(
        predicted_x - ground_truth_x,
        predicted_y - ground_truth_y,
    )

    if distance > tolerance:
        return "FP1", distance

    return "TP", distance


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """辞書の一覧をCSVへ保存する。"""
    if not rows:
        raise ValueError(
            "CSVへ保存する行がありません"
        )

    with output_path.open(
        "x",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    output_path: Path,
    value,
) -> None:
    """JSONを新規保存する。"""
    with output_path.open(
        "x",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            value,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")


def main() -> None:
    """Epoch 3のvalヒートマップを分析する。"""
    args = parse_args()

    args.dataset_root = (
        args.dataset_root.resolve()
    )
    args.mapping_csv = (
        args.mapping_csv.resolve()
    )
    args.reference_root = (
        args.reference_root.resolve()
    )
    args.checkpoint = (
        args.checkpoint.resolve()
    )
    args.output_dir = (
        args.output_dir.resolve()
    )

    validate_args(args)

    reference_commit = get_git_commit(
        args.reference_root
    )

    if reference_commit != EXPECTED_REFERENCE_COMMIT:
        raise ValueError(
            "公式参照commitが期待値と"
            "一致しません: "
            f"期待={EXPECTED_REFERENCE_COMMIT}, "
            f"実際={reference_commit}"
        )

    checkpoint_hash = calculate_sha256(
        args.checkpoint
    )

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpointのSHA-256が"
            "期待値と一致しません: "
            f"期待={EXPECTED_CHECKPOINT_SHA256}, "
            f"実際={checkpoint_hash}"
        )

    script_path = Path(
        __file__
    ).resolve()
    script_hash = calculate_sha256(
        script_path
    )
    volley_scope_commit = get_git_commit(
        PROJECT_ROOT
    )

    frame_mapping = load_val_mapping(
        args.mapping_csv
    )

    (
        dataset_class,
        get_model,
        get_metric,
        predict_location,
        to_img,
    ) = import_official_modules(
        args.reference_root
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_parameters = checkpoint[
        "param_dict"
    ]

    model_name = checkpoint_parameters[
        "model_name"
    ]
    sequence_length = checkpoint_parameters[
        "seq_len"
    ]
    background_mode = checkpoint_parameters[
        "bg_mode"
    ]
    tolerance = checkpoint_parameters[
        "tolerance"
    ]

    if model_name != EXPECTED_MODEL_NAME:
        raise ValueError(
            "モデル名が期待値と一致しません"
        )

    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            "系列長が期待値と一致しません"
        )

    if background_mode != EXPECTED_BACKGROUND_MODE:
        raise ValueError(
            "背景モードが期待値と一致しません"
        )

    if tolerance != EXPECTED_TOLERANCE:
        raise ValueError(
            "座標許容距離が期待値と一致しません"
        )

    validation_dataset = dataset_class(
        root_dir=str(args.dataset_root),
        split="val",
        seq_len=sequence_length,
        sliding_step=sequence_length,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    if (
        len(validation_dataset)
        != EXPECTED_VALIDATION_SEQUENCES
    ):
        raise ValueError(
            "val系列数が期待値と一致しません: "
            f"{len(validation_dataset)}"
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model = get_model(
        model_name,
        sequence_length,
        background_mode,
    )

    load_result = model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    frame_rows = []
    threshold_rows = []

    threshold_counts = {
        threshold: {
            "TP": 0,
            "TN": 0,
            "FP1": 0,
            "FP2": 0,
            "FN": 0,
        }
        for threshold in args.thresholds
    }

    processed_local_frames = set()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    with torch.inference_mode():
        for batch in validation_loader:
            (
                indices,
                model_input,
                target_heatmaps,
                _,
                _,
            ) = batch

            model_input = (
                model_input
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            predictions = model(
                model_input
            )

            if not torch.isfinite(
                predictions
            ).all().item():
                raise ValueError(
                    "予測にNaNまたは無限値があります"
                )

            indices_array = (
                indices
                .detach()
                .cpu()
                .numpy()
            )
            target_array = (
                target_heatmaps
                .detach()
                .cpu()
                .numpy()
            )
            prediction_array = (
                predictions
                .detach()
                .cpu()
                .numpy()
            )

            batch_size = (
                indices_array.shape[0]
            )

            for batch_index in range(
                batch_size
            ):
                for sequence_index in range(
                    sequence_length
                ):
                    local_frame = int(
                        indices_array[
                            batch_index,
                            sequence_index,
                            1,
                        ]
                    )

                    if (
                        local_frame
                        in processed_local_frames
                    ):
                        raise ValueError(
                            "valフレームが重複しました: "
                            f"{local_frame}"
                        )

                    processed_local_frames.add(
                        local_frame
                    )

                    mapping_row = frame_mapping[
                        local_frame
                    ]

                    target_heatmap = (
                        target_array[
                            batch_index,
                            sequence_index,
                        ]
                    )
                    predicted_heatmap = (
                        prediction_array[
                            batch_index,
                            sequence_index,
                        ]
                    )

                    ground_truth_visible = bool(
                        np.max(target_heatmap) > 0
                    )

                    if ground_truth_visible:
                        (
                            ground_truth_x,
                            ground_truth_y,
                        ) = get_bbox_center(
                            target_heatmap,
                            predict_location,
                            to_img,
                        )
                    else:
                        ground_truth_x = 0
                        ground_truth_y = 0

                    raw_flat_index = int(
                        np.argmax(
                            predicted_heatmap
                        )
                    )
                    (
                        raw_peak_y,
                        raw_peak_x,
                    ) = np.unravel_index(
                        raw_flat_index,
                        predicted_heatmap.shape,
                    )

                    raw_max = float(
                        predicted_heatmap[
                            raw_peak_y,
                            raw_peak_x,
                        ]
                    )

                    if ground_truth_visible:
                        raw_peak_distance = (
                            math.hypot(
                                raw_peak_x
                                - ground_truth_x,
                                raw_peak_y
                                - ground_truth_y,
                            )
                        )
                    else:
                        raw_peak_distance = None

                    official_classification = None
                    official_predicted_x = None
                    official_predicted_y = None
                    official_distance = None

                    for threshold in args.thresholds:
                        binary_heatmap = (
                            predicted_heatmap
                            > threshold
                        )

                        (
                            predicted_x,
                            predicted_y,
                        ) = get_bbox_center(
                            binary_heatmap,
                            predict_location,
                            to_img,
                        )

                        (
                            classification,
                            distance,
                        ) = classify_prediction(
                            ground_truth_visible,
                            ground_truth_x,
                            ground_truth_y,
                            predicted_x,
                            predicted_y,
                            tolerance,
                        )

                        threshold_counts[
                            threshold
                        ][classification] += 1

                        threshold_rows.append(
                            {
                                "threshold": (
                                    threshold
                                ),
                                "local_frame": (
                                    local_frame
                                ),
                                "source_frame": int(
                                    mapping_row[
                                        "source_frame"
                                    ]
                                ),
                                "ground_truth_visible": int(
                                    ground_truth_visible
                                ),
                                "ground_truth_x": (
                                    ground_truth_x
                                ),
                                "ground_truth_y": (
                                    ground_truth_y
                                ),
                                "predicted_visible": int(
                                    not (
                                        predicted_x == 0
                                        and predicted_y == 0
                                    )
                                ),
                                "predicted_x": (
                                    predicted_x
                                ),
                                "predicted_y": (
                                    predicted_y
                                ),
                                "distance": (
                                    distance
                                ),
                                "classification": (
                                    classification
                                ),
                                "raw_max": (
                                    raw_max
                                ),
                                "raw_peak_x": (
                                    int(raw_peak_x)
                                ),
                                "raw_peak_y": (
                                    int(raw_peak_y)
                                ),
                                "raw_peak_distance": (
                                    raw_peak_distance
                                ),
                            }
                        )

                        if (
                            threshold
                            == OFFICIAL_THRESHOLD
                        ):
                            official_classification = (
                                classification
                            )
                            official_predicted_x = (
                                predicted_x
                            )
                            official_predicted_y = (
                                predicted_y
                            )
                            official_distance = (
                                distance
                            )

                    if official_classification is None:
                        raise RuntimeError(
                            "公式しきい値の分類を"
                            "取得できませんでした"
                        )

                    fn_diagnostic = ""

                    if (
                        official_classification
                        == "FN"
                    ):
                        if (
                            raw_peak_distance
                            is not None
                            and raw_peak_distance
                            <= tolerance
                        ):
                            fn_diagnostic = (
                                "correct_raw_peak_"
                                "below_threshold"
                            )
                        else:
                            fn_diagnostic = (
                                "wrong_raw_peak_"
                                "below_threshold"
                            )

                    frame_rows.append(
                        {
                            "local_frame": (
                                local_frame
                            ),
                            "source_frame": int(
                                mapping_row[
                                    "source_frame"
                                ]
                            ),
                            "source_image": (
                                mapping_row[
                                    "source_image"
                                ]
                            ),
                            "official_image": (
                                mapping_row[
                                    "official_image"
                                ]
                            ),
                            "ground_truth_visible": int(
                                ground_truth_visible
                            ),
                            "ground_truth_x": (
                                ground_truth_x
                            ),
                            "ground_truth_y": (
                                ground_truth_y
                            ),
                            "raw_max": raw_max,
                            "raw_peak_x": int(
                                raw_peak_x
                            ),
                            "raw_peak_y": int(
                                raw_peak_y
                            ),
                            "raw_peak_distance": (
                                raw_peak_distance
                            ),
                            "official_classification": (
                                official_classification
                            ),
                            "official_predicted_x": (
                                official_predicted_x
                            ),
                            "official_predicted_y": (
                                official_predicted_y
                            ),
                            "official_distance": (
                                official_distance
                            ),
                            "fn_diagnostic": (
                                fn_diagnostic
                            ),
                        }
                    )

    torch.cuda.synchronize()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    expected_local_frames = set(
        range(EXPECTED_VALIDATION_FRAMES)
    )

    if (
        processed_local_frames
        != expected_local_frames
    ):
        raise ValueError(
            "処理したvalフレームに"
            "欠落または余分があります"
        )

    frame_rows.sort(
        key=lambda row: row["local_frame"]
    )
    threshold_rows.sort(
        key=lambda row: (
            row["threshold"],
            row["local_frame"],
        )
    )

    visible_count = sum(
        row["ground_truth_visible"]
        for row in frame_rows
    )
    invisible_count = (
        len(frame_rows) - visible_count
    )

    if visible_count != EXPECTED_VISIBLE_FRAMES:
        raise ValueError(
            "ボールありフレーム数が"
            "期待値と一致しません: "
            f"{visible_count}"
        )

    if invisible_count != EXPECTED_INVISIBLE_FRAMES:
        raise ValueError(
            "ボールなしフレーム数が"
            "期待値と一致しません: "
            f"{invisible_count}"
        )

    threshold_summary_rows = []

    for threshold in args.thresholds:
        counts = threshold_counts[
            threshold
        ]

        total = sum(
            counts.values()
        )

        if total != EXPECTED_VALIDATION_FRAMES:
            raise ValueError(
                "しきい値別分類数が"
                "120ではありません: "
                f"threshold={threshold}, "
                f"total={total}"
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

        threshold_summary_rows.append(
            {
                "threshold": threshold,
                "TP": counts["TP"],
                "TN": counts["TN"],
                "FP1": counts["FP1"],
                "FP2": counts["FP2"],
                "FN": counts["FN"],
                "accuracy": float(
                    accuracy
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(f1),
                "miss_rate": float(
                    miss_rate
                ),
            }
        )

    official_summary = next(
        row
        for row in threshold_summary_rows
        if row["threshold"]
        == OFFICIAL_THRESHOLD
    )

    expected_official_counts = {
        "TP": 38,
        "TN": 1,
        "FP1": 0,
        "FP2": 1,
        "FN": 80,
    }

    for key, expected_value in (
        expected_official_counts.items()
    ):
        if (
            official_summary[key]
            != expected_value
        ):
            raise ValueError(
                "公式しきい値0.5の結果を"
                "再現できません: "
                f"{key}="
                f"{official_summary[key]}, "
                f"期待={expected_value}"
            )

    fn_rows = [
        row
        for row in frame_rows
        if (
            row["official_classification"]
            == "FN"
        )
    ]

    fn_diagnostic_counts = {
        "correct_raw_peak_below_threshold": 0,
        "wrong_raw_peak_below_threshold": 0,
    }

    for row in fn_rows:
        fn_diagnostic_counts[
            row["fn_diagnostic"]
        ] += 1

    peak_values = [
        row["raw_max"]
        for row in frame_rows
        if row["ground_truth_visible"] == 1
    ]

    analysis_summary = {
        "schema_version": 1,
        "analysis_name": (
            "tracknet_epoch_003_"
            "val_threshold_analysis"
        ),
        "volley_scope_commit": (
            volley_scope_commit
        ),
        "analysis_script": str(
            script_path
        ),
        "analysis_script_sha256": (
            script_hash
        ),
        "reference_root": str(
            args.reference_root
        ),
        "reference_commit": (
            reference_commit
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_sha256": (
            checkpoint_hash
        ),
        "dataset_root": str(
            args.dataset_root
        ),
        "mapping_csv": str(
            args.mapping_csv
        ),
        "thresholds": (
            args.thresholds
        ),
        "official_threshold": (
            OFFICIAL_THRESHOLD
        ),
        "tolerance": tolerance,
        "evaluated_frames": (
            len(frame_rows)
        ),
        "visible_frames": (
            visible_count
        ),
        "invisible_frames": (
            invisible_count
        ),
        "official_result": (
            official_summary
        ),
        "official_fn_count": (
            len(fn_rows)
        ),
        "fn_diagnostic_counts": (
            fn_diagnostic_counts
        ),
        "visible_raw_max": {
            "minimum": float(
                np.min(peak_values)
            ),
            "median": float(
                np.median(peak_values)
            ),
            "maximum": float(
                np.max(peak_values)
            ),
        },
        "threshold_summary": (
            threshold_summary_rows
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024**2
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024**2
        ),
        "missing_keys": len(
            load_result.missing_keys
        ),
        "unexpected_keys": len(
            load_result.unexpected_keys
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        args.output_dir
        / "frame_peaks.csv",
        frame_rows,
    )
    write_csv(
        args.output_dir
        / "per_frame_thresholds.csv",
        threshold_rows,
    )
    write_csv(
        args.output_dir
        / "threshold_summary.csv",
        threshold_summary_rows,
    )
    write_json(
        args.output_dir
        / "analysis.json",
        analysis_summary,
    )

    print("ヒートマップしきい値分析が完了しました")
    print(
        "checkpoint SHA-256: "
        f"{checkpoint_hash}"
    )
    print(
        "分析スクリプトSHA-256: "
        f"{script_hash}"
    )
    print(
        "評価フレーム数: "
        f"{len(frame_rows)}"
    )
    print(
        "FN分析対象: "
        f"{len(fn_rows)}"
    )

    print("=== しきい値別結果 ===")

    for row in threshold_summary_rows:
        print(
            "threshold="
            f"{row['threshold']:.2f}, "
            f"TP={row['TP']}, "
            f"TN={row['TN']}, "
            f"FP1={row['FP1']}, "
            f"FP2={row['FP2']}, "
            f"FN={row['FN']}, "
            "accuracy="
            f"{row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"f1={row['f1']:.4f}"
        )

    print("=== FNの生ピーク診断 ===")

    for key, value in (
        fn_diagnostic_counts.items()
    ):
        print(f"{key}: {value}")

    print(
        "出力先: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()