import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from convert_yolo_to_tracknet_csv import round_half_up
from evaluate_pretrained_model import (
    IMAGE_DIR,
    LABEL_DIR,
    SPORTS_BALL_CLASS_ID,
    calculate_iou,
    load_ground_truths,
    safe_divide,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_DIR = IMAGE_DIR
DEFAULT_LABEL_DIR = LABEL_DIR

DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "yolo11n.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_common_center"
    / "ultralytics_8_4_105"
)

EXPECTED_WEIGHT_SHA256 = (
    "0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1"
)

EXPECTED_IMAGE_COUNT = 150
EXPECTED_POSITIVE_COUNT = 104
EXPECTED_NEGATIVE_COUNT = 46

HEATMAP_WIDTH = 512
HEATMAP_HEIGHT = 288
CENTER_TOLERANCE = 4.0
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

EXPERIMENTS = (
    {
        "name": "conf025_img960",
        "image_size": 960,
        "expected_tp": 27,
        "expected_fp": 1,
        "expected_fn": 77,
    },
    {
        "name": "conf025_img1280",
        "image_size": 1280,
        "expected_tp": 32,
        "expected_fp": 3,
        "expected_fn": 72,
    },
)

def parse_sha256(value: str) -> str:
    """SHA-256を64桁の16進数として検証する。"""
    normalized_value = value.strip().upper()

    if (
        len(normalized_value) != 64
        or any(
            character not in "0123456789ABCDEF"
            for character in normalized_value
        )
    ):
        raise argparse.ArgumentTypeError(
            "SHA-256は64桁の16進数で指定してください"
        )

    return normalized_value


def parse_image_sizes(value: str) -> list[int]:
    """カンマ区切りの画像サイズを検証する。"""
    try:
        image_sizes = [
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "画像サイズを整数へ変換できません"
        ) from error

    if not image_sizes:
        raise argparse.ArgumentTypeError(
            "画像サイズを1つ以上指定してください"
        )

    if any(image_size <= 0 for image_size in image_sizes):
        raise argparse.ArgumentTypeError(
            "画像サイズは正の整数にしてください"
        )

    return image_sizes

def parse_args() -> argparse.Namespace:
    """共通中心距離評価の入出力を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "YOLO11nを既存IoU基準と"
            "TrackNet共通中心距離基準で評価する"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="YOLOへ入力する評価画像フォルダ",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Label Studioから出力したYOLO正解ラベル",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="評価するYOLO重み",
    )
    parser.add_argument(
        "--expected-weights-sha256",
        type=parse_sha256,
        default=EXPECTED_WEIGHT_SHA256,
        help=(
            "評価対象の重みに期待するSHA-256。"
            "既定値はCOCO事前学習済みYOLO11n"
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="推論時のconfidenceしきい値",
    )
    parser.add_argument(
        "--image-sizes",
        type=parse_image_sizes,
        default=parse_image_sizes("960,1280"),
        help="カンマ区切りの入力画像サイズ",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=SPORTS_BALL_CLASS_ID,
        help=(
            "検出対象のクラスID。"
            "COCO sports ballは32、"
            "追加学習済み単一ballモデルは0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="評価結果の新規出力先",
    )

    return parser.parse_args()


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


def validate_inputs(
    image_dir: Path,
    label_dir: Path,
    weights: Path,
    expected_weights_sha256: str,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """画像、ラベル、重み、出力先を検証する。"""
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    weights = weights.resolve()
    output_dir = output_dir.resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            "評価画像フォルダがありません: "
            f"{image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            "正解ラベルフォルダがありません: "
            f"{label_dir}"
        )

    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO重みがありません: {weights}"
        )

    weight_hash = calculate_sha256(weights)

    if weight_hash != expected_weights_sha256:
        raise RuntimeError(
            "YOLO重みのSHA-256が"
            "期待値と異なります: "
            f"期待={expected_weights_sha256}, "
            f"実際={weight_hash}"
        )

    if output_dir.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{output_dir}"
        )

    return (
        image_dir,
        label_dir,
        weights,
        output_dir,
    )


def normalized_center_to_heatmap(
    x_normalized: float,
    y_normalized: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """正規化中心をTrackNetと同じ整数座標へ戻す。"""
    x_original = round_half_up(
        x_normalized * image_width
    )
    y_original = round_half_up(
        y_normalized * image_height
    )

    width_scaler = (
        image_width / HEATMAP_WIDTH
    )
    height_scaler = (
        image_height / HEATMAP_HEIGHT
    )

    x_heatmap = int(
        x_original / width_scaler
    )
    y_heatmap = int(
        y_original / height_scaler
    )

    return x_heatmap, y_heatmap


def prediction_center_to_heatmap(
    prediction_box_pixels: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """YOLO予測矩形中心を512×288へ変換する。"""
    x_center = (
        prediction_box_pixels[0]
        + prediction_box_pixels[2]
    ) / 2
    y_center = (
        prediction_box_pixels[1]
        + prediction_box_pixels[3]
    ) / 2

    width_scaler = (
        image_width / HEATMAP_WIDTH
    )
    height_scaler = (
        image_height / HEATMAP_HEIGHT
    )

    x_heatmap = int(
        x_center / width_scaler
    )
    y_heatmap = int(
        y_center / height_scaler
    )

    return x_heatmap, y_heatmap


def normalize_prediction_box(
    prediction_box_pixels: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """YOLO予測矩形を0〜1へ正規化する。"""
    return [
        prediction_box_pixels[0]
        / image_width,
        prediction_box_pixels[1]
        / image_height,
        prediction_box_pixels[2]
        / image_width,
        prediction_box_pixels[3]
        / image_height,
    ]


def calculate_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """Precision・Recall・F1を計算する。"""
    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_experiment(
    model: YOLO,
    image_paths: list[Path],
    ground_truths: dict[
        str,
        list[float] | None,
    ],
    experiment: dict,
    confidence_threshold: float,
    class_id: int,
    verify_historical_baseline: bool,
) -> tuple[dict, list[dict], list[dict]]:
    """1画像ずつ推論して2種類の基準で評価する。"""
    image_size = int(
        experiment["image_size"]
    )
    experiment_name = str(
        experiment["name"]
    )

    formal_counts = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN_images": 0,
    }
    center_counts = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN_images": 0,
    }

    matched_ious = []
    per_image_rows = []
    prediction_rows = []

    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            imgsz=image_size,
            classes=[class_id],
            device=0,
            batch=1,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(
                "1画像に対するResult数が"
                "1ではありません: "
                f"{image_path}"
            )

        result = results[0]
        image_height, image_width = (
            result.orig_shape
        )
        ground_truth_box = ground_truths[
            image_path.stem
        ]

        prediction_boxes_pixels = []
        prediction_confidences = []

        if result.boxes is not None:
            prediction_boxes_pixels = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .tolist()
            )
            prediction_confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )

        prediction_count = len(
            prediction_boxes_pixels
        )

        if prediction_count != len(
            prediction_confidences
        ):
            raise RuntimeError(
                "予測矩形とconfidenceの"
                "件数が一致しません"
            )

        prediction_boxes_normalized = [
            normalize_prediction_box(
                prediction_box,
                image_width,
                image_height,
            )
            for prediction_box
            in prediction_boxes_pixels
        ]

        prediction_centers = [
            prediction_center_to_heatmap(
                prediction_box,
                image_width,
                image_height,
            )
            for prediction_box
            in prediction_boxes_pixels
        ]

        best_iou = 0.0
        minimum_center_distance = None
        best_center_prediction_index = None
        formal_classification = ""
        center_classification = ""

        if ground_truth_box is None:
            ground_truth_visible = 0
            ground_truth_x = 0
            ground_truth_y = 0

            if prediction_count == 0:
                formal_counts["TN_images"] += 1
                center_counts["TN_images"] += 1
                formal_classification = "TN"
                center_classification = "TN"
            else:
                formal_counts["FP"] += (
                    prediction_count
                )
                center_counts["FP"] += (
                    prediction_count
                )
                formal_classification = "FP"
                center_classification = "FP"
        else:
            ground_truth_visible = 1
            ground_truth_center_x = (
                ground_truth_box[0]
                + ground_truth_box[2]
            ) / 2
            ground_truth_center_y = (
                ground_truth_box[1]
                + ground_truth_box[3]
            ) / 2

            (
                ground_truth_x,
                ground_truth_y,
            ) = normalized_center_to_heatmap(
                ground_truth_center_x,
                ground_truth_center_y,
                image_width,
                image_height,
            )

            if prediction_count == 0:
                formal_counts["FN"] += 1
                center_counts["FN"] += 1
                formal_classification = "FN"
                center_classification = "FN"
            else:
                ious = [
                    calculate_iou(
                        ground_truth_box,
                        prediction_box,
                    )
                    for prediction_box
                    in prediction_boxes_normalized
                ]
                best_iou = max(ious)

                if best_iou >= IOU_THRESHOLD:
                    formal_counts["TP"] += 1
                    formal_counts["FP"] += (
                        prediction_count - 1
                    )
                    matched_ious.append(best_iou)
                    formal_classification = "TP"
                else:
                    formal_counts["FN"] += 1
                    formal_counts["FP"] += (
                        prediction_count
                    )
                    formal_classification = "FP+FN"

                center_distances = [
                    float(
                        np.hypot(
                            prediction_x
                            - ground_truth_x,
                            prediction_y
                            - ground_truth_y,
                        )
                    )
                    for (
                        prediction_x,
                        prediction_y,
                    ) in prediction_centers
                ]

                minimum_center_distance = min(
                    center_distances
                )
                best_center_prediction_index = (
                    center_distances.index(
                        minimum_center_distance
                    )
                )

                if (
                    minimum_center_distance
                    <= CENTER_TOLERANCE
                ):
                    center_counts["TP"] += 1
                    center_counts["FP"] += (
                        prediction_count - 1
                    )
                    center_classification = "TP"
                else:
                    center_counts["FN"] += 1
                    center_counts["FP"] += (
                        prediction_count
                    )
                    center_classification = "FP+FN"

        for prediction_index, (
            prediction_box_pixels,
            prediction_box_normalized,
            prediction_center,
            prediction_confidence,
        ) in enumerate(
            zip(
                prediction_boxes_pixels,
                prediction_boxes_normalized,
                prediction_centers,
                prediction_confidences,
            )
        ):
            prediction_iou = None
            prediction_distance = None

            if ground_truth_box is not None:
                prediction_iou = calculate_iou(
                    ground_truth_box,
                    prediction_box_normalized,
                )
                prediction_distance = float(
                    np.hypot(
                        prediction_center[0]
                        - ground_truth_x,
                        prediction_center[1]
                        - ground_truth_y,
                    )
                )

            prediction_rows.append(
                {
                    "experiment": experiment_name,
                    "image_name": image_path.name,
                    "prediction_index": (
                        prediction_index
                    ),
                    "confidence": float(
                        prediction_confidence
                    ),
                    "x1_original": (
                        prediction_box_pixels[0]
                    ),
                    "y1_original": (
                        prediction_box_pixels[1]
                    ),
                    "x2_original": (
                        prediction_box_pixels[2]
                    ),
                    "y2_original": (
                        prediction_box_pixels[3]
                    ),
                    "center_x_heatmap": (
                        prediction_center[0]
                    ),
                    "center_y_heatmap": (
                        prediction_center[1]
                    ),
                    "iou": prediction_iou,
                    "center_distance": (
                        prediction_distance
                    ),
                    "best_center_prediction": int(
                        prediction_index
                        == best_center_prediction_index
                    ),
                }
            )

        per_image_rows.append(
            {
                "experiment": experiment_name,
                "image_name": image_path.name,
                "ground_truth_visible": (
                    ground_truth_visible
                ),
                "ground_truth_x_heatmap": (
                    ground_truth_x
                ),
                "ground_truth_y_heatmap": (
                    ground_truth_y
                ),
                "prediction_count": (
                    prediction_count
                ),
                "best_iou": best_iou,
                "minimum_center_distance": (
                    minimum_center_distance
                ),
                "formal_classification": (
                    formal_classification
                ),
                "center_classification": (
                    center_classification
                ),
            }
        )

        if (
            image_index == 1
            or image_index % 25 == 0
            or image_index == len(image_paths)
        ):
            print(
                f"{experiment_name}: "
                f"{image_index}/"
                f"{len(image_paths)}"
            )

    torch.cuda.synchronize()
    elapsed_seconds = (
        perf_counter() - start_time
    )

    formal_metrics = calculate_metrics(
        formal_counts["TP"],
        formal_counts["FP"],
        formal_counts["FN"],
    )
    center_metrics = calculate_metrics(
        center_counts["TP"],
        center_counts["FP"],
        center_counts["FN"],
    )

    if verify_historical_baseline:
        formal_reproduced = (
            formal_counts["TP"]
            == experiment["expected_tp"]
            and formal_counts["FP"]
            == experiment["expected_fp"]
            and formal_counts["FN"]
            == experiment["expected_fn"]
        )
    else:
        formal_reproduced = None

    summary = {
        "experiment": experiment_name,
        "confidence_threshold": (
            confidence_threshold
        ),
        "image_size": image_size,
        "center_tolerance": (
            CENTER_TOLERANCE
        ),
        "formal_TP": formal_counts["TP"],
        "formal_FP": formal_counts["FP"],
        "formal_FN": formal_counts["FN"],
        "formal_TN_images": (
            formal_counts["TN_images"]
        ),
        "formal_precision": (
            formal_metrics["precision"]
        ),
        "formal_recall": (
            formal_metrics["recall"]
        ),
        "formal_f1": formal_metrics["f1"],
        "average_matched_iou": (
            safe_divide(
                sum(matched_ious),
                len(matched_ious),
            )
        ),
        "formal_reproduced": (
            formal_reproduced
        ),
        "center_TP": center_counts["TP"],
        "center_FP": center_counts["FP"],
        "center_FN": center_counts["FN"],
        "center_TN_images": (
            center_counts["TN_images"]
        ),
        "center_precision": (
            center_metrics["precision"]
        ),
        "center_recall": (
            center_metrics["recall"]
        ),
        "center_f1": center_metrics["f1"],
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024**2
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024**2
        ),
    }

    return (
        summary,
        per_image_rows,
        prediction_rows,
    )


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """辞書行をCSVへ保存する。"""
    if not rows:
        raise ValueError(
            f"保存する行がありません: "
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
    """960・1280の正式指標と共通指標を評価する。"""
    args = parse_args()
    (
        image_dir,
        label_dir,
        weights,
        output_dir,
    ) = validate_inputs(
        args.image_dir,
        args.label_dir,
        args.weights,
        args.expected_weights_sha256,
        args.output_dir,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPUを利用できません"
        )
    if not 0 < args.confidence_threshold < 1:
        raise ValueError(
            "--confidence-thresholdは"
            "0より大きく1より小さくしてください"
        )

    if args.class_id < 0:
        raise ValueError(
            "--class-idは0以上にしてください"
        )
    verify_historical_baseline = (
        image_dir
        == DEFAULT_IMAGE_DIR.resolve()
        and label_dir
        == DEFAULT_LABEL_DIR.resolve()
        and weights
        == DEFAULT_WEIGHTS.resolve()
        and args.expected_weights_sha256
        == EXPECTED_WEIGHT_SHA256
        and args.confidence_threshold
        == CONFIDENCE_THRESHOLD
        and args.class_id
        == SPORTS_BALL_CLASS_ID
        and args.image_sizes == [960, 1280]
    )

    if verify_historical_baseline:
        experiments = EXPERIMENTS
    else:
        confidence_label = (
            f"{args.confidence_threshold:.2f}"
            .replace(".", "")
        )
        experiments = tuple(
            {
                "name": (
                    f"conf{confidence_label}_"
                    f"img{image_size}"
                ),
                "image_size": image_size,
            }
            for image_size in args.image_sizes
        )

    supported_image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }
    image_paths = sorted(
        image_path
        for image_path in image_dir.iterdir()
        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in supported_image_extensions
        )
    )

    if not image_paths:
        raise RuntimeError(
            "評価画像が見つかりません: "
            f"{image_dir}"
        )

    ground_truths = load_ground_truths(
        label_dir
    )

    image_stems = {
        image_path.stem
        for image_path in image_paths
    }
    ground_truth_stems = set(
        ground_truths
    )

    missing_label_stems = sorted(
        image_stems - ground_truth_stems
    )
    unexpected_label_stems = sorted(
        ground_truth_stems - image_stems
    )

    if missing_label_stems:
        raise ValueError(
            "正解ラベルがない評価画像があります: "
            f"{missing_label_stems}"
        )

    if unexpected_label_stems:
        raise ValueError(
            "対応画像がない正解ラベルがあります: "
            f"{unexpected_label_stems}"
        )

    positive_count = sum(
        ground_truth is not None
        for ground_truth
        in ground_truths.values()
    )
    negative_count = (
        len(ground_truths) - positive_count
    )

    if verify_historical_baseline:
        if len(image_paths) != EXPECTED_IMAGE_COUNT:
            raise ValueError(
                "従来評価画像数が想定と"
                "一致しません: "
                f"{len(image_paths)}"
            )

        if (
            positive_count
            != EXPECTED_POSITIVE_COUNT
        ):
            raise ValueError(
                "従来評価の正例数が想定と"
                "一致しません: "
                f"{positive_count}"
            )

        if (
            negative_count
            != EXPECTED_NEGATIVE_COUNT
        ):
            raise ValueError(
                "従来評価の負例数が想定と"
                "一致しません: "
                f"{negative_count}"
            )

    model = YOLO(str(weights))
    summaries = []
    all_per_image_rows = []
    all_prediction_rows = []

    for experiment in experiments:
        (
            summary,
            per_image_rows,
            prediction_rows,
        ) = evaluate_experiment(
            model,
            image_paths,
            ground_truths,
            experiment,
            args.confidence_threshold,
            args.class_id,
            verify_historical_baseline,
        )

        summaries.append(summary)
        all_per_image_rows.extend(
            per_image_rows
        )
        all_prediction_rows.extend(
            prediction_rows
        )

        reproduction_result = (
            summary["formal_reproduced"]
        )

        if reproduction_result is None:
            reproduction_text = "対象外"
        else:
            reproduction_text = str(
                reproduction_result
            )

        print(
            f"{summary['experiment']} "
            f"formal reproduced: "
            f"{reproduction_text}"
        )

    if (
        verify_historical_baseline
        and not all(
            summary["formal_reproduced"] is True
            for summary in summaries
        )
    ):
        raise RuntimeError(
            "既存IoU評価を再現できなかったため、"
            "共通比較結果を保存しません"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output_dir / "summary.csv",
        summaries,
    )
    write_csv(
        output_dir / "per_image.csv",
        all_per_image_rows,
    )

    if all_prediction_rows:
        write_csv(
            output_dir / "predictions.csv",
            all_prediction_rows,
        )

    script_path = Path(__file__).resolve()

    analysis = {
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "torch_version": torch.__version__,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "gpu": torch.cuda.get_device_name(0),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "historical_reproduction_check": (
            verify_historical_baseline
        ),
        "weights": str(weights),
        "weights_sha256": (
            calculate_sha256(weights)
        ),
        "script_sha256": (
            calculate_sha256(script_path)
        ),
        "image_count": len(image_paths),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "class_id": args.class_id,
        "confidence_threshold": (
            args.confidence_threshold
        ),
        "image_sizes": args.image_sizes,
        "expected_weights_sha256": (
            args.expected_weights_sha256
        ),
        "formal_iou_threshold": (
            IOU_THRESHOLD
        ),
        "common_center_tolerance": (
            CENTER_TOLERANCE
        ),
        "heatmap_width": HEATMAP_WIDTH,
        "heatmap_height": HEATMAP_HEIGHT,
        "summaries": summaries,
    }

    with (
        output_dir / "analysis.json"
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
        "YOLO共通中心距離評価が"
        "完了しました"
    )

    for summary in summaries:
        print(
            f"{summary['experiment']}: "
            f"Center TP="
            f"{summary['center_TP']}, "
            f"FP={summary['center_FP']}, "
            f"FN={summary['center_FN']}, "
            f"Precision="
            f"{summary['center_precision']:.4f}, "
            f"Recall="
            f"{summary['center_recall']:.4f}, "
            f"F1="
            f"{summary['center_f1']:.4f}"
        )

    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()