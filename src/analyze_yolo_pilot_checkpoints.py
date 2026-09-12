import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from evaluate_pretrained_model import (
    load_ground_truths,
    safe_divide,
)
from evaluate_yolo_common_center import (
    CENTER_TOLERANCE,
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    calculate_sha256,
    normalized_center_to_heatmap,
    prediction_center_to_heatmap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_pilot_v2"
    / "images"
    / "val"
)

DEFAULT_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_pilot_v2"
    / "labels"
    / "val"
)

DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_training"
    / (
        "pilot_v2_transfer_img1280_bs2_"
        "lr1e-4_seed13_epochs20"
    )
    / "weights"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_checkpoint_analysis"
    / "pilot_v2_val_img1280_epochs16_to_20"
)

CHECKPOINTS = (
    {
        "training_epoch": 16,
        "file_name": "epoch15.pt",
        "expected_sha256": (
            "8AB80222D4A5B45A4508124F9E425674"
            "EC065287AD0463AC07B29CAE6EFFC98C"
        ),
    },
    {
        "training_epoch": 17,
        "file_name": "epoch16.pt",
        "expected_sha256": (
            "E463C63BC3DB8EC5E8BB7FBA91FE424B"
            "A36DAF41692F4403D8B5BFA1C8CCB70E"
        ),
    },
    {
        "training_epoch": 18,
        "file_name": "epoch17.pt",
        "expected_sha256": (
            "40E203223ACF1C9EC91DDDED44081C965"
            "8E1301A77EADFC2F26BC8E1519D9973"
        ),
    },
    {
        "training_epoch": 19,
        "file_name": "epoch18.pt",
        "expected_sha256": (
            "4D5E1E49EF90F8800CB7D43949733E01"
            "F9DE29A8B4298355AA041EB666350595"
        ),
    },
    {
        "training_epoch": 20,
        "file_name": "epoch19.pt",
        "expected_sha256": (
            "D17221BC9F6EB148D640E6A1EC08722B"
            "C7CCCE1F0F1B702309DD23FCED9EC4E9"
        ),
    },
)

DEFAULT_THRESHOLDS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)

IMAGE_SIZE = 1280
BALL_CLASS_ID = 0
EXPECTED_IMAGE_COUNT = 120
EXPECTED_POSITIVE_COUNT = 118
EXPECTED_NEGATIVE_COUNT = 2


def parse_args() -> argparse.Namespace:
    """val選定分析の入出力としきい値を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "YOLO追加学習checkpointとconfidenceを、"
            "valの共通中心距離基準で選定する"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="YOLOパイロットval画像フォルダ",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="YOLOパイロットvalラベルフォルダ",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="epoch0.ptからepoch2.ptを含むフォルダ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="分析結果の新規出力先",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(
            f"{threshold:.2f}"
            for threshold in DEFAULT_THRESHOLDS
        ),
        help="カンマ区切りのconfidence候補",
    )
    return parser.parse_args()

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

def parse_thresholds(value: str) -> list[float]:
    """カンマ区切りのconfidence候補を検証する。"""
    try:
        thresholds = [
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        ]
    except ValueError as error:
        raise ValueError(
            "confidence候補を数値として読み込めません"
        ) from error

    if not thresholds:
        raise ValueError(
            "confidence候補を1つ以上指定してください"
        )

    if any(
        threshold <= 0.0 or threshold > 1.0
        for threshold in thresholds
    ):
        raise ValueError(
            "confidence候補は0より大きく1以下にしてください"
        )

    if len(thresholds) != len(set(thresholds)):
        raise ValueError(
            "confidence候補に重複があります"
        )

    return sorted(thresholds)


def collect_image_paths(
    image_dir: Path,
) -> list[Path]:
    """評価画像を名前順に取得する。"""
    supported_extensions = {
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
            in supported_extensions
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"val画像が見つかりません: {image_dir}"
        )

    return image_paths


def validate_inputs(
    image_dir: Path,
    label_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    thresholds: list[float],
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    list[Path],
    dict[str, list[float] | None],
    list[dict],
]:
    """入力データ、checkpoint、出力先を検証する。"""
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"val画像フォルダがありません: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"valラベルフォルダがありません: {label_dir}"
        )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "checkpointフォルダがありません: "
            f"{checkpoint_dir}"
        )

    if output_dir.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{output_dir}"
        )

    if min(thresholds) < 0.001:
        raise ValueError(
            "最小confidenceは0.001以上にしてください"
        )

    image_paths = collect_image_paths(image_dir)
    ground_truths = load_ground_truths(label_dir)

    image_stems = {
        image_path.stem
        for image_path in image_paths
    }
    ground_truth_stems = set(ground_truths)

    missing_label_stems = sorted(
        image_stems - ground_truth_stems
    )
    unexpected_label_stems = sorted(
        ground_truth_stems - image_stems
    )

    if missing_label_stems:
        raise ValueError(
            "正解ラベルがないval画像があります: "
            f"{missing_label_stems}"
        )

    if unexpected_label_stems:
        raise ValueError(
            "対応画像がないvalラベルがあります: "
            f"{unexpected_label_stems}"
        )

    positive_count = sum(
        ground_truth is not None
        for ground_truth in ground_truths.values()
    )
    negative_count = (
        len(ground_truths) - positive_count
    )

    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            "val画像数が想定と一致しません: "
            f"{len(image_paths)}"
        )

    if positive_count != EXPECTED_POSITIVE_COUNT:
        raise ValueError(
            "val正例数が想定と一致しません: "
            f"{positive_count}"
        )

    if negative_count != EXPECTED_NEGATIVE_COUNT:
        raise ValueError(
            "val負例数が想定と一致しません: "
            f"{negative_count}"
        )

    checkpoint_records = []

    for checkpoint in CHECKPOINTS:
        checkpoint_path = (
            checkpoint_dir
            / str(checkpoint["file_name"])
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "checkpointがありません: "
                f"{checkpoint_path}"
            )

        actual_hash = calculate_sha256(
            checkpoint_path
        )
        expected_hash = str(
            checkpoint["expected_sha256"]
        )

        if actual_hash != expected_hash:
            raise RuntimeError(
                "checkpointのSHA-256が"
                "固定値と異なります: "
                f"{checkpoint_path}, "
                f"actual={actual_hash}"
            )

        checkpoint_records.append(
            {
                "training_epoch": int(
                    checkpoint["training_epoch"]
                ),
                "file_name": str(
                    checkpoint["file_name"]
                ),
                "path": checkpoint_path,
                "sha256": actual_hash,
            }
        )

    return (
        image_dir,
        label_dir,
        checkpoint_dir,
        output_dir,
        image_paths,
        ground_truths,
        checkpoint_records,
    )


def run_inference(
    checkpoint_record: dict,
    image_paths: list[Path],
    inference_confidence: float,
) -> tuple[
    dict[str, dict],
    list[dict],
    dict,
]:
    """1 checkpointを最小confidenceで1回推論する。"""
    checkpoint_path = Path(
        checkpoint_record["path"]
    )
    training_epoch = int(
        checkpoint_record["training_epoch"]
    )

    print()
    print(
        f"=== 学習Epoch {training_epoch}の推論 ==="
    )
    print(f"checkpoint: {checkpoint_path.name}")
    print(
        "推論時の最小confidence: "
        f"{inference_confidence:.2f}"
    )

    model = YOLO(str(checkpoint_path))
    predictions_by_image = {}
    prediction_rows = []

    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):
        results = model.predict(
            source=str(image_path),
            conf=inference_confidence,
            imgsz=IMAGE_SIZE,
            classes=[BALL_CLASS_ID],
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
        image_predictions = []

        if result.boxes is not None:
            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .tolist()
            )
            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )
            class_ids = (
                result.boxes.cls
                .detach()
                .cpu()
                .tolist()
            )

            if not (
                len(boxes)
                == len(confidences)
                == len(class_ids)
            ):
                raise RuntimeError(
                    "予測矩形、confidence、"
                    "クラスIDの件数が一致しません"
                )

            for prediction_index, (
                box,
                confidence,
                class_id,
            ) in enumerate(
                zip(
                    boxes,
                    confidences,
                    class_ids,
                )
            ):
                integer_class_id = int(
                    class_id
                )

                if integer_class_id != BALL_CLASS_ID:
                    raise RuntimeError(
                        "ball以外の予測が含まれています: "
                        f"class_id={integer_class_id}"
                    )

                center_x, center_y = (
                    prediction_center_to_heatmap(
                        box,
                        image_width,
                        image_height,
                    )
                )

                prediction = {
                    "prediction_index": (
                        prediction_index
                    ),
                    "confidence": float(
                        confidence
                    ),
                    "class_id": integer_class_id,
                    "x1_original": float(box[0]),
                    "y1_original": float(box[1]),
                    "x2_original": float(box[2]),
                    "y2_original": float(box[3]),
                    "center_x_heatmap": center_x,
                    "center_y_heatmap": center_y,
                }
                image_predictions.append(
                    prediction
                )

                prediction_rows.append(
                    {
                        "training_epoch": (
                            training_epoch
                        ),
                        "checkpoint_file": (
                            checkpoint_path.name
                        ),
                        "image_name": (
                            image_path.name
                        ),
                        **prediction,
                    }
                )

        predictions_by_image[
            image_path.stem
        ] = {
            "image_name": image_path.name,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "predictions": image_predictions,
        }

        if (
            image_number == 1
            or image_number % 25 == 0
            or image_number == len(image_paths)
        ):
            print(
                f"学習Epoch {training_epoch}: "
                f"{image_number}/"
                f"{len(image_paths)}"
            )

    torch.cuda.synchronize()
    elapsed_seconds = (
        perf_counter() - start_time
    )

    runtime = {
        "training_epoch": training_epoch,
        "checkpoint_file": (
            checkpoint_path.name
        ),
        "checkpoint_sha256": (
            checkpoint_record["sha256"]
        ),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024
            / 1024
        ),
    }

    del model
    torch.cuda.empty_cache()

    return (
        predictions_by_image,
        prediction_rows,
        runtime,
    )


def calculate_metrics(
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """共通中心距離評価の指標を計算する。"""
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
    accuracy = safe_divide(
        true_positive + true_negative,
        (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        ),
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_threshold(
    checkpoint_record: dict,
    predictions_by_image: dict[str, dict],
    image_paths: list[Path],
    ground_truths: dict[
        str,
        list[float] | None,
    ],
    threshold: float,
) -> tuple[dict, list[dict]]:
    """保存済み予測を1つのconfidenceで再集計する。"""
    training_epoch = int(
        checkpoint_record["training_epoch"]
    )
    counts = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }
    per_image_rows = []

    for image_path in image_paths:
        image_record = predictions_by_image[
            image_path.stem
        ]
        image_width = int(
            image_record["image_width"]
        )
        image_height = int(
            image_record["image_height"]
        )
        predictions = [
            prediction
            for prediction
            in image_record["predictions"]
            if (
                float(prediction["confidence"])
                >= threshold
            )
        ]
        prediction_count = len(predictions)
        ground_truth_box = ground_truths[
            image_path.stem
        ]

        minimum_center_distance = None
        classification = ""
        ground_truth_visible = int(
            ground_truth_box is not None
        )
        ground_truth_x = 0
        ground_truth_y = 0

        if ground_truth_box is None:
            if prediction_count == 0:
                counts["TN"] += 1
                classification = "TN"
            else:
                counts["FP"] += prediction_count
                classification = "FP"
        else:
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
                counts["FN"] += 1
                classification = "FN"
            else:
                distances = [
                    float(
                        np.hypot(
                            int(
                                prediction[
                                    "center_x_heatmap"
                                ]
                            )
                            - ground_truth_x,
                            int(
                                prediction[
                                    "center_y_heatmap"
                                ]
                            )
                            - ground_truth_y,
                        )
                    )
                    for prediction in predictions
                ]
                minimum_center_distance = min(
                    distances
                )

                if (
                    minimum_center_distance
                    <= CENTER_TOLERANCE
                ):
                    counts["TP"] += 1
                    counts["FP"] += (
                        prediction_count - 1
                    )
                    classification = "TP"
                else:
                    counts["FN"] += 1
                    counts["FP"] += prediction_count
                    classification = "FP+FN"

        per_image_rows.append(
            {
                "training_epoch": (
                    training_epoch
                ),
                "checkpoint_file": (
                    checkpoint_record["file_name"]
                ),
                "confidence_threshold": (
                    threshold
                ),
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
                "minimum_center_distance": (
                    minimum_center_distance
                ),
                "classification": (
                    classification
                ),
            }
        )

    metrics = calculate_metrics(
        counts["TP"],
        counts["TN"],
        counts["FP"],
        counts["FN"],
    )

    summary = {
        "training_epoch": training_epoch,
        "checkpoint_file": (
            checkpoint_record["file_name"]
        ),
        "checkpoint_sha256": (
            checkpoint_record["sha256"]
        ),
        "confidence_threshold": threshold,
        "image_size": IMAGE_SIZE,
        "center_tolerance": (
            CENTER_TOLERANCE
        ),
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "selected": False,
    }

    return summary, per_image_rows


def select_best_summary(
    summaries: list[dict],
) -> dict:
    """val F1を最優先に採用条件を決める。"""
    if not summaries:
        raise RuntimeError(
            "選定対象の結果がありません"
        )

    selected = max(
        summaries,
        key=lambda summary: (
            float(summary["f1"]),
            float(summary["precision"]),
            float(summary["recall"]),
            float(
                summary[
                    "confidence_threshold"
                ]
            ),
            int(summary["training_epoch"]),
        ),
    )

    selected["selected"] = True
    return selected


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """辞書行をUTF-8 BOM付きCSVへ保存する。"""
    if not rows:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """checkpointとconfidenceのval選定を実行する。"""
    args = parse_args()
    thresholds = parse_thresholds(
        args.thresholds
    )

    (
        image_dir,
        label_dir,
        checkpoint_dir,
        output_dir,
        image_paths,
        ground_truths,
        checkpoint_records,
    ) = validate_inputs(
        args.image_dir,
        args.label_dir,
        args.checkpoint_dir,
        args.output_dir,
        thresholds,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPUを利用できません"
        )

    inference_confidence = min(thresholds)
    all_summaries = []
    all_per_image_rows = []
    all_prediction_rows = []
    runtimes = []

    print(
        "YOLOパイロットcheckpoint分析を"
        "開始します"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"val画像数: {len(image_paths)}")
    print(f"画像サイズ: {IMAGE_SIZE}")
    print(
        "共通中心距離tolerance: "
        f"{CENTER_TOLERANCE}"
    )
    print(
        "confidence候補: "
        + ", ".join(
            f"{threshold:.2f}"
            for threshold in thresholds
        )
    )

    for checkpoint_record in checkpoint_records:
        (
            predictions_by_image,
            prediction_rows,
            runtime,
        ) = run_inference(
            checkpoint_record,
            image_paths,
            inference_confidence,
        )

        all_prediction_rows.extend(
            prediction_rows
        )
        runtimes.append(runtime)

        for threshold in thresholds:
            (
                summary,
                per_image_rows,
            ) = evaluate_threshold(
                checkpoint_record,
                predictions_by_image,
                image_paths,
                ground_truths,
                threshold,
            )
            all_summaries.append(summary)
            all_per_image_rows.extend(
                per_image_rows
            )

            print(
                "Epoch "
                f"{summary['training_epoch']}, "
                f"confidence={threshold:.2f}: "
                f"TP={summary['TP']}, "
                f"TN={summary['TN']}, "
                f"FP={summary['FP']}, "
                f"FN={summary['FN']}, "
                f"Precision="
                f"{summary['precision']:.4f}, "
                f"Recall="
                f"{summary['recall']:.4f}, "
                f"F1={summary['f1']:.4f}"
            )

    selected = select_best_summary(
        all_summaries
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output_dir / "summary.csv",
        all_summaries,
    )
    write_csv(
        output_dir / "per_image.csv",
        all_per_image_rows,
    )
    write_csv(
        output_dir / "predictions.csv",
        all_prediction_rows,
    )
    write_csv(
        output_dir / "runtime.csv",
        runtimes,
    )

    positive_count = sum(
        ground_truth is not None
        for ground_truth in ground_truths.values()
    )
    negative_count = (
        len(ground_truths) - positive_count
    )
    script_path = Path(__file__).resolve()

    analysis = {
        "task": (
            "yolo_pilot_checkpoint_and_"
            "confidence_selection"
        ),
        "selection_dataset": "validation",
        "test_results_used_for_selection": False,
        "selection_rule": [
            "maximum_f1",
            "maximum_precision",
            "maximum_recall",
            "maximum_confidence_threshold",
            "maximum_training_epoch",
        ],
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "gpu": torch.cuda.get_device_name(0),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "checkpoint_dir": str(
            checkpoint_dir
        ),
        "output_dir": str(output_dir),
        "image_count": len(image_paths),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "image_size": IMAGE_SIZE,
        "ball_class_id": BALL_CLASS_ID,
        "heatmap_width": HEATMAP_WIDTH,
        "heatmap_height": HEATMAP_HEIGHT,
        "center_tolerance": (
            CENTER_TOLERANCE
        ),
        "inference_confidence": (
            inference_confidence
        ),
        "thresholds": thresholds,
        "script_sha256": (
            calculate_sha256(script_path)
        ),
        "checkpoints": [
            {
                "training_epoch": (
                    checkpoint["training_epoch"]
                ),
                "file_name": (
                    checkpoint["file_name"]
                ),
                "path": str(
                    checkpoint["path"]
                ),
                "sha256": (
                    checkpoint["sha256"]
                ),
            }
            for checkpoint in checkpoint_records
        ],
        "runtimes": runtimes,
        "selected": selected,
        "summaries": all_summaries,
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

    print()
    print("=== val選定結果 ===")
    print(
        "採用学習Epoch: "
        f"{selected['training_epoch']}"
    )
    print(
        "採用checkpoint: "
        f"{selected['checkpoint_file']}"
    )
    print(
        "採用confidence: "
        f"{selected['confidence_threshold']:.2f}"
    )
    print(f"TP: {selected['TP']}")
    print(f"TN: {selected['TN']}")
    print(f"FP: {selected['FP']}")
    print(f"FN: {selected['FN']}")
    print(
        "Precision: "
        f"{selected['precision']:.4f}"
    )
    print(
        "Recall: "
        f"{selected['recall']:.4f}"
    )
    print(f"F1: {selected['f1']:.4f}")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()