import csv
import re
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_metrics"

MODEL_NAME = "yolo11n.pt"
SPORTS_BALL_CLASS_ID = 32
IOU_THRESHOLD = 0.5

EXPERIMENTS = [
    {
        "name": "conf025_img640",
        "confidence": 0.25,
        "image_size": 640,
    },
    {
        "name": "conf025_img960",
        "confidence": 0.25,
        "image_size": 960,
    },
    {
        "name": "conf025_img1280",
        "confidence": 0.25,
        "image_size": 1280,
    },
]


def get_source_image_stem(label_path: Path) -> str | None:
    """Label Studioの接頭辞を除き、元画像名を取得する。"""
    match = re.search(r"(frame_\d{6})$", label_path.stem)

    if match is None:
        return None

    return match.group(1)


def load_ground_truths() -> dict[str, list[float] | None]:
    """各画像の正解枠を読み込む。負例画像にはNoneを設定する。"""
    ground_truths = {}

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        source_stem = get_source_image_stem(label_path)

        if source_stem is None:
            raise ValueError(
                f"元画像名を取得できません: {label_path.name}"
            )

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            ground_truths[source_stem] = None
            continue

        parts = lines[0].split()
        x_center, y_center, width, height = map(
            float,
            parts[1:],
        )

        # YOLO形式の中心座標と幅・高さを、左上・右下座標へ変換する。
        ground_truths[source_stem] = [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ]

    return ground_truths


def calculate_iou(
    first_box: list[float],
    second_box: list[float],
) -> float:
    """2つの枠がどの程度重なっているかを0〜1で計算する。"""
    intersection_x_min = max(first_box[0], second_box[0])
    intersection_y_min = max(first_box[1], second_box[1])
    intersection_x_max = min(first_box[2], second_box[2])
    intersection_y_max = min(first_box[3], second_box[3])

    intersection_width = max(
        0.0,
        intersection_x_max - intersection_x_min,
    )
    intersection_height = max(
        0.0,
        intersection_y_max - intersection_y_min,
    )
    intersection_area = intersection_width * intersection_height

    first_area = (
        max(0.0, first_box[2] - first_box[0])
        * max(0.0, first_box[3] - first_box[1])
    )
    second_area = (
        max(0.0, second_box[2] - second_box[0])
        * max(0.0, second_box[3] - second_box[1])
    )

    union_area = first_area + second_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def convert_predictions_to_normalized_boxes(result) -> list[list[float]]:
    """YOLOのピクセル座標を、0〜1の正規化座標へ変換する。"""
    image_height, image_width = result.orig_shape
    prediction_boxes = []

    if result.boxes is None:
        return prediction_boxes

    for box in result.boxes.xyxy.cpu().tolist():
        prediction_boxes.append(
            [
                box[0] / image_width,
                box[1] / image_height,
                box[2] / image_width,
                box[3] / image_height,
            ]
        )

    return prediction_boxes


def safe_divide(numerator: float, denominator: float) -> float:
    """0による割り算を避けて割合を計算する。"""
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_experiment(
    model: YOLO,
    image_paths: list[Path],
    ground_truths: dict[str, list[float] | None],
    experiment: dict,
) -> dict:
    experiment_name = experiment["name"]

    print(f"\n評価開始: {experiment_name}")

    start_time = perf_counter()

    results = model.predict(
        source=[str(image_path) for image_path in image_paths],
        conf=experiment["confidence"],
        imgsz=experiment["image_size"],
        classes=[SPORTS_BALL_CLASS_ID],
        verbose=False,
    )

    elapsed_seconds = perf_counter() - start_time

    true_positive = 0
    false_positive = 0
    false_negative = 0

    true_negative_images = 0
    negative_images_with_false_positive = 0
    multiple_prediction_images = 0

    matched_ious = []
    detail_rows = []

    for image_path, result in zip(image_paths, results):
        ground_truth_box = ground_truths[image_path.stem]
        prediction_boxes = convert_predictions_to_normalized_boxes(
            result
        )

        prediction_count = len(prediction_boxes)
        matched = False
        best_iou = 0.0
        image_false_positive = 0
        image_false_negative = 0

        if prediction_count > 1:
            multiple_prediction_images += 1

        if ground_truth_box is None:
            if prediction_count == 0:
                true_negative_images += 1
            else:
                false_positive += prediction_count
                image_false_positive = prediction_count
                negative_images_with_false_positive += 1
        elif prediction_count == 0:
            false_negative += 1
            image_false_negative = 1
        else:
            ious = [
                calculate_iou(
                    ground_truth_box,
                    prediction_box,
                )
                for prediction_box in prediction_boxes
            ]
            best_iou = max(ious)

            if best_iou >= IOU_THRESHOLD:
                matched = True
                true_positive += 1
                matched_ious.append(best_iou)

                extra_predictions = prediction_count - 1
                false_positive += extra_predictions
                image_false_positive = extra_predictions
            else:
                false_negative += 1
                image_false_negative = 1
                false_positive += prediction_count
                image_false_positive = prediction_count

        detail_rows.append(
            {
                "image_name": image_path.name,
                "ground_truth_present": ground_truth_box is not None,
                "prediction_count": prediction_count,
                "matched_at_iou_050": matched,
                "best_iou": round(best_iou, 6),
                "false_positive_count": image_false_positive,
                "false_negative_count": image_false_negative,
            }
        )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )
    average_matched_iou = safe_divide(
        sum(matched_ious),
        len(matched_ious),
    )
    negative_false_positive_rate = safe_divide(
        negative_images_with_false_positive,
        sum(
            ground_truth is None
            for ground_truth in ground_truths.values()
        ),
    )
    processing_fps = safe_divide(
        len(image_paths),
        elapsed_seconds,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / f"{experiment_name}_details.csv"

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as detail_file:
        writer = csv.DictWriter(
            detail_file,
            fieldnames=detail_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = {
        "experiment": experiment_name,
        "confidence": experiment["confidence"],
        "image_size": experiment["image_size"],
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "average_matched_iou": average_matched_iou,
        "true_negative_images": true_negative_images,
        "negative_images_with_false_positive": (
            negative_images_with_false_positive
        ),
        "negative_false_positive_rate": (
            negative_false_positive_rate
        ),
        "multiple_prediction_images": (
            multiple_prediction_images
        ),
        "elapsed_seconds": elapsed_seconds,
        "processing_fps": processing_fps,
    }

    print(f"評価完了: {experiment_name}")
    print(f"TP: {true_positive}")
    print(f"FP: {false_positive}")
    print(f"FN: {false_negative}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1_score:.3f}")
    print(f"平均IoU: {average_matched_iou:.3f}")
    print(
        "負例で誤検出した画像: "
        f"{negative_images_with_false_positive}"
    )
    print(f"処理時間: {elapsed_seconds:.2f}秒")
    print(f"処理速度: {processing_fps:.2f}画像/秒")

    return summary


def save_summaries(summaries: list[dict]) -> None:
    summary_path = OUTPUT_DIR / "summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=summaries[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n比較結果の保存先: {summary_path}")


def main() -> None:
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    ground_truths = load_ground_truths()

    if len(image_paths) != len(ground_truths):
        print(
            "画像数と正解ラベル数が一致しないため、"
            "評価を中止します"
        )
        return

    model = YOLO(MODEL_NAME)
    summaries = []

    for experiment in EXPERIMENTS:
        summary = evaluate_experiment(
            model,
            image_paths,
            ground_truths,
            experiment,
        )
        summaries.append(summary)

    save_summaries(summaries)

    print("\n比較評価がすべて完了しました")
    print("-" * 78)
    print(
        f"{'実験名':<22}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1':>9}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'時間':>10}"
    )

    for summary in summaries:
        print(
            f"{summary['experiment']:<22}"
            f"{summary['precision']:>11.3f}"
            f"{summary['recall']:>10.3f}"
            f"{summary['f1_score']:>9.3f}"
            f"{summary['false_positive']:>7}"
            f"{summary['false_negative']:>7}"
            f"{summary['elapsed_seconds']:>9.1f}秒"
        )


if __name__ == "__main__":
    main()