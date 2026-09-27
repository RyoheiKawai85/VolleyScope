import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluate_pretrained_model import load_ground_truths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "external_evaluation_003"
    / "evaluation_images"
)

DEFAULT_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "external_evaluation_003_final"
    / "evaluation"
    / "labels"
)

DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_common_center"
    / "external_evaluation_003_diverse_epoch06_conf015"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_prediction_review"
    / "external_evaluation_003_diverse_epoch06_conf015"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "保存済みYOLO評価CSVの予測座標を"
            "元画像へ再描画する"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="評価元画像フォルダ",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="YOLO形式の正解ラベルフォルダ",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=(
            DEFAULT_EVALUATION_DIR
            / "per_image.csv"
        ),
        help="画像単位の評価結果CSV",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=(
            DEFAULT_EVALUATION_DIR
            / "predictions.csv"
        ),
        help="予測bbox単位のCSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="確認画像の新規出力先",
    )
    return parser.parse_args()


def read_csv_rows(
    csv_path: Path,
) -> list[dict[str, str]]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def parse_boolean(value: str) -> bool:
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def clamp_coordinate(
    value: float,
    maximum: int,
) -> int:
    rounded = int(round(value))

    return max(
        0,
        min(maximum - 1, rounded),
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )

    top = max(
        0,
        y - text_height - baseline - 8,
    )
    bottom = min(
        image.shape[0] - 1,
        y + 2,
    )
    right = min(
        image.shape[1] - 1,
        x + text_width + 8,
    )

    cv2.rectangle(
        image,
        (x, top),
        (right, bottom),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, bottom - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_ground_truth(
    image: np.ndarray,
    normalized_box: list[float] | None,
) -> None:
    if normalized_box is None:
        return

    image_height, image_width = image.shape[:2]

    x1 = clamp_coordinate(
        normalized_box[0] * image_width,
        image_width,
    )
    y1 = clamp_coordinate(
        normalized_box[1] * image_height,
        image_height,
    )
    x2 = clamp_coordinate(
        normalized_box[2] * image_width,
        image_width,
    )
    y2 = clamp_coordinate(
        normalized_box[3] * image_height,
        image_height,
    )

    color = (0, 200, 0)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness=3,
    )
    draw_label(
        image,
        "GROUND TRUTH",
        x1,
        y1,
        color,
    )


def draw_prediction(
    image: np.ndarray,
    prediction: dict[str, str],
) -> None:
    image_height, image_width = image.shape[:2]

    x1 = clamp_coordinate(
        float(prediction["x1_original"]),
        image_width,
    )
    y1 = clamp_coordinate(
        float(prediction["y1_original"]),
        image_height,
    )
    x2 = clamp_coordinate(
        float(prediction["x2_original"]),
        image_width,
    )
    y2 = clamp_coordinate(
        float(prediction["y2_original"]),
        image_height,
    )

    is_best = parse_boolean(
        prediction["best_center_prediction"]
    )

    color = (
        (255, 255, 0)
        if is_best
        else (0, 0, 255)
    )

    confidence = float(
        prediction["confidence"]
    )
    prediction_index = int(
        prediction["prediction_index"]
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness=3,
    )
    draw_label(
        image,
        (
            f"PRED {prediction_index} "
            f"conf={confidence:.3f}"
        ),
        x1,
        y1,
        color,
    )


def add_header(
    image: np.ndarray,
    per_image_row: dict[str, str],
) -> np.ndarray:
    header_height = 100

    header = np.full(
        (
            header_height,
            image.shape[1],
            3,
        ),
        245,
        dtype=np.uint8,
    )

    first_line = (
        f"Image: {per_image_row['image_name']}   "
        f"Class: "
        f"{per_image_row['center_classification']}"
    )
    second_line = (
        "Ground truth visible: "
        f"{per_image_row['ground_truth_visible']}   "
        "Predictions: "
        f"{per_image_row['prediction_count']}   "
        "Minimum center distance: "
        f"{per_image_row['minimum_center_distance']}"
    )

    cv2.putText(
        header,
        first_line,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        header,
        second_line,
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    return np.vstack(
        (header, image)
    )


def should_visualize(
    per_image_row: dict[str, str],
) -> bool:
    classification = per_image_row[
        "center_classification"
    ]
    prediction_count = int(
        per_image_row["prediction_count"]
    )

    if classification in {
        "FP",
        "FP+FN",
    }:
        return True

    return (
        classification == "TP"
        and prediction_count > 1
    )


def main() -> None:
    args = parse_arguments()

    image_dir = args.image_dir.resolve()
    label_dir = args.label_dir.resolve()
    per_image_csv = (
        args.per_image_csv.resolve()
    )
    predictions_csv = (
        args.predictions_csv.resolve()
    )
    output_dir = args.output_dir.resolve()

    required_paths = [
        image_dir,
        label_dir,
        per_image_csv,
        predictions_csv,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"入力がありません: "
                f"{required_path}"
            )

    if output_dir.exists():
        raise FileExistsError(
            f"出力先が既に存在します: "
            f"{output_dir}"
        )

    per_image_rows = read_csv_rows(
        per_image_csv
    )
    prediction_rows = read_csv_rows(
        predictions_csv
    )
    ground_truths = load_ground_truths(
        label_dir
    )

    predictions_by_image = defaultdict(list)

    for prediction_row in prediction_rows:
        predictions_by_image[
            prediction_row["image_name"]
        ].append(prediction_row)

    selected_rows = [
        row
        for row in per_image_rows
        if should_visualize(row)
    ]

    selected_names = {
        row["image_name"]
        for row in selected_rows
    }

    if len(selected_rows) != 23:
        raise ValueError(
            "可視化対象が23枚ではありません: "
            f"{len(selected_rows)}"
        )

    missing_images = sorted(
        image_name
        for image_name in selected_names
        if not (
            image_dir / image_name
        ).exists()
    )

    if missing_images:
        raise FileNotFoundError(
            "元画像がありません: "
            f"{missing_images}"
        )

    missing_labels = sorted(
        Path(image_name).stem
        for image_name in selected_names
        if Path(image_name).stem
        not in ground_truths
    )

    if missing_labels:
        raise ValueError(
            "正解ラベルがありません: "
            f"{missing_labels}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_rows = []

    for per_image_row in selected_rows:
        image_name = per_image_row[
            "image_name"
        ]
        image_path = image_dir / image_name

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"画像を読み込めません: "
                f"{image_path}"
            )

        image_stem = Path(image_name).stem

        draw_ground_truth(
            image,
            ground_truths[image_stem],
        )

        image_predictions = sorted(
            predictions_by_image.get(
                image_name,
                [],
            ),
            key=lambda row: int(
                row["prediction_index"]
            ),
        )

        expected_prediction_count = int(
            per_image_row[
                "prediction_count"
            ]
        )

        if (
            len(image_predictions)
            != expected_prediction_count
        ):
            raise ValueError(
                "予測数が一致しません: "
                f"{image_name}, "
                f"CSV={len(image_predictions)}, "
                f"expected="
                f"{expected_prediction_count}"
            )

        for prediction in image_predictions:
            draw_prediction(
                image,
                prediction,
            )

        review_image = add_header(
            image,
            per_image_row,
        )

        classification = per_image_row[
            "center_classification"
        ].replace(
            "+",
            "_plus_",
        )

        output_name = (
            f"{image_stem}_"
            f"{classification}.png"
        )
        output_path = (
            output_dir / output_name
        )

        write_succeeded = cv2.imwrite(
            str(output_path),
            review_image,
        )

        if not write_succeeded:
            raise RuntimeError(
                f"画像保存に失敗しました: "
                f"{output_path}"
            )

        summary_rows.append(
            {
                "image_name": image_name,
                "classification": (
                    per_image_row[
                        "center_classification"
                    ]
                ),
                "ground_truth_visible": (
                    per_image_row[
                        "ground_truth_visible"
                    ]
                ),
                "prediction_count": (
                    expected_prediction_count
                ),
                "output_name": output_name,
            }
        )

    summary_path = (
        output_dir / "review_manifest.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        fieldnames = [
            "image_name",
            "classification",
            "ground_truth_visible",
            "prediction_count",
            "output_name",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        "保存済みYOLO予測の確認画像を"
        "作成しました"
    )
    print(
        f"可視化対象: "
        f"{len(selected_rows)}枚"
    )
    print(
        "負例FP: "
        f"{sum(row['center_classification'] == 'FP' for row in selected_rows)}枚"
    )
    print(
        "正例位置外れ: "
        f"{sum(row['center_classification'] == 'FP+FN' for row in selected_rows)}枚"
    )
    print(
        "TP＋余分な予測: "
        f"{sum(row['center_classification'] == 'TP' for row in selected_rows)}枚"
    )
    print(
        f"出力先: {output_dir}"
    )
    print(
        f"対応表: {summary_path}"
    )


if __name__ == "__main__":
    main()