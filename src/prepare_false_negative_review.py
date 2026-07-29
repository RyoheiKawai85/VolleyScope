import csv
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)
DETAIL_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_metrics"
    / "conf025_img1280_details.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "false_negative_review"
    / "imgsz1280"
)
REVIEW_CSV_PATH = OUTPUT_DIR / "review_manifest.csv"

ZOOM_PANEL_WIDTH = 500
ZOOM_IMAGE_SIZE = 440


def get_source_image_stem(label_path: Path) -> str | None:
    """Label Studioの接頭辞を除き、元画像名を取得する。"""
    match = re.search(r"(frame_\d{6})$", label_path.stem)

    if match is None:
        return None

    return match.group(1)


def load_ground_truths() -> dict[str, list[float]]:
    """ボールあり画像の正解枠を読み込む。"""
    ground_truths = {}

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        source_stem = get_source_image_stem(label_path)

        if source_stem is None:
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        parts = lines[0].split()
        x_center, y_center, width, height = map(
            float,
            parts[1:],
        )

        ground_truths[source_stem] = [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ]

    return ground_truths


def load_false_negative_names() -> list[str]:
    """1280の評価結果から、見逃し画像名だけを取得する。"""
    false_negative_names = []

    with DETAIL_CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as detail_file:
        reader = csv.DictReader(detail_file)

        for row in reader:
            if int(row["false_negative_count"]) == 1:
                false_negative_names.append(row["image_name"])

    return false_negative_names


def convert_box_to_pixels(
    normalized_box: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """0〜1の正規化座標を画像上のピクセル座標へ戻す。"""
    x_min = round(normalized_box[0] * image_width)
    y_min = round(normalized_box[1] * image_height)
    x_max = round(normalized_box[2] * image_width)
    y_max = round(normalized_box[3] * image_height)

    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_max = max(0, min(y_max, image_height - 1))

    return x_min, y_min, x_max, y_max


def create_zoom_crop(
    image,
    box: tuple[int, int, int, int],
):
    """ボール周辺を正方形で切り出し、確認しやすく拡大する。"""
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = box

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    box_size = max(x_max - x_min, y_max - y_min)
    crop_radius = max(80, box_size * 5)

    crop_x_min = max(0, center_x - crop_radius)
    crop_y_min = max(0, center_y - crop_radius)
    crop_x_max = min(image_width, center_x + crop_radius)
    crop_y_max = min(image_height, center_y + crop_radius)

    crop = image[
        crop_y_min:crop_y_max,
        crop_x_min:crop_x_max,
    ]

    if crop.size == 0:
        raise ValueError("拡大領域を作成できませんでした")

    return cv2.resize(
        crop,
        (ZOOM_IMAGE_SIZE, ZOOM_IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def create_review_image(
    image,
    image_name: str,
    box: tuple[int, int, int, int],
):
    """全体画像とボール周辺の拡大画像を横に並べる。"""
    x_min, y_min, x_max, y_max = box

    annotated_image = image.copy()

    cv2.rectangle(
        annotated_image,
        (x_min, y_min),
        (x_max, y_max),
        color=(0, 255, 0),
        thickness=3,
    )

    zoom_crop = create_zoom_crop(
        annotated_image,
        box,
    )

    image_height, image_width = annotated_image.shape[:2]

    zoom_panel = np.zeros(
        (image_height, ZOOM_PANEL_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        zoom_panel,
        "FALSE NEGATIVE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        zoom_panel,
        image_name,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        zoom_panel,
        "Green box = ground truth",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    panel_x = (ZOOM_PANEL_WIDTH - ZOOM_IMAGE_SIZE) // 2
    panel_y = 140

    if panel_y + ZOOM_IMAGE_SIZE <= image_height:
        zoom_panel[
            panel_y:panel_y + ZOOM_IMAGE_SIZE,
            panel_x:panel_x + ZOOM_IMAGE_SIZE,
        ] = zoom_crop

    return cv2.hconcat(
        [
            annotated_image,
            zoom_panel,
        ]
    )


def save_review_manifest(image_names: list[str]) -> None:
    """後で見逃し原因を記録するためのCSVを作成する。"""
    fieldnames = [
        "image_name",
        "small_or_far",
        "net_overlap",
        "player_or_hand_occlusion",
        "motion_blur",
        "frame_edge",
        "low_contrast",
        "other",
        "notes",
    ]

    with REVIEW_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as review_file:
        writer = csv.DictWriter(
            review_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for image_name in image_names:
            writer.writerow(
                {
                    "image_name": image_name,
                    "small_or_far": "",
                    "net_overlap": "",
                    "player_or_hand_occlusion": "",
                    "motion_blur": "",
                    "frame_edge": "",
                    "low_contrast": "",
                    "other": "",
                    "notes": "",
                }
            )


def main() -> None:
    if not DETAIL_CSV_PATH.exists():
        print(f"評価結果CSVが見つかりません: {DETAIL_CSV_PATH}")
        return

    ground_truths = load_ground_truths()
    false_negative_names = load_false_negative_names()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    errors = []

    for image_name in false_negative_names:
        image_path = IMAGE_DIR / image_name
        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"画像を開けません: {image_name}")
            continue

        ground_truth_box = ground_truths.get(image_path.stem)

        if ground_truth_box is None:
            errors.append(f"正解枠がありません: {image_name}")
            continue

        image_height, image_width = image.shape[:2]

        pixel_box = convert_box_to_pixels(
            ground_truth_box,
            image_width,
            image_height,
        )

        try:
            review_image = create_review_image(
                image,
                image_name,
                pixel_box,
            )
        except ValueError as error:
            errors.append(f"{image_name}: {error}")
            continue

        output_path = OUTPUT_DIR / image_name

        if not cv2.imwrite(str(output_path), review_image):
            errors.append(f"保存できません: {output_path}")
            continue

        saved_count += 1

    save_review_manifest(false_negative_names)

    print("見逃し確認データを作成しました")
    print(f"見逃し画像数: {len(false_negative_names)}")
    print(f"確認画像の保存数: {saved_count}")
    print(f"エラー数: {len(errors)}")
    print(f"確認画像の保存先: {OUTPUT_DIR}")
    print(f"分類用CSV: {REVIEW_CSV_PATH}")

    if errors:
        print("\n確認が必要な内容")

        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()