import csv

import cv2
import numpy as np

from prepare_false_negative_review import (
    PROJECT_ROOT,
    IMAGE_DIR,
    DETAIL_CSV_PATH,
    ZOOM_PANEL_WIDTH,
    ZOOM_IMAGE_SIZE,
    load_ground_truths,
    convert_box_to_pixels,
    create_zoom_crop,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "true_positive_review"
    / "imgsz1280"
)


def load_true_positive_names() -> list[str]:
    """1280の評価結果から、正しく検出できた画像名を取得する。"""
    true_positive_names = []

    with DETAIL_CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as detail_file:
        reader = csv.DictReader(detail_file)

        for row in reader:
            matched = (
                row["matched_at_iou_050"]
                .strip()
                .lower()
                == "true"
            )

            if matched:
                true_positive_names.append(
                    row["image_name"]
                )

    return true_positive_names


def create_true_positive_review_image(
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

    image_height = annotated_image.shape[0]

    zoom_panel = np.zeros(
        (image_height, ZOOM_PANEL_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        zoom_panel,
        "TRUE POSITIVE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 200, 0),
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

    panel_x = (
        ZOOM_PANEL_WIDTH - ZOOM_IMAGE_SIZE
    ) // 2
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


def main() -> None:
    if not DETAIL_CSV_PATH.exists():
        print(
            f"評価結果CSVが見つかりません: "
            f"{DETAIL_CSV_PATH}"
        )
        return

    ground_truths = load_ground_truths()
    true_positive_names = (
        load_true_positive_names()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0
    errors = []

    for image_name in true_positive_names:
        image_path = IMAGE_DIR / image_name
        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(
                f"画像を開けません: {image_name}"
            )
            continue

        ground_truth_box = ground_truths.get(
            image_path.stem
        )

        if ground_truth_box is None:
            errors.append(
                f"正解枠がありません: {image_name}"
            )
            continue

        image_height, image_width = image.shape[:2]

        pixel_box = convert_box_to_pixels(
            ground_truth_box,
            image_width,
            image_height,
        )

        review_image = (
            create_true_positive_review_image(
                image,
                image_name,
                pixel_box,
            )
        )

        output_path = OUTPUT_DIR / image_name

        if not cv2.imwrite(
            str(output_path),
            review_image,
        ):
            errors.append(
                f"保存できません: {output_path}"
            )
            continue

        saved_count += 1

    print("検出成功例の確認データを作成しました")
    print(
        f"検出成功画像数: "
        f"{len(true_positive_names)}"
    )
    print(f"確認画像の保存数: {saved_count}")
    print(f"エラー数: {len(errors)}")
    print(f"保存先: {OUTPUT_DIR}")

    if errors:
        print("\n確認が必要な内容")

        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()