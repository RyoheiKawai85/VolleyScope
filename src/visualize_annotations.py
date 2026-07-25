import re
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "annotation_review"
    / "evaluation_001"
)


def get_source_image_stem(label_path: Path) -> str | None:
    """Label Studioの接頭辞を除き、元画像名を取得する。"""
    match = re.search(r"(frame_\d{6})$", label_path.stem)

    if match is None:
        return None

    return match.group(1)


def draw_yolo_box(
    image,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
) -> None:
    """正規化されたYOLO座標をピクセル座標へ戻して描画する。"""
    image_height, image_width = image.shape[:2]

    x_min = int(
        (x_center - box_width / 2) * image_width
    )
    x_max = int(
        (x_center + box_width / 2) * image_width
    )
    y_min = int(
        (y_center - box_height / 2) * image_height
    )
    y_max = int(
        (y_center + box_height / 2) * image_height
    )

    # 計算誤差で画像外へ出ないよう、座標を画像範囲内に収める。
    x_min = max(0, min(x_min, image_width - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    y_max = max(0, min(y_max, image_height - 1))

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        color=(0, 255, 0),
        thickness=3,
    )

    cv2.putText(
        image,
        "ball",
        (x_min, max(30, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(LABEL_DIR.glob("*.txt"))
    saved_count = 0
    positive_count = 0
    negative_count = 0

    for label_path in label_paths:
        source_stem = get_source_image_stem(label_path)

        if source_stem is None:
            print(f"元画像名を取得できません: {label_path.name}")
            continue

        image_path = IMAGE_DIR / f"{source_stem}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"画像を開けません: {image_path}")
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            negative_count += 1

            cv2.putText(
                image,
                "NO BALL - negative",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        else:
            positive_count += 1

            for line in lines:
                parts = line.split()
                x_center, y_center, width, height = map(
                    float,
                    parts[1:],
                )

                draw_yolo_box(
                    image,
                    x_center,
                    y_center,
                    width,
                    height,
                )

        output_path = OUTPUT_DIR / image_path.name

        if not cv2.imwrite(str(output_path), image):
            print(f"画像を保存できません: {output_path}")
            continue

        saved_count += 1

    print("アノテーション確認画像を作成しました")
    print(f"保存枚数: {saved_count}")
    print(f"ボールあり画像: {positive_count}")
    print(f"ボールなし画像: {negative_count}")
    print(f"出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()