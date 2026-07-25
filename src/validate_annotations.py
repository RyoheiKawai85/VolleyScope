import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)

EXPECTED_FILE_COUNT = 150
BALL_CLASS_ID = 0
COORDINATE_TOLERANCE = 1e-6


def get_source_image_stem(label_path: Path) -> str | None:
    """Label Studioの接頭辞を除き、元画像の名前を取得する。"""
    match = re.search(r"(frame_\d{6})$", label_path.stem)

    if match is None:
        return None

    return match.group(1)


def main() -> None:
    label_paths = sorted(LABEL_DIR.glob("*.txt"))
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    image_stems = {image_path.stem for image_path in image_paths}

    errors = []
    source_stems = []

    positive_image_count = 0
    negative_image_count = 0
    total_box_count = 0
    multiple_box_image_count = 0

    if len(label_paths) != EXPECTED_FILE_COUNT:
        errors.append(
            "ラベルファイル数が想定と異なります: "
            f"{len(label_paths)} / {EXPECTED_FILE_COUNT}"
        )

    if len(image_paths) != EXPECTED_FILE_COUNT:
        errors.append(
            "評価画像数が想定と異なります: "
            f"{len(image_paths)} / {EXPECTED_FILE_COUNT}"
        )

    for label_path in label_paths:
        source_stem = get_source_image_stem(label_path)

        if source_stem is None:
            errors.append(
                f"元画像名を取得できません: {label_path.name}"
            )
            continue

        source_stems.append(source_stem)

        if source_stem not in image_stems:
            errors.append(
                f"対応する元画像がありません: {label_path.name}"
            )

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            negative_image_count += 1
            continue

        positive_image_count += 1
        total_box_count += len(lines)

        if len(lines) > 1:
            multiple_box_image_count += 1
            errors.append(
                f"複数の正解枠があります: "
                f"{label_path.name} ({len(lines)}枠)"
            )

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"項目数が5ではありません: "
                    f"{label_path.name} {line_number}行目"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(
                    float,
                    parts[1:],
                )
            except ValueError:
                errors.append(
                    f"数値へ変換できません: "
                    f"{label_path.name} {line_number}行目"
                )
                continue

            if class_id != BALL_CLASS_ID:
                errors.append(
                    f"クラス番号が0ではありません: "
                    f"{label_path.name} {line_number}行目"
                )

            coordinates = {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }

            for coordinate_name, coordinate_value in coordinates.items():
                if not 0 <= coordinate_value <= 1:
                    errors.append(
                        f"{coordinate_name}が0〜1の範囲外です: "
                        f"{label_path.name} {line_number}行目"
                    )

            if width <= 0 or height <= 0:
                errors.append(
                    f"枠の幅または高さが0以下です: "
                    f"{label_path.name} {line_number}行目"
                )
                continue

            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            if (
                x_min < -COORDINATE_TOLERANCE
                or x_max > 1 + COORDINATE_TOLERANCE
                or y_min < -COORDINATE_TOLERANCE
                or y_max > 1 + COORDINATE_TOLERANCE
            ):
                errors.append(
                    f"正解枠が画像範囲外へ出ています: "
                    f"{label_path.name} {line_number}行目"
                )

    unique_source_stems = set(source_stems)

    if len(unique_source_stems) != len(source_stems):
        errors.append(
            "複数のラベルファイルが同じ元画像へ対応しています"
        )

    missing_label_stems = image_stems - unique_source_stems

    if missing_label_stems:
        errors.append(
            "ラベルファイルがない評価画像があります: "
            f"{len(missing_label_stems)}枚"
        )

    print("アノテーション検査結果")
    print("-" * 50)
    print(f"評価画像数: {len(image_paths)}")
    print(f"ラベルファイル数: {len(label_paths)}")
    print(f"ボールあり画像: {positive_image_count}")
    print(f"ボールなし画像: {negative_image_count}")
    print(f"正解枠の総数: {total_box_count}")
    print(f"複数枠がある画像: {multiple_box_image_count}")
    print(f"エラー数: {len(errors)}")

    if errors:
        print("\n確認が必要な内容")

        for error in errors:
            print(f"- {error}")

        return

    print("\n形式上の問題は見つかりませんでした")


if __name__ == "__main__":
    main()