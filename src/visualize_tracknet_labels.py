import argparse
import csv
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
    / "train"
    / "images"
)

TRACKNET_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_smoke"
    / "converted"
    / "tracknet_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_label_review"
    / "smoke"
)


def parse_args() -> argparse.Namespace:
    """可視化に使用する入出力パスを取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3用CSVの座標を"
            "元画像へ重ねて確認画像を作成する"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIR,
        help="元画像フォルダ",
    )
    parser.add_argument(
        "--tracknet-csv",
        type=Path,
        default=TRACKNET_CSV_PATH,
        help="TrackNetV3用CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="確認画像の出力先",
    )
    return parser.parse_args()


def read_rows(
    tracknet_csv_path: Path,
) -> list[dict[str, int]]:
    """TrackNetV3用CSVを整数へ変換して読み込む。"""
    if not tracknet_csv_path.is_file():
        raise FileNotFoundError(
            f"CSVが見つかりません: "
            f"{tracknet_csv_path}"
        )

    with tracknet_csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []

        for source_row in reader:
            rows.append(
                {
                    "Frame": int(
                        source_row["Frame"]
                    ),
                    "Visibility": int(
                        source_row["Visibility"]
                    ),
                    "X": int(source_row["X"]),
                    "Y": int(source_row["Y"]),
                }
            )

    if not rows:
        raise RuntimeError(
            "CSVにデータ行がありません"
        )

    return rows


def validate_row(
    row: dict[str, int],
    image_width: int,
    image_height: int,
) -> None:
    """可視性と座標の関係を検証する。"""
    visibility = row["Visibility"]
    x = row["X"]
    y = row["Y"]

    if visibility not in {0, 1}:
        raise ValueError(
            "Visibilityは0または1である必要が"
            f"あります: {row}"
        )

    if visibility == 0:
        if x != 0 or y != 0:
            raise ValueError(
                "Visibility=0の座標は"
                f"0,0である必要があります: {row}"
            )

        return

    if not (
        0 <= x < image_width
        and 0 <= y < image_height
    ):
        raise ValueError(
            "可視ボールの座標が画像外です: "
            f"{row}"
        )


def draw_visible_ball(
    image,
    x: int,
    y: int,
) -> None:
    """可視ボールの中心へ十字と円を描く。"""
    cv2.drawMarker(
        image,
        (x, y),
        color=(0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=3,
        line_type=cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (x, y),
        radius=18,
        color=(0, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"VISIBLE ({x}, {y})",
        (max(10, x + 25), max(35, y - 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_invisible_ball(
    image,
) -> None:
    """不可視フレームであることを画像上へ表示する。"""
    cv2.putText(
        image,
        "INVISIBLE - no coordinate",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )


def main() -> None:
    """CSV座標を画像へ重ねて確認画像を作成する。"""
    args = parse_args()

    image_dir = args.image_dir.resolve()
    tracknet_csv_path = args.tracknet_csv.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        existing_files = [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
        ]

        if existing_files:
            raise FileExistsError(
                "出力先に既存ファイルがあります。"
                "上書きを防ぐため停止します: "
                f"{output_dir}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_rows(tracknet_csv_path)
    visible_count = 0
    invisible_count = 0

    for row in rows:
        frame_index = row["Frame"]
        image_path = (
            image_dir
            / f"frame_{frame_index:06d}.png"
        )
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"画像を開けません: {image_path}"
            )

        image_height, image_width = (
            image.shape[:2]
        )
        validate_row(
            row,
            image_width,
            image_height,
        )

        if row["Visibility"] == 1:
            draw_visible_ball(
                image,
                row["X"],
                row["Y"],
            )
            visible_count += 1
        else:
            draw_invisible_ball(image)
            invisible_count += 1

        output_path = (
            output_dir / image_path.name
        )

        if not cv2.imwrite(
            str(output_path),
            image,
        ):
            raise RuntimeError(
                f"確認画像を保存できません: "
                f"{output_path}"
            )

    if visible_count + invisible_count != len(
        rows
    ):
        raise RuntimeError(
            "可視・不可視数の合計が"
            "CSV行数と一致しません"
        )

    print(
        "TrackNetV3座標の確認画像を"
        "作成しました"
    )
    print(f"保存枚数: {len(rows)}")
    print(f"可視: {visible_count}")
    print(f"不可視: {invisible_count}")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()