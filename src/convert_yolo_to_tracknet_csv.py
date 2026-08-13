import argparse
import csv
import re
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
    / "train"
    / "images"
)

DEFAULT_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_smoke"
    / "extracted"
    / "labels"
)

DEFAULT_CLASSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_smoke"
    / "extracted"
    / "classes.txt"
)

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_smoke"
    / "converted"
    / "tracknet_labels.csv"
)

LABEL_FILE_PATTERN = re.compile(
    r"(frame_(\d{6}))$"
)


def parse_args() -> argparse.Namespace:
    """変換元、出力先、対象フレーム範囲を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Label StudioのYOLOラベルを"
            "TrackNetV3用CSVへ変換する"
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="元画像フォルダ",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="YOLOラベルフォルダ",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES_PATH,
        help="YOLOのclasses.txt",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="TrackNetV3用CSVの出力先",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=2977,
        help="対象に含める最初の元フレーム番号",
    )
    parser.add_argument(
        "--end-frame-exclusive",
        type=int,
        default=2985,
        help="対象に含めない終了フレーム番号",
    )

    return parser.parse_args()


def validate_paths(
    args: argparse.Namespace,
) -> None:
    """入力パスとフレーム範囲を検証する。"""
    if not args.image_dir.is_dir():
        raise FileNotFoundError(
            f"画像フォルダが見つかりません: "
            f"{args.image_dir}"
        )

    if not args.label_dir.is_dir():
        raise FileNotFoundError(
            f"ラベルフォルダが見つかりません: "
            f"{args.label_dir}"
        )

    if not args.classes.is_file():
        raise FileNotFoundError(
            f"classes.txtが見つかりません: "
            f"{args.classes}"
        )

    if args.start_frame < 0:
        raise ValueError(
            "--start-frameは0以上にしてください"
        )

    if (
        args.end_frame_exclusive
        <= args.start_frame
    ):
        raise ValueError(
            "終了フレームは開始フレームより"
            "大きくしてください"
        )

    if args.output_csv.exists():
        raise FileExistsError(
            "出力CSVが既に存在します。"
            "上書きを防ぐため停止します: "
            f"{args.output_csv}"
        )


def read_classes(
    classes_path: Path,
) -> list[str]:
    """classes.txtから空行を除いてクラス名を読む。"""
    classes = [
        line.strip()
        for line in classes_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    if classes != ["ball"]:
        raise ValueError(
            "クラス定義が期待値と異なります。"
            f"期待=['ball'], 実際={classes}"
        )

    return classes


def build_label_mapping(
    label_dir: Path,
) -> dict[int, Path]:
    """Label Studio接頭辞付きラベルをフレーム番号へ対応付ける。"""
    label_mapping = {}

    for label_path in sorted(
        label_dir.glob("*.txt")
    ):
        match = LABEL_FILE_PATTERN.search(
            label_path.stem
        )

        if match is None:
            raise ValueError(
                "ラベル名からフレーム番号を"
                "取得できません: "
                f"{label_path.name}"
            )

        frame_index = int(match.group(2))

        if frame_index in label_mapping:
            raise ValueError(
                "同じフレームに対応するラベルが"
                "複数あります: "
                f"{frame_index}"
            )

        label_mapping[frame_index] = label_path

    if not label_mapping:
        raise RuntimeError(
            "YOLOラベルが見つかりません"
        )

    return label_mapping


def read_nonempty_lines(
    label_path: Path,
) -> list[str]:
    """ラベルファイルから空行を除いて読む。"""
    return [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def round_half_up(
    value: float,
) -> int:
    """0以上の値を0.5以上で切り上げる。"""
    return int(value + 0.5)


def convert_positive_label(
    label_line: str,
    image_width: int,
    image_height: int,
    label_path: Path,
) -> tuple[int, int]:
    """1行のYOLOラベルを元解像度の中心座標へ変換する。"""
    parts = label_line.split()

    if len(parts) != 5:
        raise ValueError(
            "YOLOラベルは5列必要です: "
            f"{label_path}"
        )

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])
    except ValueError as error:
        raise ValueError(
            "YOLOラベルを数値へ変換できません: "
            f"{label_path}"
        ) from error

    if class_id != 0:
        raise ValueError(
            "ball以外のクラスが含まれています: "
            f"class_id={class_id}, "
            f"file={label_path}"
        )

    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
    ):
        raise ValueError(
            "中心座標が0〜1の範囲外です: "
            f"{label_path}"
        )

    if not (
        0.0 < box_width <= 1.0
        and 0.0 < box_height <= 1.0
    ):
        raise ValueError(
            "矩形サイズが正しい範囲にありません: "
            f"{label_path}"
        )

    x_pixel = round_half_up(
        x_center * image_width
    )
    y_pixel = round_half_up(
        y_center * image_height
    )

    if not (
        0 <= x_pixel < image_width
        and 0 <= y_pixel < image_height
    ):
        raise ValueError(
            "変換後の中心座標が画像外です: "
            f"X={x_pixel}, Y={y_pixel}, "
            f"file={label_path}"
        )

    return x_pixel, y_pixel


def convert_frame(
    frame_index: int,
    image_dir: Path,
    label_mapping: dict[int, Path],
) -> dict[str, int]:
    """1フレーム分の画像とラベルを検証して変換する。"""
    image_path = (
        image_dir
        / f"frame_{frame_index:06d}.png"
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"対応画像がありません: "
            f"{image_path}"
        )

    if frame_index not in label_mapping:
        raise FileNotFoundError(
            "対応ラベルがありません: "
            f"frame={frame_index}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"画像を開けません: {image_path}"
        )

    image_height, image_width = (
        image.shape[:2]
    )
    label_path = label_mapping[frame_index]
    label_lines = read_nonempty_lines(
        label_path
    )

    if len(label_lines) > 1:
        raise ValueError(
            "1フレームに複数の矩形があります: "
            f"{label_path}"
        )

    if not label_lines:
        return {
            "Frame": frame_index,
            "Visibility": 0,
            "X": 0,
            "Y": 0,
        }

    x_pixel, y_pixel = convert_positive_label(
        label_lines[0],
        image_width,
        image_height,
        label_path,
    )

    return {
        "Frame": frame_index,
        "Visibility": 1,
        "X": x_pixel,
        "Y": y_pixel,
    }


def main() -> None:
    """指定範囲を変換し、TrackNetV3用CSVへ保存する。"""
    args = parse_args()
    validate_paths(args)
    read_classes(args.classes)

    label_mapping = build_label_mapping(
        args.label_dir
    )
    expected_frame_indices = list(
        range(
            args.start_frame,
            args.end_frame_exclusive,
        )
    )

    unexpected_frames = sorted(
        set(label_mapping)
        - set(expected_frame_indices)
    )

    if unexpected_frames:
        raise ValueError(
            "指定範囲外のラベルがあります: "
            f"{unexpected_frames}"
        )

    rows = [
        convert_frame(
            frame_index,
            args.image_dir,
            label_mapping,
        )
        for frame_index in expected_frame_indices
    ]

    if len(rows) != len(
        expected_frame_indices
    ):
        raise RuntimeError(
            "変換後の行数が予定と一致しません"
        )

    positive_count = sum(
        row["Visibility"] == 1
        for row in rows
    )
    negative_count = sum(
        row["Visibility"] == 0
        for row in rows
    )

    if positive_count + negative_count != len(
        rows
    ):
        raise RuntimeError(
            "正例数と負例数の合計が"
            "全行数と一致しません"
        )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "Frame",
                "Visibility",
                "X",
                "Y",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "YOLOラベルをTrackNetV3用CSVへ"
        "変換しました"
    )
    print(f"出力先: {args.output_csv}")
    print(f"総フレーム数: {len(rows)}")
    print(f"ボールあり: {positive_count}")
    print(f"ボールなし: {negative_count}")
    print()
    print("Frame  Visibility     X     Y")

    for row in rows:
        print(
            f"{row['Frame']:>5}"
            f"{row['Visibility']:>12}"
            f"{row['X']:>6}"
            f"{row['Y']:>6}"
        )


if __name__ == "__main__":
    main()