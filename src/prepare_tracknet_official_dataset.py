import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
    / "train"
    / "images"
)
TRAIN_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_train_final"
    / "converted"
    / "tracknet_labels.csv"
)

VAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
    / "val"
    / "images"
)
VAL_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_pilot_v2_val_final"
    / "converted"
    / "tracknet_labels.csv"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tracknet_official_pilot_v2"
)

CSV_FIELDS = [
    "Frame",
    "Visibility",
    "X",
    "Y",
]

MAPPING_FIELDS = [
    "split",
    "local_frame",
    "source_frame",
    "source_image",
    "official_image",
]


def parse_args() -> argparse.Namespace:
    """整形データの出力条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "VolleyScopeのパイロットラベルを、"
            "TrackNetV3公式Dataset互換形式へ整形する"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="公式互換データセットの出力先",
    )
    parser.add_argument(
        "--median-chunk-rows",
        type=int,
        default=32,
        help=(
            "中央値を一度に計算する画像の行数。"
            "小さくするとメモリ使用量が減る"
        ),
    )
    return parser.parse_args()


def read_tracknet_rows(
    csv_path: Path,
) -> list[dict[str, int]]:
    """TrackNetV3用CSVを読み、値と連続性を検証する。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"入力CSVが見つかりません: {csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                "CSV列が期待値と一致しません: "
                f"期待={CSV_FIELDS}, "
                f"実際={reader.fieldnames}"
            )

        rows = [
            {
                "Frame": int(source_row["Frame"]),
                "Visibility": int(
                    source_row["Visibility"]
                ),
                "X": int(source_row["X"]),
                "Y": int(source_row["Y"]),
            }
            for source_row in reader
        ]

    if not rows:
        raise ValueError(
            f"CSVにデータ行がありません: {csv_path}"
        )

    rows.sort(key=lambda row: row["Frame"])

    source_frames = [
        row["Frame"]
        for row in rows
    ]

    if len(source_frames) != len(set(source_frames)):
        raise ValueError(
            f"フレーム番号が重複しています: {csv_path}"
        )

    expected_frames = list(
        range(
            source_frames[0],
            source_frames[-1] + 1,
        )
    )

    if source_frames != expected_frames:
        raise ValueError(
            f"フレーム番号が連続していません: {csv_path}"
        )

    for row in rows:
        visibility = row["Visibility"]
        x = row["X"]
        y = row["Y"]

        if visibility not in (0, 1):
            raise ValueError(
                "Visibilityは0または1である必要があります: "
                f"{row}"
            )

        if visibility == 0 and (x != 0 or y != 0):
            raise ValueError(
                "不可視フレームの座標が0,0ではありません: "
                f"{row}"
            )

    return rows


def write_tracknet_csv(
    output_path: Path,
    rows: list[dict[str, int]],
) -> None:
    """0始まりへ変換したTrackNetV3用CSVを保存する。"""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    split: str,
    source_image_dir: Path,
    source_csv_path: Path,
    staging_root: Path,
) -> tuple[
    int,
    tuple[int, int],
    list[dict[str, str | int]],
]:
    """1つのsplitを公式フォルダ構造へコピーする。"""
    if not source_image_dir.is_dir():
        raise FileNotFoundError(
            "入力画像フォルダが見つかりません: "
            f"{source_image_dir}"
        )

    source_rows = read_tracknet_rows(
        source_csv_path
    )

    source_png_paths = sorted(
        source_image_dir.glob("*.png")
    )

    if len(source_png_paths) != len(source_rows):
        raise ValueError(
            f"{split}の画像数とCSV行数が一致しません: "
            f"画像={len(source_png_paths)}, "
            f"CSV={len(source_rows)}"
        )

    match_dir = (
        staging_root
        / split
        / "match1"
    )
    official_frame_dir = (
        match_dir
        / "frame"
        / "1"
    )
    official_csv_path = (
        match_dir
        / "csv"
        / "1_ball.csv"
    )

    official_frame_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    official_rows = []
    mapping_rows = []
    expected_source_paths = []
    image_shape = None

    for local_frame, source_row in enumerate(
        source_rows
    ):
        source_frame = source_row["Frame"]
        source_image_path = (
            source_image_dir
            / f"frame_{source_frame:06d}.png"
        )

        if not source_image_path.is_file():
            raise FileNotFoundError(
                "CSVに対応する画像が見つかりません: "
                f"{source_image_path}"
            )

        expected_source_paths.append(
            source_image_path.resolve()
        )

        image = cv2.imread(
            str(source_image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"画像を読み込めません: {source_image_path}"
            )

        current_shape = image.shape

        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(
                f"{split}内で画像shapeが一致しません: "
                f"期待={image_shape}, "
                f"実際={current_shape}, "
                f"画像={source_image_path}"
            )

        image_height, image_width = (
            current_shape[:2]
        )

        if source_row["Visibility"] == 1:
            x = source_row["X"]
            y = source_row["Y"]

            if (
                x < 0
                or x >= image_width
                or y < 0
                or y >= image_height
            ):
                raise ValueError(
                    "可視座標が画像範囲外です: "
                    f"{source_row}"
                )

        official_image_path = (
            official_frame_dir
            / f"{local_frame}.png"
        )

        shutil.copy2(
            source_image_path,
            official_image_path,
        )

        official_rows.append(
            {
                "Frame": local_frame,
                "Visibility": (
                    source_row["Visibility"]
                ),
                "X": source_row["X"],
                "Y": source_row["Y"],
            }
        )

        mapping_rows.append(
            {
                "split": split,
                "local_frame": local_frame,
                "source_frame": source_frame,
                "source_image": str(
                    source_image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "official_image": str(
                    official_image_path.relative_to(
                        staging_root
                    )
                ),
            }
        )

    actual_source_paths = [
        path.resolve()
        for path in source_png_paths
    ]

    if sorted(expected_source_paths) != sorted(
        actual_source_paths
    ):
        raise ValueError(
            f"{split}の画像集合とCSVが一致しません"
        )

    write_tracknet_csv(
        official_csv_path,
        official_rows,
    )

    if image_shape is None:
        raise RuntimeError(
            f"{split}の画像shapeを取得できません"
        )

    height, width = image_shape[:2]

    return (
        len(source_rows),
        (width, height),
        mapping_rows,
    )


def write_frame_mapping(
    output_path: Path,
    mapping_rows: list[
        dict[str, str | int]
    ],
) -> None:
    """ローカル番号と元フレーム番号の対応を保存する。"""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=MAPPING_FIELDS,
        )
        writer.writeheader()
        writer.writerows(mapping_rows)


def build_train_median(
    train_frame_dir: Path,
    frame_count: int,
    output_path: Path,
    temporary_directory: Path,
    chunk_rows: int,
) -> tuple[int, int, int]:
    """train画像から中央値背景を分割計算する。"""
    first_image_path = (
        train_frame_dir
        / "0.png"
    )
    first_image = cv2.imread(
        str(first_image_path),
        cv2.IMREAD_COLOR,
    )

    if first_image is None:
        raise ValueError(
            f"0.pngを読み込めません: {first_image_path}"
        )

    height, width, channels = first_image.shape

    temporary_array_path = (
        temporary_directory
        / "median_frame_stack.dat"
    )

    frame_stack = np.memmap(
        temporary_array_path,
        dtype=np.uint8,
        mode="w+",
        shape=(
            frame_count,
            height,
            width,
            channels,
        ),
    )

    try:
        for frame_index in range(frame_count):
            frame_path = (
                train_frame_dir
                / f"{frame_index}.png"
            )
            frame_bgr = cv2.imread(
                str(frame_path),
                cv2.IMREAD_COLOR,
            )

            if frame_bgr is None:
                raise ValueError(
                    f"中央値用画像を読めません: {frame_path}"
                )

            if frame_bgr.shape != first_image.shape:
                raise ValueError(
                    "中央値用画像のshapeが一致しません: "
                    f"{frame_path}"
                )

            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
            frame_stack[frame_index] = frame_rgb

        frame_stack.flush()

        median_image = np.empty(
            (
                height,
                width,
                channels,
            ),
            dtype=np.float64,
        )

        for start_row in range(
            0,
            height,
            chunk_rows,
        ):
            end_row = min(
                start_row + chunk_rows,
                height,
            )

            median_image[start_row:end_row] = (
                np.median(
                    frame_stack[
                        :,
                        start_row:end_row,
                        :,
                        :,
                    ],
                    axis=0,
                )
            )

            print(
                "中央値計算: "
                f"{end_row}/{height}行"
            )

        np.savez(
            output_path,
            median=median_image,
        )
    finally:
        frame_stack.flush()
        del frame_stack

        if temporary_array_path.exists():
            temporary_array_path.unlink()

    return median_image.shape


def calculate_sha256(
    file_path: Path,
) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with file_path.open("rb") as binary_file:
        while True:
            chunk = binary_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def calculate_sequence_count(
    frame_count: int,
    sequence_length: int,
    sliding_step: int,
) -> int:
    """paddingなしで作れる完全系列数を計算する。"""
    if frame_count < sequence_length:
        return 0

    return (
        (frame_count - sequence_length)
        // sliding_step
        + 1
    )


def main() -> None:
    """公式互換データセットを安全に生成する。"""
    args = parse_args()

    output_root = args.output_root.resolve()
    chunk_rows = args.median_chunk_rows

    if chunk_rows <= 0:
        raise ValueError(
            "--median-chunk-rowsには"
            "1以上を指定してください"
        )

    if output_root.exists():
        raise FileExistsError(
            "出力先が既に存在するため停止します: "
            f"{output_root}"
        )

    output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{output_root.name}_staging_"
            ),
            dir=output_root.parent,
        )
    )

    print(f"一時出力先: {staging_root}")

    try:
        (
            train_count,
            train_size,
            train_mapping,
        ) = prepare_split(
            split="train",
            source_image_dir=TRAIN_IMAGE_DIR,
            source_csv_path=TRAIN_CSV_PATH,
            staging_root=staging_root,
        )

        (
            val_count,
            val_size,
            val_mapping,
        ) = prepare_split(
            split="val",
            source_image_dir=VAL_IMAGE_DIR,
            source_csv_path=VAL_CSV_PATH,
            staging_root=staging_root,
        )

        if train_size != val_size:
            raise ValueError(
                "trainとvalの画像サイズが一致しません: "
                f"train={train_size}, val={val_size}"
            )

        write_frame_mapping(
            staging_root / "frame_mapping.csv",
            train_mapping + val_mapping,
        )

        train_match_dir = (
            staging_root
            / "train"
            / "match1"
        )
        val_match_dir = (
            staging_root
            / "val"
            / "match1"
        )

        train_median_path = (
            train_match_dir
            / "median.npz"
        )
        val_median_path = (
            val_match_dir
            / "median.npz"
        )

        median_shape = build_train_median(
            train_frame_dir=(
                train_match_dir
                / "frame"
                / "1"
            ),
            frame_count=train_count,
            output_path=train_median_path,
            temporary_directory=staging_root,
            chunk_rows=chunk_rows,
        )

        shutil.copy2(
            train_median_path,
            val_median_path,
        )

        train_median_hash = calculate_sha256(
            train_median_path
        )
        val_median_hash = calculate_sha256(
            val_median_path
        )

        if train_median_hash != val_median_hash:
            raise RuntimeError(
                "trainとvalのmedian.npzが一致しません"
            )

        sequence_length = 8
        train_sliding_step = 1
        val_sliding_step = 8

        train_sequence_count = (
            calculate_sequence_count(
                train_count,
                sequence_length,
                train_sliding_step,
            )
        )
        val_sequence_count = (
            calculate_sequence_count(
                val_count,
                sequence_length,
                val_sliding_step,
            )
        )

        metadata = {
            "dataset_name": (
                "tracknet_official_pilot_v2"
            ),
            "train_frame_count": train_count,
            "val_frame_count": val_count,
            "image_width": train_size[0],
            "image_height": train_size[1],
            "sequence_length": sequence_length,
            "train_sliding_step": (
                train_sliding_step
            ),
            "val_sliding_step": (
                val_sliding_step
            ),
            "expected_train_sequences": (
                train_sequence_count
            ),
            "expected_val_sequences": (
                val_sequence_count
            ),
            "median_source": (
                "train frames 0-127"
            ),
            "median_shape": list(median_shape),
            "median_sha256": train_median_hash,
            "official_reference_commit": (
                "77c123ad4dd449b7d275f16cc43f316ba5b54042"
            ),
        }

        metadata_path = (
            staging_root
            / "metadata.json"
        )
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        staging_root.replace(output_root)

        print("公式互換データセットを作成しました")
        print(f"出力先: {output_root}")
        print(f"trainフレーム数: {train_count}")
        print(f"valフレーム数: {val_count}")
        print(
            "予想train系列数: "
            f"{train_sequence_count}"
        )
        print(
            "予想val系列数: "
            f"{val_sequence_count}"
        )
        print(
            "中央値shape: "
            f"{median_shape}"
        )
        print(
            "中央値SHA-256: "
            f"{train_median_hash}"
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()