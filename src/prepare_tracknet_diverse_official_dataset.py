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

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tracknet_official_diverse_v1"
)

OFFICIAL_REFERENCE_COMMIT = (
    "77c123ad4dd449b7d275f16cc43f316ba5b54042"
)

SEQUENCE_LENGTH = 8
TRAIN_SLIDING_STEP = 1
VAL_SLIDING_STEP = 8

CSV_FIELDS = [
    "Frame",
    "Visibility",
    "X",
    "Y",
]

MAPPING_FIELDS = [
    "split",
    "official_match",
    "official_rally",
    "local_frame",
    "source_frame",
    "source_image",
    "official_image",
]

RALLY_CONFIGS = (
    {
        "split": "train",
        "official_match": 1,
        "official_rally": 1,
        "source_name": "match01_train",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "train"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_train_final"
            / "converted"
            / "tracknet_labels.csv"
        ),
        "image_name_template": "frame_{frame:06d}.png",
        "expected_count": 128,
        "expected_positive": 126,
        "expected_empty": 2,
    },
    {
        "split": "train",
        "official_match": 2,
        "official_rally": 1,
        "source_name": "match04_rally_01",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "train"
            / "match04"
            / "rally_01"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "train"
            / "match04"
            / "rally_01"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match04_r01_frame_{frame:06d}.png"
        ),
        "expected_count": 64,
        "expected_positive": 56,
        "expected_empty": 8,
    },
    {
        "split": "train",
        "official_match": 2,
        "official_rally": 2,
        "source_name": "match04_rally_02",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "train"
            / "match04"
            / "rally_02"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "train"
            / "match04"
            / "rally_02"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match04_r02_frame_{frame:06d}.png"
        ),
        "expected_count": 64,
        "expected_positive": 64,
        "expected_empty": 0,
    },
    {
        "split": "val",
        "official_match": 1,
        "official_rally": 1,
        "source_name": "match01_val",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "val"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_val_final"
            / "converted"
            / "tracknet_labels.csv"
        ),
        "image_name_template": "frame_{frame:06d}.png",
        "expected_count": 120,
        "expected_positive": 118,
        "expected_empty": 2,
    },
    {
        "split": "val",
        "official_match": 2,
        "official_rally": 1,
        "source_name": "match06_rally_01",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "val"
            / "match06"
            / "rally_01"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "val"
            / "match06"
            / "rally_01"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match06_r01_frame_{frame:06d}.png"
        ),
        "expected_count": 32,
        "expected_positive": 32,
        "expected_empty": 0,
    },
    {
        "split": "val",
        "official_match": 2,
        "official_rally": 2,
        "source_name": "match06_rally_02",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "val"
            / "match06"
            / "rally_02"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "val"
            / "match06"
            / "rally_02"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match06_r02_frame_{frame:06d}.png"
        ),
        "expected_count": 32,
        "expected_positive": 27,
        "expected_empty": 5,
    },
    {
        "split": "val",
        "official_match": 2,
        "official_rally": 3,
        "source_name": "match06_rally_03",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "val"
            / "match06"
            / "rally_03"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "val"
            / "match06"
            / "rally_03"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match06_r03_frame_{frame:06d}.png"
        ),
        "expected_count": 32,
        "expected_positive": 31,
        "expected_empty": 1,
    },
    {
        "split": "val",
        "official_match": 2,
        "official_rally": 4,
        "source_name": "match06_rally_04",
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "val"
            / "match06"
            / "rally_04"
            / "images"
        ),
        "csv_path": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "val"
            / "match06"
            / "rally_04"
            / "tracknet_labels.csv"
        ),
        "image_name_template": (
            "match06_r04_frame_{frame:06d}.png"
        ),
        "expected_count": 32,
        "expected_positive": 32,
        "expected_empty": 0,
    },
)


def parse_args() -> argparse.Namespace:
    """公式互換データセットの出力条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "既存match01と新しいmatch04・match06を、"
            "TrackNetV3公式Dataset互換形式へまとめる"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="公式互換データセットの新規出力先",
    )
    parser.add_argument(
        "--median-chunk-rows",
        type=int,
        default=32,
        help=(
            "中央値背景を一度に計算する画像の行数。"
            "小さくするとメモリ使用量が減る"
        ),
    )
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def read_tracknet_rows(
    csv_path: Path,
) -> list[dict[str, int]]:
    """TrackNet CSVを読み、値と連続性を検証する。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"TrackNet CSVがありません: {csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                "TrackNet CSVの列が一致しません: "
                f"期待={CSV_FIELDS}, "
                f"実際={reader.fieldnames}, "
                f"CSV={csv_path}"
            )

        rows = [
            {
                "Frame": int(row["Frame"]),
                "Visibility": int(row["Visibility"]),
                "X": int(row["X"]),
                "Y": int(row["Y"]),
            }
            for row in reader
        ]

    if not rows:
        raise ValueError(
            f"TrackNet CSVが空です: {csv_path}"
        )

    rows.sort(key=lambda row: row["Frame"])

    frame_numbers = [
        row["Frame"]
        for row in rows
    ]

    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError(
            f"Frame番号が重複しています: {csv_path}"
        )

    expected_frames = list(
        range(
            frame_numbers[0],
            frame_numbers[-1] + 1,
        )
    )

    if frame_numbers != expected_frames:
        raise ValueError(
            "ラリー内のFrame番号が連続していません: "
            f"{csv_path}"
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

        if visibility == 1 and (x < 0 or y < 0):
            raise ValueError(
                "可視座標に負数があります: "
                f"{row}"
            )

    return rows


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """辞書の一覧をCSVとして保存する。"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def calculate_sequence_count(
    frame_count: int,
    sliding_step: int,
) -> int:
    """1ラリー内で作れる8フレーム系列数を計算する。"""
    if frame_count < SEQUENCE_LENGTH:
        return 0

    return (
        (frame_count - SEQUENCE_LENGTH)
        // sliding_step
        + 1
    )


def prepare_rally(
    config: dict,
    staging_root: Path,
) -> dict:
    """1ラリーを公式Dataset形式へ変換する。"""
    split = str(config["split"])
    official_match = int(
        config["official_match"]
    )
    official_rally = int(
        config["official_rally"]
    )
    source_name = str(config["source_name"])
    image_dir = Path(config["image_dir"])
    csv_path = Path(config["csv_path"])
    image_name_template = str(
        config["image_name_template"]
    )
    expected_count = int(
        config["expected_count"]
    )
    expected_positive = int(
        config["expected_positive"]
    )
    expected_empty = int(
        config["expected_empty"]
    )

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"画像フォルダがありません: {image_dir}"
        )

    rows = read_tracknet_rows(csv_path)

    if len(rows) != expected_count:
        raise ValueError(
            f"{source_name}のCSV行数が想定外です: "
            f"期待={expected_count}, 実際={len(rows)}"
        )

    positive_count = sum(
        row["Visibility"] == 1
        for row in rows
    )
    empty_count = len(rows) - positive_count

    if positive_count != expected_positive:
        raise ValueError(
            f"{source_name}の正例数が想定外です: "
            f"期待={expected_positive}, "
            f"実際={positive_count}"
        )

    if empty_count != expected_empty:
        raise ValueError(
            f"{source_name}の空ラベル数が想定外です: "
            f"期待={expected_empty}, "
            f"実際={empty_count}"
        )

    official_match_dir = (
        staging_root
        / split
        / f"match{official_match}"
    )
    official_frame_dir = (
        official_match_dir
        / "frame"
        / str(official_rally)
    )
    official_csv_path = (
        official_match_dir
        / "csv"
        / f"{official_rally}_ball.csv"
    )

    official_frame_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    official_rows = []
    mapping_rows = []
    expected_source_paths = []
    image_shape = None

    for local_frame, source_row in enumerate(rows):
        source_frame = source_row["Frame"]
        source_file_name = (
            image_name_template.format(
                frame=source_frame
            )
        )
        source_image_path = (
            image_dir / source_file_name
        )

        if not source_image_path.is_file():
            raise FileNotFoundError(
                "CSVに対応する画像がありません: "
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
                "同じラリー内で画像shapeが異なります: "
                f"期待={image_shape}, "
                f"実際={current_shape}, "
                f"画像={source_image_path}"
            )

        image_height, image_width = current_shape[:2]

        if source_row["Visibility"] == 1:
            x = source_row["X"]
            y = source_row["Y"]

            if (
                x >= image_width
                or y >= image_height
            ):
                raise ValueError(
                    "可視座標が画像範囲外です: "
                    f"行={source_row}, "
                    f"画像={source_image_path}"
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
                "official_match": official_match,
                "official_rally": official_rally,
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

    actual_source_paths = sorted(
        path.resolve()
        for path in image_dir.glob("*.png")
    )

    if sorted(expected_source_paths) != (
        actual_source_paths
    ):
        raise ValueError(
            "画像フォルダとCSVが双方向に一致しません: "
            f"{source_name}"
        )

    write_csv(
        official_csv_path,
        official_rows,
        CSV_FIELDS,
    )

    if image_shape is None:
        raise RuntimeError(
            f"画像shapeを取得できません: {source_name}"
        )

    sliding_step = (
        TRAIN_SLIDING_STEP
        if split == "train"
        else VAL_SLIDING_STEP
    )
    sequence_count = calculate_sequence_count(
        len(rows),
        sliding_step,
    )

    return {
        "split": split,
        "official_match": official_match,
        "official_rally": official_rally,
        "source_name": source_name,
        "frame_count": len(rows),
        "positive_count": positive_count,
        "empty_count": empty_count,
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "sequence_count": sequence_count,
        "source_csv": str(
            csv_path.relative_to(PROJECT_ROOT)
        ),
        "source_csv_sha256": (
            calculate_sha256(csv_path)
        ),
        "mapping_rows": mapping_rows,
    }


def collect_match_frame_paths(
    match_dir: Path,
    rally_ids: list[int],
) -> list[Path]:
    """指定matchの全ラリー画像を順番に集める。"""
    frame_paths = []

    for rally_id in rally_ids:
        rally_dir = (
            match_dir
            / "frame"
            / str(rally_id)
        )

        if not rally_dir.is_dir():
            raise FileNotFoundError(
                f"公式rallyフォルダがありません: {rally_dir}"
            )

        rally_paths = sorted(
            rally_dir.glob("*.png"),
            key=lambda path: int(path.stem),
        )

        if not rally_paths:
            raise ValueError(
                f"公式rally画像が空です: {rally_dir}"
            )

        expected_names = [
            f"{index}.png"
            for index in range(len(rally_paths))
        ]
        actual_names = [
            path.name
            for path in rally_paths
        ]

        if actual_names != expected_names:
            raise ValueError(
                "公式rally画像が0から連続していません: "
                f"{rally_dir}"
            )

        frame_paths.extend(rally_paths)

    return frame_paths


def build_match_median(
    frame_paths: list[Path],
    output_path: Path,
    temporary_directory: Path,
    chunk_rows: int,
) -> tuple[int, int, int]:
    """1試合の全ラリーから中央値背景を作成する。"""
    if not frame_paths:
        raise ValueError(
            "中央値を作る画像がありません"
        )

    first_image = cv2.imread(
        str(frame_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_image is None:
        raise ValueError(
            f"画像を読み込めません: {frame_paths[0]}"
        )

    height, width, channels = first_image.shape

    temporary_array_path = (
        temporary_directory
        / (
            output_path.parent.parent.name
            + "_"
            + output_path.parent.name
            + "_median_stack.dat"
        )
    )

    frame_stack = np.memmap(
        temporary_array_path,
        dtype=np.uint8,
        mode="w+",
        shape=(
            len(frame_paths),
            height,
            width,
            channels,
        ),
    )

    try:
        for frame_index, frame_path in enumerate(
            frame_paths
        ):
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
                    "同じmatch内で画像shapeが異なります: "
                    f"期待={first_image.shape}, "
                    f"実際={frame_bgr.shape}, "
                    f"画像={frame_path}"
                )

            frame_stack[frame_index] = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

        frame_stack.flush()

        median_image = np.empty(
            (height, width, channels),
            dtype=np.uint8,
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
                f"{output_path.parent}: "
                f"中央値計算 {end_row}/{height}行"
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


def main() -> None:
    """多様化版の公式互換データセットを生成する。"""
    args = parse_args()
    output_root = args.output_root.resolve()
    chunk_rows = args.median_chunk_rows

    if chunk_rows <= 0:
        raise ValueError(
            "--median-chunk-rowsは1以上にしてください"
        )

    if output_root.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{output_root}"
        )

    output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{output_root.name}_staging_"
            ),
            dir=output_root.parent,
        )
    )
    staging_root = (
        staging_parent / output_root.name
    )
    staging_root.mkdir()

    rally_summaries = []
    mapping_rows = []

    print(f"一時出力先: {staging_root}")

    try:
        for config in RALLY_CONFIGS:
            summary = prepare_rally(
                config,
                staging_root,
            )
            mapping_rows.extend(
                summary.pop("mapping_rows")
            )
            rally_summaries.append(summary)

            print(
                f"{summary['split']} / "
                f"match{summary['official_match']} / "
                f"rally {summary['official_rally']}: "
                f"frames={summary['frame_count']}, "
                f"positive={summary['positive_count']}, "
                f"empty={summary['empty_count']}, "
                f"sequences={summary['sequence_count']}"
            )

        write_csv(
            staging_root / "frame_mapping.csv",
            mapping_rows,
            MAPPING_FIELDS,
        )

        match_groups = {}

        for summary in rally_summaries:
            group_key = (
                summary["split"],
                summary["official_match"],
            )
            match_groups.setdefault(
                group_key,
                [],
            ).append(
                summary["official_rally"]
            )

        median_summaries = []

        for (
            split,
            official_match,
        ), rally_ids in sorted(
            match_groups.items()
        ):
            if split == "val" and official_match == 1:
                continue

            match_dir = (
                staging_root
                / split
                / f"match{official_match}"
            )
            sorted_rally_ids = sorted(rally_ids)
            frame_paths = collect_match_frame_paths(
                match_dir,
                sorted_rally_ids,
            )
            median_path = (
                match_dir / "median.npz"
            )
            median_shape = build_match_median(
                frame_paths,
                median_path,
                staging_root,
                chunk_rows,
            )

            median_summaries.append(
                {
                    "split": split,
                    "official_match": (
                        official_match
                    ),
                    "source": (
                        "all frames from rallies "
                        + ", ".join(
                            str(rally_id)
                            for rally_id
                            in sorted_rally_ids
                        )
                    ),
                    "frame_count": len(frame_paths),
                    "shape": list(median_shape),
                    "sha256": calculate_sha256(
                        median_path
                    ),
                }
            )

        train_match1_median = (
            staging_root
            / "train"
            / "match1"
            / "median.npz"
        )
        val_match1_median = (
            staging_root
            / "val"
            / "match1"
            / "median.npz"
        )

        if not train_match1_median.is_file():
            raise FileNotFoundError(
                "train/match1のmedian.npzがありません"
            )

        shutil.copy2(
            train_match1_median,
            val_match1_median,
        )

        if (
            calculate_sha256(train_match1_median)
            != calculate_sha256(val_match1_median)
        ):
            raise RuntimeError(
                "match01のtrain・val中央値が一致しません"
            )

        median_summaries.append(
            {
                "split": "val",
                "official_match": 1,
                "source": (
                    "copied from train/match1; "
                    "same source match01 environment"
                ),
                "frame_count": 128,
                "shape": next(
                    item["shape"]
                    for item in median_summaries
                    if (
                        item["split"] == "train"
                        and item["official_match"] == 1
                    )
                ),
                "sha256": calculate_sha256(
                    val_match1_median
                ),
            }
        )

        train_summaries = [
            summary
            for summary in rally_summaries
            if summary["split"] == "train"
        ]
        val_summaries = [
            summary
            for summary in rally_summaries
            if summary["split"] == "val"
        ]

        train_frame_count = sum(
            summary["frame_count"]
            for summary in train_summaries
        )
        val_frame_count = sum(
            summary["frame_count"]
            for summary in val_summaries
        )
        train_sequence_count = sum(
            summary["sequence_count"]
            for summary in train_summaries
        )
        val_sequence_count = sum(
            summary["sequence_count"]
            for summary in val_summaries
        )

        if train_frame_count != 256:
            raise RuntimeError(
                "trainフレーム数が256ではありません: "
                f"{train_frame_count}"
            )

        if val_frame_count != 248:
            raise RuntimeError(
                "valフレーム数が248ではありません: "
                f"{val_frame_count}"
            )

        if train_sequence_count != 235:
            raise RuntimeError(
                "train系列数が235ではありません: "
                f"{train_sequence_count}"
            )

        if val_sequence_count != 31:
            raise RuntimeError(
                "val系列数が31ではありません: "
                f"{val_sequence_count}"
            )

        metadata = {
            "dataset_name": (
                "tracknet_official_diverse_v1"
            ),
            "official_reference_commit": (
                OFFICIAL_REFERENCE_COMMIT
            ),
            "median_dtype": "uint8",
            "sequence_length": SEQUENCE_LENGTH,
            "train_sliding_step": (
                TRAIN_SLIDING_STEP
            ),
            "val_sliding_step": (
                VAL_SLIDING_STEP
            ),
            "train_frame_count": (
                train_frame_count
            ),
            "val_frame_count": val_frame_count,
            "expected_train_sequences": (
                train_sequence_count
            ),
            "expected_val_sequences": (
                val_sequence_count
            ),
            "rallies": rally_summaries,
            "medians": median_summaries,
            "preparation_script": str(
                Path(__file__).resolve()
            ),
            "preparation_script_sha256": (
                calculate_sha256(
                    Path(__file__).resolve()
                )
            ),
            "design_notes": [
                (
                    "match01 train and val remain in "
                    "separate splits"
                ),
                (
                    "match04 is used only for train"
                ),
                (
                    "match06 is used only for val"
                ),
                (
                    "8-frame sequences never cross "
                    "rally boundaries"
                ),
                (
                    "match05 is reserved for a future "
                    "unseen external test"
                ),
            ],
        }

        metadata_path = (
            staging_root / "metadata.json"
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

        shutil.move(
            str(staging_root),
            str(output_root),
        )
    finally:
        shutil.rmtree(
            staging_parent,
            ignore_errors=True,
        )

    print(
        "TrackNetV3多様化版の公式互換"
        "データセットを作成しました"
    )
    print(f"出力先: {output_root}")
    print(
        f"train: frames={train_frame_count}, "
        f"sequences={train_sequence_count}"
    )
    print(
        f"val: frames={val_frame_count}, "
        f"sequences={val_sequence_count}"
    )
    print(
        "対応表: "
        f"{output_root / 'frame_mapping.csv'}"
    )
    print(
        "再現情報: "
        f"{output_root / 'metadata.json'}"
    )


if __name__ == "__main__":
    main()