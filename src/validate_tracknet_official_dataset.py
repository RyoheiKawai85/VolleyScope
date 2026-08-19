import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tracknet_official_pilot_v2"
)

DEFAULT_REFERENCE_ROOT = Path(
    r"C:\GitHub\TrackNetV3-reference"
)

SEQUENCE_LENGTH = 8

EXPECTED_SPLITS = {
    "train": {
        "frame_count": 128,
        "sliding_step": 1,
        "sequence_count": 121,
        "first_files": list(range(0, 8)),
        "last_files": list(range(120, 128)),
    },
    "val": {
        "frame_count": 120,
        "sliding_step": 8,
        "sequence_count": 15,
        "first_files": list(range(0, 8)),
        "last_files": list(range(112, 120)),
    },
}


def parse_args() -> argparse.Namespace:
    """検証対象のパスを取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "公式TrackNetV3 Datasetで"
            "VolleyScopeの整形データを検証する"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="公式互換データセットのルート",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="固定したTrackNetV3公式実装のルート",
    )
    return parser.parse_args()


def import_official_dataset(
    reference_root: Path,
):
    """固定した公式実装からDatasetクラスを読み込む。"""
    if not reference_root.is_dir():
        raise FileNotFoundError(
            "TrackNetV3参照リポジトリが"
            f"見つかりません: {reference_root}"
        )

    reference_path = str(
        reference_root.resolve()
    )

    if reference_path not in sys.path:
        sys.path.insert(
            0,
            reference_path,
        )

    from dataset import (
        Shuttlecock_Trajectory_Dataset,
    )

    return Shuttlecock_Trajectory_Dataset


def expected_file_names(
    frame_numbers: list[int],
) -> list[str]:
    """期待するPNGファイル名を作る。"""
    return [
        f"{frame_number}.png"
        for frame_number in frame_numbers
    ]


def actual_file_names(
    frame_paths,
) -> list[str]:
    """公式Dataset内のパスからファイル名だけを取得する。"""
    return [
        Path(frame_path).name
        for frame_path in frame_paths
    ]


def validate_sample(
    split: str,
    sample,
) -> None:
    """先頭系列の型、shape、値域を検証する。"""
    if len(sample) != 5:
        raise ValueError(
            f"{split}の戻り値要素数が"
            f"5ではありません: {len(sample)}"
        )

    (
        indices,
        model_input,
        heatmaps,
        coordinates,
        visibility,
    ) = sample

    expected_shapes = [
        (SEQUENCE_LENGTH, 2),
        (27, 288, 512),
        (SEQUENCE_LENGTH, 288, 512),
        (SEQUENCE_LENGTH, 2),
        (SEQUENCE_LENGTH,),
    ]

    actual_shapes = [
        indices.shape,
        model_input.shape,
        heatmaps.shape,
        coordinates.shape,
        visibility.shape,
    ]

    if actual_shapes != expected_shapes:
        raise ValueError(
            f"{split}のshapeが期待値と"
            "一致しません: "
            f"期待={expected_shapes}, "
            f"実際={actual_shapes}"
        )

    arrays = {
        "indices": indices,
        "model_input": model_input,
        "heatmaps": heatmaps,
        "coordinates": coordinates,
        "visibility": visibility,
    }

    for array_name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"{split}の{array_name}に"
                "NaNまたは無限値があります"
            )

    if (
        model_input.min() < 0
        or model_input.max() > 1
    ):
        raise ValueError(
            f"{split}の入力値が0〜1の"
            "範囲外です"
        )

    if heatmaps.min() < 0 or heatmaps.max() > 1:
        raise ValueError(
            f"{split}のヒートマップが"
            "0〜1の範囲外です"
        )

    if (
        coordinates.min() < 0
        or coordinates.max() > 1
    ):
        raise ValueError(
            f"{split}の正規化座標が"
            "0〜1の範囲外です"
        )

    unique_visibility = set(
        np.unique(visibility).tolist()
    )

    if not unique_visibility.issubset({0.0, 1.0}):
        raise ValueError(
            f"{split}のVisibilityに"
            "0と1以外が含まれます: "
            f"{unique_visibility}"
        )

    print(f"{split}先頭系列のshapeと値域: 正常")
    print(
        f"{split}入力dtype: "
        f"{model_input.dtype}"
    )
    print(
        f"{split}入力値範囲: "
        f"{model_input.min():.4f}〜"
        f"{model_input.max():.4f}"
    )


def validate_split(
    split: str,
    dataset_class,
    dataset_root: Path,
) -> None:
    """1つのsplitを公式Datasetで検証する。"""
    expected = EXPECTED_SPLITS[split]

    dataset = dataset_class(
        root_dir=str(dataset_root),
        split=split,
        seq_len=SEQUENCE_LENGTH,
        sliding_step=expected["sliding_step"],
        data_mode="heatmap",
        bg_mode="concat",
    )

    if len(dataset) != expected["sequence_count"]:
        raise ValueError(
            f"{split}の系列数が期待値と"
            "一致しません: "
            f"期待={expected['sequence_count']}, "
            f"実際={len(dataset)}"
        )

    frame_files = dataset.data_dict[
        "frame_file"
    ]

    first_names = actual_file_names(
        frame_files[0]
    )
    last_names = actual_file_names(
        frame_files[-1]
    )

    expected_first_names = expected_file_names(
        expected["first_files"]
    )
    expected_last_names = expected_file_names(
        expected["last_files"]
    )

    if first_names != expected_first_names:
        raise ValueError(
            f"{split}の先頭系列が"
            "期待値と一致しません: "
            f"{first_names}"
        )

    if last_names != expected_last_names:
        raise ValueError(
            f"{split}の最終系列が"
            "期待値と一致しません: "
            f"{last_names}"
        )

    missing_files = [
        frame_path
        for sequence_paths in frame_files
        for frame_path in sequence_paths
        if not Path(frame_path).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{split}の系列内に不足画像があります: "
            f"{missing_files[0]}"
        )

    validate_sample(
        split,
        dataset[0],
    )

    print(f"{split}系列数: {len(dataset)}")
    print(
        f"{split}先頭系列: "
        f"{', '.join(first_names)}"
    )
    print(
        f"{split}最終系列: "
        f"{', '.join(last_names)}"
    )


def main() -> None:
    """trainとvalを公式Datasetで検証する。"""
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    reference_root = (
        args.reference_root.resolve()
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "公式互換データセットが"
            f"見つかりません: {dataset_root}"
        )

    dataset_class = import_official_dataset(
        reference_root
    )

    print(f"データセット: {dataset_root}")
    print(f"公式実装: {reference_root}")

    validate_split(
        split="train",
        dataset_class=dataset_class,
        dataset_root=dataset_root,
    )

    validate_split(
        split="val",
        dataset_class=dataset_class,
        dataset_root=dataset_root,
    )

    print(
        "公式TrackNetV3 Datasetとの"
        "互換性検証に合格しました"
    )


if __name__ == "__main__":
    main()