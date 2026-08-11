import csv
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO_PATH = (
    PROJECT_ROOT / "data" / "raw" / "match01.mp4"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
)
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"

SPLIT_CONFIGS = {
    "train": {
        "start_second": 48.0,
        "frame_count": 128,
    },
    "val": {
        "start_second": 1.0,
        "frame_count": 120,
    },
}


def validate_inputs() -> None:
    """入力動画と出力先の状態を検証する。"""
    if not SOURCE_VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"元動画が見つかりません: "
            f"{SOURCE_VIDEO_PATH}"
        )

    if OUTPUT_ROOT.exists():
        existing_files = [
            path
            for path in OUTPUT_ROOT.rglob("*")
            if path.is_file()
        ]

        if existing_files:
            raise FileExistsError(
                "出力先に既存ファイルがあります。"
                "意図しない上書きを防ぐため停止します: "
                f"{OUTPUT_ROOT}"
            )


def calculate_frame_ranges(
    fps: float,
) -> dict[str, dict[str, int | float]]:
    """秒数と枚数から各splitのフレーム範囲を求める。"""
    frame_ranges = {}

    for split_name, config in SPLIT_CONFIGS.items():
        start_frame = int(
            config["start_second"] * fps
        )
        frame_count = int(
            config["frame_count"]
        )
        end_frame_exclusive = (
            start_frame + frame_count
        )

        frame_ranges[split_name] = {
            "start_second": config["start_second"],
            "start_frame": start_frame,
            "end_frame_exclusive": (
                end_frame_exclusive
            ),
            "frame_count": frame_count,
        }

    return frame_ranges


def validate_frame_ranges(
    frame_ranges: dict[
        str,
        dict[str, int | float],
    ],
    total_frames: int,
) -> None:
    """範囲外指定とsplit同士の重複を検証する。"""
    range_items = list(frame_ranges.items())

    for split_name, frame_range in range_items:
        start_frame = int(
            frame_range["start_frame"]
        )
        end_frame = int(
            frame_range["end_frame_exclusive"]
        )

        if start_frame < 0:
            raise ValueError(
                f"{split_name}の開始フレームが"
                "0未満です"
            )

        if end_frame > total_frames:
            raise ValueError(
                f"{split_name}の終了フレームが"
                "動画範囲を超えています: "
                f"{end_frame} > {total_frames}"
            )

    for first_index in range(len(range_items)):
        first_name, first_range = range_items[
            first_index
        ]
        first_start = int(
            first_range["start_frame"]
        )
        first_end = int(
            first_range["end_frame_exclusive"]
        )

        for second_index in range(
            first_index + 1,
            len(range_items),
        ):
            second_name, second_range = (
                range_items[second_index]
            )
            second_start = int(
                second_range["start_frame"]
            )
            second_end = int(
                second_range[
                    "end_frame_exclusive"
                ]
            )

            ranges_overlap = (
                first_start < second_end
                and second_start < first_end
            )

            if ranges_overlap:
                raise ValueError(
                    "splitのフレーム範囲が"
                    "重複しています: "
                    f"{first_name}, {second_name}"
                )


def find_target_split(
    frame_index: int,
    frame_ranges: dict[
        str,
        dict[str, int | float],
    ],
) -> str | None:
    """現在のフレームが属するsplitを返す。"""
    for split_name, frame_range in (
        frame_ranges.items()
    ):
        start_frame = int(
            frame_range["start_frame"]
        )
        end_frame = int(
            frame_range["end_frame_exclusive"]
        )

        if start_frame <= frame_index < end_frame:
            return split_name

    return None


def main() -> None:
    """元動画からパイロット用の連続フレームを抽出する。"""
    validate_inputs()

    video = cv2.VideoCapture(
        str(SOURCE_VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"元動画を開けません: "
            f"{SOURCE_VIDEO_PATH}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0 or total_frames <= 0:
        video.release()
        raise RuntimeError(
            "動画のFPSまたは総フレーム数を"
            "取得できません"
        )

    frame_ranges = calculate_frame_ranges(fps)
    validate_frame_ranges(
        frame_ranges,
        total_frames,
    )

    for split_name in frame_ranges:
        (
            OUTPUT_ROOT / split_name / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    saved_counts = {
        split_name: 0
        for split_name in frame_ranges
    }
    manifest_rows = []
    current_frame_index = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        target_split = find_target_split(
            current_frame_index,
            frame_ranges,
        )

        if target_split is not None:
            file_name = (
                f"frame_"
                f"{current_frame_index:06d}.png"
            )
            relative_path = (
                Path(target_split)
                / "images"
                / file_name
            )
            output_path = (
                OUTPUT_ROOT / relative_path
            )

            save_success = cv2.imwrite(
                str(output_path),
                frame,
            )

            if not save_success:
                video.release()
                raise RuntimeError(
                    f"画像を保存できません: "
                    f"{output_path}"
                )

            manifest_rows.append(
                {
                    "split": target_split,
                    "file_name": file_name,
                    "relative_path": (
                        relative_path.as_posix()
                    ),
                    "source_frame_index": (
                        current_frame_index
                    ),
                    "source_time_seconds": round(
                        current_frame_index / fps,
                        6,
                    ),
                }
            )
            saved_counts[target_split] += 1

        current_frame_index += 1

    video.release()

    for split_name, frame_range in (
        frame_ranges.items()
    ):
        expected_count = int(
            frame_range["frame_count"]
        )
        actual_count = saved_counts[
            split_name
        ]

        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name}の保存枚数が"
                "予定と一致しません: "
                f"予定={expected_count}, "
                f"実際={actual_count}"
            )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "file_name",
                "relative_path",
                "source_frame_index",
                "source_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        "TrackNetV3パイロット用フレームを"
        "抽出しました"
    )
    print(f"入力動画: {SOURCE_VIDEO_PATH}")
    print(f"FPS: {fps:.3f}")
    print(f"総フレーム数: {total_frames}")

    for split_name, frame_range in (
        frame_ranges.items()
    ):
        print()
        print(f"split: {split_name}")
        print(
            "開始フレーム: "
            f"{frame_range['start_frame']}"
        )
        print(
            "終了フレーム（含まない）: "
            f"{frame_range['end_frame_exclusive']}"
        )
        print(
            "保存枚数: "
            f"{saved_counts[split_name]}"
        )

    print()
    print(f"出力先: {OUTPUT_ROOT}")
    print(f"対応表: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()