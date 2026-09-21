import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_diverse_v1"
)

VIDEO_CONFIGS = (
    {
        "split": "train",
        "match_name": "match04",
        "video_path": (
            PROJECT_ROOT / "data" / "raw" / "match04.mp4"
        ),
        "rallies": (
            {
                "rally_id": 1,
                "start_second": 10.0,
                "end_second_inclusive": 36.0,
                "target_frame_count": 64,
            },
            {
                "rally_id": 2,
                "start_second": 61.0,
                "end_second_inclusive": 66.0,
                "target_frame_count": 64,
            },
        ),
    },
    {
        "split": "val",
        "match_name": "match06",
        "video_path": (
            PROJECT_ROOT / "data" / "raw" / "match06.mp4"
        ),
        "rallies": (
            {
                "rally_id": 1,
                "start_second": 3.0,
                "end_second_inclusive": 14.0,
                "target_frame_count": 32,
            },
            {
                "rally_id": 2,
                "start_second": 23.0,
                "end_second_inclusive": 32.0,
                "target_frame_count": 32,
            },
            {
                "rally_id": 3,
                "start_second": 49.0,
                "end_second_inclusive": 54.0,
                "target_frame_count": 32,
            },
            {
                "rally_id": 4,
                "start_second": 57.0,
                "end_second_inclusive": 68.0,
                "target_frame_count": 32,
            },
        ),
    },
)


def parse_args() -> argparse.Namespace:
    """多様化用フレームの出力条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "match04とmatch06の指定区間から、"
            "TrackNetV3追加学習用の連続フレームを抽出する"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="抽出画像、manifest、metadataの新規出力先",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルを作らず、抽出予定範囲だけを表示する",
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


def inspect_video(video_path: Path) -> dict[str, int | float]:
    """動画を開き、抽出に必要なメタデータを返す。"""
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"元動画を開けません: {video_path}"
        )

    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            "動画のFPSまたは総フレーム数を取得できません: "
            f"{video_path}"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "動画の画像サイズを取得できません: "
            f"{video_path}"
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def calculate_centered_frame_range(
    fps: float,
    total_frame_count: int,
    start_second: float,
    end_second_inclusive: float,
    target_frame_count: int,
) -> tuple[int, int]:
    """秒区間の中央から指定枚数の連続フレーム範囲を求める。"""
    interval_first_frame = math.ceil(
        start_second * fps
    )
    interval_last_frame = min(
        math.floor(end_second_inclusive * fps),
        total_frame_count - 1,
    )
    available_frame_count = (
        interval_last_frame
        - interval_first_frame
        + 1
    )

    if available_frame_count < target_frame_count:
        raise ValueError(
            "指定区間のフレーム数が抽出予定枚数より少ないです: "
            f"available={available_frame_count}, "
            f"target={target_frame_count}"
        )

    leading_margin = (
        available_frame_count - target_frame_count
    ) // 2
    first_frame = interval_first_frame + leading_margin
    end_frame_exclusive = first_frame + target_frame_count

    return first_frame, end_frame_exclusive


def extract_rally_frames(
    video_path: Path,
    image_dir: Path,
    split: str,
    match_name: str,
    rally_id: int,
    first_frame: int,
    end_frame_exclusive: int,
    fps: float,
    start_second: float,
    end_second_inclusive: float,
) -> list[dict[str, int | float | str]]:
    """一つのラリー候補から連続フレームを保存する。"""
    image_dir.mkdir(parents=True, exist_ok=False)
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"元動画を開けません: {video_path}"
        )

    video.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    manifest_rows = []

    try:
        for source_frame_index in range(
            first_frame,
            end_frame_exclusive,
        ):
            success, frame = video.read()

            if not success:
                raise RuntimeError(
                    "指定区間の途中で動画を読み込めなくなりました: "
                    f"{video_path}, frame={source_frame_index}"
                )

            local_frame_index = (
                source_frame_index - first_frame
            )
            file_name = (
                f"{match_name}_r{rally_id:02d}_"
                f"frame_{local_frame_index:06d}.png"
            )
            image_path = image_dir / file_name

            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(
                    f"画像を保存できませんでした: {image_path}"
                )

            manifest_rows.append(
                {
                    "split": split,
                    "match_name": match_name,
                    "rally_id": rally_id,
                    "file_name": file_name,
                    "relative_path": "",
                    "local_frame_index": local_frame_index,
                    "source_video": video_path.name,
                    "source_frame_index": source_frame_index,
                    "source_time_seconds": round(
                        source_frame_index / fps,
                        6,
                    ),
                    "requested_start_second": start_second,
                    "requested_end_second_inclusive": (
                        end_second_inclusive
                    ),
                }
            )
    finally:
        video.release()

    return manifest_rows


def validate_manifest_rows(
    rows: list[dict[str, int | float | str]],
    output_root: Path,
) -> None:
    """件数、ファイル、連続性、名前の重複を検証する。"""
    expected_total = sum(
        int(rally["target_frame_count"])
        for video_config in VIDEO_CONFIGS
        for rally in video_config["rallies"]
    )

    if len(rows) != expected_total:
        raise RuntimeError(
            "抽出総数が予定と一致しません: "
            f"expected={expected_total}, actual={len(rows)}"
        )

    file_names = [str(row["file_name"]) for row in rows]

    if len(file_names) != len(set(file_names)):
        raise RuntimeError(
            "抽出画像のファイル名が重複しています"
        )

    grouped_rows: dict[
        tuple[str, str, int],
        list[dict[str, int | float | str]],
    ] = {}

    for row in rows:
        key = (
            str(row["split"]),
            str(row["match_name"]),
            int(row["rally_id"]),
        )
        grouped_rows.setdefault(key, []).append(row)

        image_path = output_root / str(row["relative_path"])

        if not image_path.is_file():
            raise FileNotFoundError(
                f"抽出画像がありません: {image_path}"
            )

    for key, rally_rows in grouped_rows.items():
        local_indices = [
            int(row["local_frame_index"])
            for row in rally_rows
        ]
        source_indices = [
            int(row["source_frame_index"])
            for row in rally_rows
        ]

        if local_indices != list(range(len(rally_rows))):
            raise RuntimeError(
                "ラリー内のローカル番号が連続していません: "
                f"{key}"
            )

        expected_source_indices = list(
            range(
                source_indices[0],
                source_indices[0] + len(rally_rows),
            )
        )

        if source_indices != expected_source_indices:
            raise RuntimeError(
                "ラリー内の元フレーム番号が連続していません: "
                f"{key}"
            )


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    """抽出画像と元動画の対応表を保存する。"""
    fieldnames = [
        "split",
        "match_name",
        "rally_id",
        "file_name",
        "relative_path",
        "local_frame_index",
        "source_video",
        "source_frame_index",
        "source_time_seconds",
        "requested_start_second",
        "requested_end_second_inclusive",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def make_serializable_configuration() -> list[dict]:
    """Pathを文字列へ変換した抽出設定を返す。"""
    configuration = []

    for video_config in VIDEO_CONFIGS:
        configuration.append(
            {
                "split": video_config["split"],
                "match_name": video_config["match_name"],
                "video_path": str(
                    Path(video_config["video_path"]).resolve()
                ),
                "rallies": [
                    dict(rally)
                    for rally in video_config["rallies"]
                ],
            }
        )

    return configuration


def print_extraction_plan(
    video_records: list[dict[str, int | float | str]],
) -> None:
    """動画ごとの抽出予定範囲を表示する。"""
    print("=== TrackNetV3多様化用フレーム抽出計画 ===")
    total_frame_count = 0

    for video_config, video_record in zip(
        VIDEO_CONFIGS,
        video_records,
        strict=True,
    ):
        split = str(video_config["split"])
        match_name = str(video_config["match_name"])
        fps = float(video_record["fps"])
        source_frame_count = int(
            video_record["frame_count"]
        )

        print()
        print(
            f"[{split} / {match_name}] "
            f"FPS={fps:.6f}"
        )

        for rally in video_config["rallies"]:
            rally_id = int(rally["rally_id"])
            start_second = float(
                rally["start_second"]
            )
            end_second_inclusive = float(
                rally["end_second_inclusive"]
            )
            target_frame_count = int(
                rally["target_frame_count"]
            )
            (
                first_frame,
                end_frame_exclusive,
            ) = calculate_centered_frame_range(
                fps,
                source_frame_count,
                start_second,
                end_second_inclusive,
                target_frame_count,
            )
            last_frame = end_frame_exclusive - 1
            total_frame_count += target_frame_count

            print(
                f"rally {rally_id}: "
                f"指定={start_second:.3f}〜"
                f"{end_second_inclusive:.3f}秒（終了を含む）, "
                f"抽出={first_frame}〜{last_frame}, "
                f"実時刻={first_frame / fps:.3f}〜"
                f"{last_frame / fps:.3f}秒, "
                f"枚数={target_frame_count}"
            )

    print()
    print(f"抽出予定総数: {total_frame_count}枚")


def main() -> None:
    """多様化用の連続フレームと再現情報を作成する。"""
    args = parse_args()
    output_root = args.output_root.resolve()

    video_records = []

    for video_config in VIDEO_CONFIGS:
        video_path = Path(
            video_config["video_path"]
        ).resolve()

        if not video_path.is_file():
            raise FileNotFoundError(
                f"元動画がありません: {video_path}"
            )

        metadata = inspect_video(video_path)
        video_records.append(
            {
                "path": str(video_path),
                "sha256": calculate_sha256(video_path),
                **metadata,
            }
        )

    print_extraction_plan(video_records)

    if args.dry_run:
        print()
        print("dry-runのためファイルは作成していません")
        return

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
            prefix=".tracknet_diverse_v1_staging_",
            dir=output_root.parent,
        )
    )
    staging_root = staging_parent / output_root.name
    staging_root.mkdir()
    manifest_rows = []

    try:
        for video_config, video_record in zip(
            VIDEO_CONFIGS,
            video_records,
            strict=True,
        ):
            split = str(video_config["split"])
            match_name = str(
                video_config["match_name"]
            )
            video_path = Path(video_record["path"])
            fps = float(video_record["fps"])
            total_frame_count = int(
                video_record["frame_count"]
            )

            for rally in video_config["rallies"]:
                rally_id = int(rally["rally_id"])
                start_second = float(
                    rally["start_second"]
                )
                end_second_inclusive = float(
                    rally["end_second_inclusive"]
                )
                target_frame_count = int(
                    rally["target_frame_count"]
                )
                (
                    first_frame,
                    end_frame_exclusive,
                ) = calculate_centered_frame_range(
                    fps,
                    total_frame_count,
                    start_second,
                    end_second_inclusive,
                    target_frame_count,
                )
                relative_image_dir = (
                    Path(split)
                    / match_name
                    / f"rally_{rally_id:02d}"
                    / "images"
                )
                image_dir = (
                    staging_root / relative_image_dir
                )
                rally_rows = extract_rally_frames(
                    video_path,
                    image_dir,
                    split,
                    match_name,
                    rally_id,
                    first_frame,
                    end_frame_exclusive,
                    fps,
                    start_second,
                    end_second_inclusive,
                )

                for row in rally_rows:
                    row["relative_path"] = (
                        relative_image_dir
                        / str(row["file_name"])
                    ).as_posix()

                manifest_rows.extend(rally_rows)

        validate_manifest_rows(
            manifest_rows,
            staging_root,
        )
        write_manifest(
            staging_root / "manifest.csv",
            manifest_rows,
        )

        metadata = {
            "dataset": "tracknet_diverse_v1",
            "selection_method": (
                "centered consecutive frames within each "
                "inclusive time interval"
            ),
            "total_frame_count": len(manifest_rows),
            "split_frame_counts": {
                split: sum(
                    row["split"] == split
                    for row in manifest_rows
                )
                for split in ("train", "val")
            },
            "videos": video_records,
            "configuration": (
                make_serializable_configuration()
            ),
        }

        with (
            staging_root / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                ensure_ascii=False,
                indent=2,
            )
            metadata_file.write("\n")

        shutil.move(str(staging_root), str(output_root))
    finally:
        shutil.rmtree(
            staging_parent,
            ignore_errors=True,
        )

    print("TrackNetV3多様化用フレームを抽出しました")
    print(f"出力先: {output_root}")
    print(f"総フレーム数: {len(manifest_rows)}")

    for split in ("train", "val"):
        split_rows = [
            row
            for row in manifest_rows
            if row["split"] == split
        ]
        print(f"{split}: {len(split_rows)}枚")

    print(f"対応表: {output_root / 'manifest.csv'}")
    print(f"再現情報: {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
