import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
}


def parse_args() -> argparse.Namespace:
    """検査対象と任意のJSON保存先を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "学習・評価候補動画の識別情報、"
            "映像メタデータ、内容重複を確認する"
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="候補動画を格納したフォルダ",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "検査結果を新規保存するJSON。"
            "省略時はファイルを作成しない"
        ),
    )
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """動画内容を識別するSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def inspect_video(path: Path) -> dict[str, int | float | str]:
    """1本の動画から再現用情報と映像情報を取得する。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "動画検査にはOpenCVが必要です。"
            "VolleyScopeのTrackNet環境で実行してください"
        ) from error

    capture = cv2.VideoCapture(str(path))

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"動画を開けません: {path}"
            )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(
            f"動画サイズが不正です: {path}"
        )

    if fps <= 0:
        raise ValueError(
            f"FPSが不正です: {path}"
        )

    if frame_count <= 0:
        raise ValueError(
            f"フレーム数が不正です: {path}"
        )

    return {
        "file_name": path.name,
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def find_duplicate_groups(
    records: list[dict[str, int | float | str]],
) -> list[dict[str, str | list[str]]]:
    """内容が同じ動画をSHA-256単位でまとめる。"""
    names_by_hash: dict[str, list[str]] = {}

    for record in records:
        sha256 = str(record["sha256"])
        names_by_hash.setdefault(sha256, []).append(
            str(record["file_name"])
        )

    return [
        {
            "sha256": sha256,
            "file_names": sorted(file_names),
        }
        for sha256, file_names in sorted(
            names_by_hash.items()
        )
        if len(file_names) > 1
    ]


def validate_args(args: argparse.Namespace) -> None:
    """入力フォルダとJSON出力先を検証する。"""
    args.video_dir = args.video_dir.resolve()

    if not args.video_dir.is_dir():
        raise FileNotFoundError(
            "動画フォルダが見つかりません: "
            f"{args.video_dir}"
        )

    if args.output_json is None:
        return

    args.output_json = args.output_json.resolve()

    if args.output_json.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{args.output_json}"
        )


def main() -> None:
    """候補動画を全件検査し、結果を表示または保存する。"""
    args = parse_args()
    validate_args(args)

    video_paths = sorted(
        path
        for path in args.video_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if not video_paths:
        raise RuntimeError(
            "候補動画が見つかりません: "
            f"{args.video_dir}"
        )

    records = [
        inspect_video(video_path)
        for video_path in video_paths
    ]
    duplicate_groups = find_duplicate_groups(records)

    result = {
        "video_directory": str(args.video_dir),
        "video_count": len(records),
        "duplicate_content_group_count": len(
            duplicate_groups
        ),
        "duplicate_content_groups": duplicate_groups,
        "videos": records,
    }

    print("候補動画を検査しました")
    print(f"動画フォルダ: {args.video_dir}")
    print(f"動画数: {len(records)}")
    print(
        "内容重複グループ数: "
        f"{len(duplicate_groups)}"
    )

    for record in records:
        print()
        print(f"[{record['file_name']}]")
        print(f"SHA-256: {record['sha256']}")
        print(
            "Size: "
            f"{record['width']}x{record['height']}"
        )
        print(f"FPS: {float(record['fps']):.6f}")
        print(f"Frames: {record['frame_count']}")
        print(
            "Seconds: "
            f"{float(record['duration_seconds']):.3f}"
        )

    if duplicate_groups:
        print()
        print("同一内容の動画:")

        for group in duplicate_groups:
            print(
                f"{group['sha256']}: "
                + ", ".join(group["file_names"])
            )

    if args.output_json is None:
        print()
        print("JSON保存: なし")
        return

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_json.open(
        "x",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            result,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    print()
    print(f"JSON保存先: {args.output_json}")


if __name__ == "__main__":
    main()
