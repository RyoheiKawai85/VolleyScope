import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "match02.mp4"
)

DEFAULT_OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "clips"
    / "external_test_001.mp4"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "external_evaluation_001"
)


def parse_args() -> argparse.Namespace:
    """外部評価データの作成条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "元動画の指定区間から連続評価クリップと"
            "疎なアノテーション画像を作成する"
        ),
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help="入力する元動画",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help="連続評価クリップの出力先",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="アノテーション画像とmanifestの出力先",
    )
    parser.add_argument(
        "--start-second",
        type=float,
        default=15.0,
        help="元動画で評価を開始する秒",
    )
    parser.add_argument(
        "--end-second",
        type=float,
        default=29.0,
        help="元動画で評価を終了する秒。この時刻は含まない",
    )
    parser.add_argument(
        "--target-frame-count",
        type=int,
        default=105,
        help="等間隔に保存するアノテーション画像数",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """入力条件と上書きの危険がないことを確認する。"""
    args.input_video = args.input_video.resolve()
    args.output_video = args.output_video.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.input_video.is_file():
        raise FileNotFoundError(
            f"入力動画が見つかりません: {args.input_video}"
        )

    if args.start_second < 0:
        raise ValueError(
            "--start-secondは0以上にしてください"
        )

    if args.end_second <= args.start_second:
        raise ValueError(
            "--end-secondは--start-secondより後にしてください"
        )

    if args.target_frame_count <= 0:
        raise ValueError(
            "--target-frame-countは1以上にしてください"
        )

    if args.output_video.exists():
        raise FileExistsError(
            "出力動画が既に存在します。"
            f"上書きを防ぐため停止します: {args.output_video}"
        )

    if args.output_dir.exists():
        raise FileExistsError(
            "画像出力先が既に存在します。"
            f"上書きを防ぐため停止します: {args.output_dir}"
        )

    if args.input_video == args.output_video:
        raise ValueError(
            "入力動画と出力動画に同じパスは指定できません"
        )


def make_sample_indices(
    frame_count: int,
    target_count: int,
) -> list[int]:
    """区間全体から重複しないフレーム番号を等間隔に選ぶ。"""
    if target_count > frame_count:
        raise ValueError(
            "抽出枚数がクリップのフレーム数を超えています"
        )

    if target_count == 1:
        return [0]

    sample_indices = [
        round(
            sample_number
            * (frame_count - 1)
            / (target_count - 1)
        )
        for sample_number in range(target_count)
    ]

    if len(set(sample_indices)) != target_count:
        raise RuntimeError(
            "抽出フレーム番号に重複が発生しました"
        )

    return sample_indices


def create_clip(
    input_video: Path,
    output_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    width: int,
    height: int,
) -> int:
    """元動画から半開区間の連続クリップを作成する。"""
    video = cv2.VideoCapture(str(input_video))

    if not video.isOpened():
        raise RuntimeError(
            f"入力動画を開けませんでした: {input_video}"
        )

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        video.release()
        raise RuntimeError(
            f"出力動画を作成できませんでした: {output_video}"
        )

    written_count = 0

    try:
        for _ in range(start_frame, end_frame):
            success, frame = video.read()

            if not success:
                raise RuntimeError(
                    "指定区間の途中で元動画を"
                    "読み込めなくなりました"
                )

            writer.write(frame)
            written_count += 1
    finally:
        video.release()
        writer.release()

    return written_count


def extract_annotation_images(
    clip_path: Path,
    image_dir: Path,
    manifest_path: Path,
    sample_indices: list[int],
    source_start_frame: int,
    fps: float,
) -> int:
    """完成したクリップからアノテーション画像を抽出する。"""
    video = cv2.VideoCapture(str(clip_path))

    if not video.isOpened():
        raise RuntimeError(
            f"作成したクリップを開けませんでした: {clip_path}"
        )

    sample_index_set = set(sample_indices)
    manifest_rows = []
    clip_frame_index = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            if clip_frame_index in sample_index_set:
                file_name = (
                    f"frame_{clip_frame_index:06d}.png"
                )
                image_path = image_dir / file_name

                saved = cv2.imwrite(
                    str(image_path),
                    frame,
                )

                if not saved:
                    raise RuntimeError(
                        f"画像を保存できませんでした: {image_path}"
                    )

                source_frame_index = (
                    source_start_frame
                    + clip_frame_index
                )

                manifest_rows.append(
                    {
                        "file_name": file_name,
                        "frame_index": clip_frame_index,
                        "time_seconds": (
                            clip_frame_index / fps
                        ),
                        "source_frame_index": (
                            source_frame_index
                        ),
                        "source_time_seconds": (
                            source_frame_index / fps
                        ),
                    }
                )

            clip_frame_index += 1
    finally:
        video.release()

    if len(manifest_rows) != len(sample_indices):
        raise RuntimeError(
            "保存画像数が予定数と一致しません。"
            f"予定={len(sample_indices)}, "
            f"実際={len(manifest_rows)}"
        )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "file_name",
                "frame_index",
                "time_seconds",
                "source_frame_index",
                "source_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return clip_frame_index


def main() -> None:
    """外部評価クリップとアノテーション画像を作成する。"""
    args = parse_args()
    validate_args(args)

    source_video = cv2.VideoCapture(
        str(args.input_video)
    )

    if not source_video.isOpened():
        raise RuntimeError(
            f"入力動画を開けませんでした: {args.input_video}"
        )

    fps = source_video.get(
        cv2.CAP_PROP_FPS
    )
    width = int(
        source_video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        source_video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
    total_frames = int(
        source_video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )
    source_video.release()

    if fps <= 0:
        raise RuntimeError(
            "入力動画のFPSを取得できませんでした"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "入力動画の解像度を取得できませんでした"
        )

    start_frame = int(
        args.start_second * fps
    )
    requested_end_frame = int(
        args.end_second * fps
    )
    end_frame = min(
        requested_end_frame,
        total_frames,
    )

    if start_frame >= total_frames:
        raise ValueError(
            "開始時刻が動画の終端以降です"
        )

    clip_frame_count = end_frame - start_frame

    if clip_frame_count <= 0:
        raise ValueError(
            "評価区間にフレームがありません"
        )

    sample_indices = make_sample_indices(
        clip_frame_count,
        args.target_frame_count,
    )

    args.output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".external_evaluation_staging_",
            dir=args.output_dir.parent,
        )
    )
    staging_video = (
        staging_root
        / args.output_video.name
    )
    staging_output_dir = (
        staging_root
        / args.output_dir.name
    )
    staging_image_dir = (
        staging_output_dir
        / "images"
    )
    staging_manifest_path = (
        staging_output_dir
        / "manifest.csv"
    )

    staging_image_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        written_count = create_clip(
            args.input_video,
            staging_video,
            start_frame,
            end_frame,
            fps,
            width,
            height,
        )

        if written_count != clip_frame_count:
            raise RuntimeError(
                "クリップの保存フレーム数が"
                "予定数と一致しません"
            )

        decoded_frame_count = extract_annotation_images(
            staging_video,
            staging_image_dir,
            staging_manifest_path,
            sample_indices,
            start_frame,
            fps,
        )

        if decoded_frame_count != clip_frame_count:
            raise RuntimeError(
                "再読込したクリップのフレーム数が"
                "予定数と一致しません。"
                f"予定={clip_frame_count}, "
                f"実際={decoded_frame_count}"
            )

        shutil.move(
            str(staging_video),
            str(args.output_video),
        )
        shutil.move(
            str(staging_output_dir),
            str(args.output_dir),
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    print("外部検証用データを作成しました")
    print(f"入力動画: {args.input_video}")
    print(f"開始時刻: {args.start_second:.3f}秒")
    print(
        "実際の終了時刻: "
        f"{end_frame / fps:.3f}秒"
    )
    print(f"元動画開始フレーム: {start_frame}")
    print(
        "元動画終了フレーム（含まない）: "
        f"{end_frame}"
    )
    print(f"クリップフレーム数: {clip_frame_count}")
    print(f"抽出画像数: {len(sample_indices)}")
    print(f"連続クリップ: {args.output_video}")
    print(f"画像出力先: {args.output_dir}")
    print(
        "対応表: "
        f"{args.output_dir / 'manifest.csv'}"
    )


if __name__ == "__main__":
    main()