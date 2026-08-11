from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO_PATH = (
    PROJECT_ROOT / "data" / "raw" / "match01.mp4"
)
CLIP_PATHS = [
    PROJECT_ROOT / "data" / "clips" / "baseline_001.mp4",
    (
        PROJECT_ROOT
        / "data"
        / "clips"
        / "ball_challenge_001.mp4"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "clips"
        / "ball_challenge_002.mp4"
    ),
]

# 比較処理を軽くするため、元の縦横比に近い小画像へ変換する。
SIGNATURE_WIDTH = 160
SIGNATURE_HEIGHT = 74
TOP_CANDIDATE_COUNT = 3


def create_frame_signature(
    frame: np.ndarray,
) -> np.ndarray:
    """フレームを比較用の小さなグレースケール画像へ変換する。"""
    grayscale = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )
    resized = cv2.resize(
        grayscale,
        (SIGNATURE_WIDTH, SIGNATURE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.float32)


def read_all_source_signatures(
    video_path: Path,
) -> tuple[list[np.ndarray], float]:
    """元動画の全フレームを比較用画像として読み込む。"""
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"元動画を開けません: {video_path}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)
    signatures = []

    while True:
        success, frame = video.read()

        if not success:
            break

        signatures.append(
            create_frame_signature(frame)
        )

    video.release()

    if not signatures:
        raise RuntimeError(
            f"元動画からフレームを読み込めません: "
            f"{video_path}"
        )

    if fps <= 0:
        raise RuntimeError(
            f"元動画のFPSを取得できません: {video_path}"
        )

    return signatures, fps


def read_clip_signatures(
    video_path: Path,
) -> tuple[dict[int, np.ndarray], int]:
    """クリップの4地点から比較用画像を読み込む。"""
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"クリップを開けません: {video_path}"
        )

    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        video.release()
        raise RuntimeError(
            f"フレーム数を取得できません: {video_path}"
        )

    target_indices = sorted(
        {
            0,
            frame_count // 3,
            frame_count * 2 // 3,
            frame_count - 1,
        }
    )
    target_index_set = set(target_indices)
    signatures = {}
    current_index = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if current_index in target_index_set:
            signatures[current_index] = (
                create_frame_signature(frame)
            )

        current_index += 1

    video.release()

    if len(signatures) != len(target_indices):
        raise RuntimeError(
            "比較用フレームをすべて読み込めません: "
            f"{video_path}"
        )

    return signatures, frame_count


def calculate_candidate_scores(
    source_signatures: list[np.ndarray],
    clip_signatures: dict[int, np.ndarray],
    clip_frame_count: int,
) -> np.ndarray:
    """元動画内の各開始候補について平均絶対誤差を求める。"""
    candidate_count = (
        len(source_signatures)
        - clip_frame_count
        + 1
    )

    if candidate_count <= 0:
        raise ValueError(
            "クリップが元動画より長いため比較できません"
        )

    scores = np.empty(
        candidate_count,
        dtype=np.float64,
    )

    for candidate_start in range(candidate_count):
        frame_scores = []

        for clip_index, clip_signature in (
            clip_signatures.items()
        ):
            source_signature = source_signatures[
                candidate_start + clip_index
            ]
            mean_absolute_error = np.mean(
                np.abs(
                    source_signature
                    - clip_signature
                )
            )
            frame_scores.append(
                mean_absolute_error
            )

        scores[candidate_start] = np.mean(
            frame_scores
        )

    return scores


def main() -> None:
    """各クリップの元動画内開始位置候補を表示する。"""
    if not SOURCE_VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"元動画が見つかりません: "
            f"{SOURCE_VIDEO_PATH}"
        )

    for clip_path in CLIP_PATHS:
        if not clip_path.is_file():
            raise FileNotFoundError(
                f"クリップが見つかりません: "
                f"{clip_path}"
            )

    print("元動画の比較用データを読み込みます")
    source_signatures, source_fps = (
        read_all_source_signatures(
            SOURCE_VIDEO_PATH
        )
    )

    print(
        f"元動画フレーム数: "
        f"{len(source_signatures)}"
    )
    print(f"元動画FPS: {source_fps:.3f}")

    for clip_path in CLIP_PATHS:
        clip_signatures, clip_frame_count = (
            read_clip_signatures(clip_path)
        )
        scores = calculate_candidate_scores(
            source_signatures,
            clip_signatures,
            clip_frame_count,
        )
        candidate_indices = np.argsort(scores)[
            :TOP_CANDIDATE_COUNT
        ]

        print()
        print(f"クリップ: {clip_path.name}")
        print(
            f"クリップフレーム数: "
            f"{clip_frame_count}"
        )
        print("開始位置候補:")

        for rank, candidate_index in enumerate(
            candidate_indices,
            start=1,
        ):
            start_seconds = (
                int(candidate_index) / source_fps
            )
            end_seconds = (
                (
                    int(candidate_index)
                    + clip_frame_count
                )
                / source_fps
            )

            print(
                f"  {rank}位: "
                f"元フレーム={candidate_index}, "
                f"開始={start_seconds:.3f}秒, "
                f"終了={end_seconds:.3f}秒, "
                f"平均絶対誤差="
                f"{scores[candidate_index]:.4f}"
            )


if __name__ == "__main__":
    main()