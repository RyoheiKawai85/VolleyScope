import csv
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "clips" / "ball_challenge_002.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

TARGET_FRAME_COUNT = 150


def calculate_target_indices(
    total_frames: int,
    target_count: int,
) -> list[int]:
    """動画全体から、指定した枚数のフレーム番号を均等に選ぶ。"""
    if total_frames <= 0:
        raise ValueError("動画の総フレーム数が正しく取得できませんでした")

    actual_count = min(total_frames, target_count)

    if actual_count == 1:
        return [0]

    return [
        round(index * (total_frames - 1) / (actual_count - 1))
        for index in range(actual_count)
    ]


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"動画が見つかりません: {VIDEO_PATH}")
        return

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print(f"動画を開けませんでした: {VIDEO_PATH}")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        print("動画情報を正しく取得できませんでした")
        video.release()
        return

    target_indices = calculate_target_indices(
        total_frames,
        TARGET_FRAME_COUNT,
    )
    target_index_set = set(target_indices)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    current_frame_index = 0
    saved_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if current_frame_index in target_index_set:
            file_name = f"frame_{current_frame_index:06d}.jpg"
            output_path = OUTPUT_DIR / file_name

            save_success = cv2.imwrite(str(output_path), frame)

            if not save_success:
                print(f"画像の保存に失敗しました: {output_path}")
                video.release()
                return

            manifest_rows.append(
                {
                    "file_name": file_name,
                    "frame_index": current_frame_index,
                    "time_seconds": round(current_frame_index / fps, 3),
                }
            )
            saved_count += 1

        current_frame_index += 1

    video.release()

    # 各画像が元動画の何フレーム目・何秒地点かをCSVへ記録する。
    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "file_name",
                "frame_index",
                "time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("評価用フレームの抽出が完了しました")
    print(f"入力動画: {VIDEO_PATH}")
    print(f"動画の総フレーム数: {total_frames}")
    print(f"動画のFPS: {fps:.3f}")
    print(f"抽出予定枚数: {len(target_indices)}")
    print(f"実際に保存した枚数: {saved_count}")
    print(f"画像の保存先: {OUTPUT_DIR}")
    print(f"対応表: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()