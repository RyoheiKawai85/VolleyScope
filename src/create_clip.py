from pathlib import Path

import cv2


# プロジェクト内の入出力パス
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = PROJECT_ROOT / "data" / "raw" / "match01.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "data" / "clips" / "ball_challenge_002.mp4"

# 切り出す時間を秒単位で指定する
START_SECOND = 15
DURATION_SECONDS = 10


def main():
    """元動画から、YOLOの初期検証に使用する短い動画を切り出す。"""

    if not INPUT_VIDEO.exists():
        print(f"入力動画が見つかりません: {INPUT_VIDEO}")
        return

    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("入力動画をOpenCVで開けませんでした")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("動画のFPSを取得できませんでした")
        video.release()
        return

    total_seconds = total_frames / fps

    if START_SECOND >= total_seconds:
        print(
            f"開始位置が動画時間を超えています: "
            f"開始={START_SECOND}秒、動画={total_seconds:.2f}秒"
        )
        video.release()
        return

    # 動画の終端を超えないよう、終了フレームを調整する
    start_frame = int(START_SECOND * fps)
    requested_end_frame = int((START_SECOND + DURATION_SECONDS) * fps)
    end_frame = min(requested_end_frame, total_frames)

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # MP4形式の出力動画を作成する
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("出力動画を作成できませんでした")
        video.release()
        return

    # 指定した開始位置へ移動する
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    written_frames = 0

    while current_frame < end_frame:
        success, frame = video.read()

        if not success:
            print("動画の途中でフレームを読み込めなくなりました")
            break

        writer.write(frame)
        current_frame += 1
        written_frames += 1

    video.release()
    writer.release()

    output_seconds = written_frames / fps

    print("検証用クリップを作成しました")
    print(f"入力動画: {INPUT_VIDEO}")
    print(f"出力動画: {OUTPUT_VIDEO}")
    print(f"開始位置: {START_SECOND}秒")
    print(f"出力時間: {output_seconds:.2f}秒")
    print(f"出力フレーム数: {written_frames}")


if __name__ == "__main__":
    main()