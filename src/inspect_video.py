from pathlib import Path

import cv2


# 実行場所に左右されないよう、ファイル位置を基準にプロジェクト直下を取得する
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "match01.mp4"


def main():
    """試合動画を読み込み、解析前に必要な基本情報を表示する。"""

    if not VIDEO_PATH.exists():
        print(f"動画が見つかりません: {VIDEO_PATH}")
        return

    # OpenCVを使用して動画ファイルへアクセスする
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print("動画ファイルは存在しますが、OpenCVで開けませんでした")
        return

    # 後続の切り出し・推論時間の見積もりに必要な情報を取得する
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # FPSが取得できなかった場合のゼロ除算を防ぐ
    duration_seconds = frame_count / fps if fps > 0 else 0

    print("動画を正常に読み込めました")
    print(f"ファイル: {VIDEO_PATH}")
    print(f"解像度: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"総フレーム数: {frame_count}")
    print(f"動画時間: {duration_seconds:.2f}秒")
    print(f"動画時間: {duration_seconds / 60:.2f}分")

    # 使用した動画ファイルを確実に解放する
    video.release()


if __name__ == "__main__":
    main()