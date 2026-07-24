from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


# プロジェクト内の入出力パス
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = PROJECT_ROOT / "data" / "clips" / "baseline_001.mp4"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# COCOデータセット上のクラス番号
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


def main():
    """事前学習済みYOLOを動画へ適用し、初期性能を確認する。"""

    if not INPUT_VIDEO.exists():
        print(f"入力動画が見つかりません: {INPUT_VIDEO}")
        return

    # COCOデータセットで事前学習された軽量モデルを読み込む。
    # 初回実行時は、モデルファイルが自動的にダウンロードされる。
    model = YOLO("yolo11n.pt")

    start_time = perf_counter()

    # stream=Trueにより、全フレームの結果を一度にメモリへ保持せず、
    # 1フレームずつ処理結果を受け取る。
    results = model.predict(
        source=str(INPUT_VIDEO),
        classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
        conf=0.25,
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_ROOT),
        name="baseline_yolo11n",
        exist_ok=True,
        stream=True,
    )

    processed_frames = 0
    person_detections = 0
    ball_detections = 0

    for result in results:
        processed_frames += 1

        if result.boxes is None:
            continue

        detected_classes = result.boxes.cls.cpu().tolist()

        person_detections += detected_classes.count(PERSON_CLASS_ID)
        ball_detections += detected_classes.count(SPORTS_BALL_CLASS_ID)

    elapsed_seconds = perf_counter() - start_time
    processing_fps = (
        processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0
    )

    print()
    print("ベースライン検出が完了しました")
    print(f"入力動画: {INPUT_VIDEO}")
    print(f"処理フレーム数: {processed_frames}")
    print(f"人物の延べ検出数: {person_detections}")
    print(f"ボールの延べ検出数: {ball_detections}")
    print(f"処理時間: {elapsed_seconds:.2f}秒")
    print(f"処理速度: {processing_fps:.2f} FPS")
    print(f"出力先: {OUTPUT_ROOT / 'baseline_yolo11n'}")


if __name__ == "__main__":
    main()