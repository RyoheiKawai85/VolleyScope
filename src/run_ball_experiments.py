from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = (
    PROJECT_ROOT / "data" / "clips" / "ball_challenge_002.mp4"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "clean_size_experiments"

SPORTS_BALL_CLASS_ID = 32

# 一度に変更する条件を一つに限定し、影響を比較する
EXPERIMENTS = [
    {
        "name": "conf025_img640",
        "confidence": 0.25,
        "image_size": 640,
    },
    {
        "name": "conf025_img960",
        "confidence": 0.25,
        "image_size": 960,
    },
    {
        "name": "conf025_img1280",
        "confidence": 0.25,
        "image_size": 1280,
    },
]


def run_experiment(model, experiment):
    """指定された条件でボール検出を行い、基本集計を返す。"""

    start_time = perf_counter()

    results = model.predict(
        source=str(INPUT_VIDEO),
        classes=[SPORTS_BALL_CLASS_ID],
        conf=experiment["confidence"],
        imgsz=experiment["image_size"],
        save=True,
        project=str(OUTPUT_ROOT),
        name=experiment["name"],
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    processed_frames = 0
    frames_with_ball = 0
    total_ball_detections = 0
    multiple_ball_frames = 0
    confidence_values = []

    for result in results:
        processed_frames += 1

        detection_count = (
            len(result.boxes) if result.boxes is not None else 0
        )

        total_ball_detections += detection_count

        if detection_count > 0:
            frames_with_ball += 1

        if detection_count > 1:
            multiple_ball_frames += 1

        if result.boxes is not None:
            confidence_values.extend(
                result.boxes.conf.cpu().tolist()
            )

    elapsed_seconds = perf_counter() - start_time
    detection_frame_rate = (
        frames_with_ball / processed_frames
        if processed_frames > 0
        else 0
    )
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0
    )

    return {
        "name": experiment["name"],
        "processed_frames": processed_frames,
        "frames_with_ball": frames_with_ball,
        "detection_frame_rate": detection_frame_rate,
        "total_ball_detections": total_ball_detections,
        "multiple_ball_frames": multiple_ball_frames,
        "average_confidence": average_confidence,
        "elapsed_seconds": elapsed_seconds,
    }


def main():
    """信頼度と入力サイズによるボール検出の違いを比較する。"""

    if not INPUT_VIDEO.exists():
        print(f"入力動画が見つかりません: {INPUT_VIDEO}")
        return

    model = YOLO("yolo11n.pt")
    summaries = []

    for experiment in EXPERIMENTS:
        print()
        print(f"実験開始: {experiment['name']}")

        summary = run_experiment(model, experiment)
        summaries.append(summary)

        print(f"実験完了: {experiment['name']}")
        print(
            f"ボール検出フレーム: "
            f"{summary['frames_with_ball']} / "
            f"{summary['processed_frames']}"
        )
        print(
            f"検出フレーム率: "
            f"{summary['detection_frame_rate']:.2%}"
        )
        print(
            f"複数ボール検出フレーム: "
            f"{summary['multiple_ball_frames']}"
        )
        print(
            f"平均信頼度: "
            f"{summary['average_confidence']:.3f}"
        )
        print(
            f"処理時間: "
            f"{summary['elapsed_seconds']:.2f}秒"
        )

    print()
    print("比較実験がすべて完了しました")
    print("-" * 72)
    print(
        f"{'実験名':<24}"
        f"{'検出フレーム':>14}"
        f"{'検出率':>12}"
        f"{'複数検出':>10}"
        f"{'時間':>10}"
    )

    for summary in summaries:
        print(
            f"{summary['name']:<24}"
            f"{summary['frames_with_ball']:>14}"
            f"{summary['detection_frame_rate']:>11.2%}"
            f"{summary['multiple_ball_frames']:>10}"
            f"{summary['elapsed_seconds']:>9.1f}秒"
        )


if __name__ == "__main__":
    main()