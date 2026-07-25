from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "frames" / "evaluation_001"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_error_review"

MODEL_NAME = "yolo11n.pt"
SPORTS_BALL_CLASS_ID = 32
CONFIDENCE = 0.25

ERROR_CASES = {
    960: [
        "frame_000112.jpg",
    ],
    1280: [
        "frame_000008.jpg",
        "frame_000025.jpg",
        "frame_000087.jpg",
    ],
}


def main() -> None:
    model = YOLO(MODEL_NAME)

    for image_size, image_names in ERROR_CASES.items():
        image_paths = [
            str(IMAGE_DIR / image_name)
            for image_name in image_names
        ]

        print(f"誤検出画像を再処理します: imgsz={image_size}")

        model.predict(
            source=image_paths,
            conf=CONFIDENCE,
            imgsz=image_size,
            classes=[SPORTS_BALL_CLASS_ID],
            save=True,
            project=str(OUTPUT_DIR),
            name=f"imgsz_{image_size}",
            exist_ok=True,
            verbose=False,
        )

    print("誤検出確認画像を作成しました")
    print(f"出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()