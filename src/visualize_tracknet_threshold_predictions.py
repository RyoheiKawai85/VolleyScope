import argparse
import csv
from collections import deque
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_pilot_v2"
    / "val"
    / "images"
)

DEFAULT_PREDICTION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_heatmap_analysis"
    / "epoch_003_val_thresholds"
    / "per_frame_thresholds.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_heatmap_analysis"
    / "epoch_003_val_thresholds"
    / "threshold_040_raw_peak_review"
)

HEATMAP_WIDTH = 512
HEATMAP_HEIGHT = 288
EXPECTED_FRAME_COUNT = 120

GROUND_TRUTH_COLOR = (0, 255, 0)
PREDICTION_COLOR = (0, 0, 255)
RAW_PEAK_COLOR = (0, 165, 255)
CONNECTION_COLOR = (255, 255, 0)
GROUND_TRUTH_TRAIL_COLOR = (0, 180, 0)
PREDICTION_TRAIL_COLOR = (0, 0, 180)
TEXT_COLOR = (255, 255, 255)

CLASSIFICATION_COLORS = {
    "TP": (0, 160, 0),
    "TN": (80, 80, 80),
    "FP1": (0, 0, 220),
    "FP2": (0, 100, 255),
    "FN": (180, 0, 180),
}


def parse_args() -> argparse.Namespace:
    """可視化条件をコマンドラインから取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3のしきい値別予測を"
            "元画像へ重ね、確認画像と動画を作成する"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="元のval画像フォルダ",
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=DEFAULT_PREDICTION_CSV,
        help="しきい値別フレーム予測CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="確認画像と動画の新規出力先",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="可視化するヒートマップしきい値",
    )
    parser.add_argument(
        "--review-fps",
        type=float,
        default=12.0,
        help="確認動画の再生FPS",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=8,
        help="表示する直近の軌跡長",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """入力と出力条件を検証する。"""
    args.image_dir = args.image_dir.resolve()
    args.prediction_csv = args.prediction_csv.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(
            f"画像フォルダが見つかりません: "
            f"{args.image_dir}"
        )

    if not args.prediction_csv.is_file():
        raise FileNotFoundError(
            f"予測CSVが見つかりません: "
            f"{args.prediction_csv}"
        )

    if args.output_dir.exists():
        raise FileExistsError(
            "上書きを防ぐため、既存の出力先には"
            "保存しません: "
            f"{args.output_dir}"
        )

    if not 0 < args.threshold < 1:
        raise ValueError(
            "--thresholdには0より大きく"
            "1未満の値を指定してください"
        )

    if args.review_fps <= 0:
        raise ValueError(
            "--review-fpsには0より大きい値を"
            "指定してください"
        )

    if args.trail_length <= 0:
        raise ValueError(
            "--trail-lengthには1以上を指定してください"
        )


def read_prediction_rows(
    prediction_csv: Path,
    threshold: float,
) -> list[dict[str, int | float | str]]:
    """指定したしきい値の予測だけをCSVから読む。"""
    rows = []

    with prediction_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for source_row in reader:
            row_threshold = float(
                source_row["threshold"]
            )

            if abs(row_threshold - threshold) > 1e-9:
                continue

            rows.append(
                {
                    "local_frame": int(
                        source_row["local_frame"]
                    ),
                    "source_frame": int(
                        source_row["source_frame"]
                    ),
                    "ground_truth_visible": int(
                        source_row[
                            "ground_truth_visible"
                        ]
                    ),
                    "ground_truth_x": int(
                        source_row["ground_truth_x"]
                    ),
                    "ground_truth_y": int(
                        source_row["ground_truth_y"]
                    ),
                    "predicted_visible": int(
                        source_row["predicted_visible"]
                    ),
                    "predicted_x": int(
                        source_row["predicted_x"]
                    ),
                    "predicted_y": int(
                        source_row["predicted_y"]
                    ),
                    "distance": (
                        float(source_row["distance"])
                        if source_row["distance"]
                        else None
                    ),
                    "classification": source_row[
                        "classification"
                    ],
                    "raw_max": float(
                        source_row["raw_max"]
                    ),
                    "raw_peak_x": int(
                        source_row["raw_peak_x"]
                    ),
                    "raw_peak_y": int(
                        source_row["raw_peak_y"]
                    ),
                    "raw_peak_distance": (
                        float(
                            source_row[
                                "raw_peak_distance"
                            ]
                        )
                        if source_row[
                            "raw_peak_distance"
                        ]
                        else None
                    ),
                }
            )

    rows.sort(
        key=lambda row: int(row["local_frame"])
    )

    return rows


def validate_rows(
    rows: list[dict[str, int | float | str]],
) -> None:
    """対象件数、番号、分類内訳を検証する。"""
    if len(rows) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            "対象フレーム数が想定と一致しません: "
            f"{len(rows)}"
        )

    local_frames = [
        int(row["local_frame"])
        for row in rows
    ]

    expected_local_frames = list(
        range(EXPECTED_FRAME_COUNT)
    )

    if local_frames != expected_local_frames:
        raise ValueError(
            "ローカルフレーム番号に"
            "欠番または重複があります"
        )

    source_frames = [
        int(row["source_frame"])
        for row in rows
    ]

    if len(set(source_frames)) != len(source_frames):
        raise ValueError(
            "元フレーム番号が重複しています"
        )

    valid_classifications = {
        "TP",
        "TN",
        "FP1",
        "FP2",
        "FN",
    }

    for row in rows:
        classification = str(
            row["classification"]
        )

        if classification not in valid_classifications:
            raise ValueError(
                "未知の分類があります: "
                f"{classification}"
            )


def scale_point(
    x: int,
    y: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """ヒートマップ座標を元画像座標へ戻す。"""
    scaled_x = round(
        x * image_width / HEATMAP_WIDTH
    )
    scaled_y = round(
        y * image_height / HEATMAP_HEIGHT
    )

    scaled_x = max(
        0,
        min(scaled_x, image_width - 1),
    )
    scaled_y = max(
        0,
        min(scaled_y, image_height - 1),
    )

    return scaled_x, scaled_y


def draw_marker_with_label(
    image,
    point: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    """現在位置を円、十字、文字で描く。"""
    x, y = point

    cv2.circle(
        image,
        point,
        28,
        color,
        4,
    )
    cv2.drawMarker(
        image,
        point,
        color,
        cv2.MARKER_CROSS,
        50,
        4,
    )
    cv2.putText(
        image,
        label,
        (x + 35, max(35, y - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_trail(
    image,
    trail: deque,
    color: tuple[int, int, int],
) -> None:
    """欠損で途切れる直近の軌跡を描く。"""
    trail_points = list(trail)

    for index in range(
        1,
        len(trail_points),
    ):
        previous_point = trail_points[index - 1]
        current_point = trail_points[index]

        if (
            previous_point is None
            or current_point is None
        ):
            continue

        cv2.line(
            image,
            previous_point,
            current_point,
            color,
            3,
            cv2.LINE_AA,
        )


def draw_information_panel(
    image,
    row: dict[str, int | float | str],
    threshold: float,
) -> None:
    """フレーム番号、分類、信頼度を表示する。"""
    classification = str(
        row["classification"]
    )
    panel_color = CLASSIFICATION_COLORS[
        classification
    ]

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 115),
        panel_color,
        -1,
    )

    distance = row["distance"]
    distance_text = (
        "N/A"
        if distance is None
        else f"{float(distance):.2f}"
    )

    first_line = (
        f"Source frame: {row['source_frame']}  "
        f"Local frame: {row['local_frame']}  "
        f"Class: {classification}"
    )
    second_line = (
        f"Threshold: {threshold:.2f}  "
        f"Raw max: {float(row['raw_max']):.4f}  "
        f"Distance: {distance_text}"
    )

    cv2.putText(
        image,
        first_line,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        second_line,
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    """全120枚の確認画像とMP4を作成する。"""
    args = parse_args()
    validate_args(args)

    rows = read_prediction_rows(
        args.prediction_csv,
        args.threshold,
    )
    validate_rows(rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    image_output_dir = (
        args.output_dir / "images"
    )
    image_output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    first_source_frame = int(
        rows[0]["source_frame"]
    )
    first_image_path = (
        args.image_dir
        / f"frame_{first_source_frame:06d}.png"
    )
    first_image = cv2.imread(
        str(first_image_path)
    )

    if first_image is None:
        raise FileNotFoundError(
            f"画像を読み込めません: "
            f"{first_image_path}"
        )

    image_height, image_width = (
        first_image.shape[:2]
    )

    video_path = (
        args.output_dir
        / "threshold_040_review.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )
    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        args.review_fps,
        (image_width, image_height),
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            f"確認動画を作成できません: "
            f"{video_path}"
        )

    ground_truth_trail = deque(
        maxlen=args.trail_length
    )
    prediction_trail = deque(
        maxlen=args.trail_length
    )

    classification_counts = {
        classification: 0
        for classification
        in CLASSIFICATION_COLORS
    }

    try:
        for row in rows:
            source_frame = int(
                row["source_frame"]
            )
            image_path = (
                args.image_dir
                / f"frame_{source_frame:06d}.png"
            )
            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                raise FileNotFoundError(
                    f"画像を読み込めません: "
                    f"{image_path}"
                )

            if image.shape[:2] != (
                image_height,
                image_width,
            ):
                raise ValueError(
                    "画像サイズが統一されていません: "
                    f"{image_path}"
                )

            ground_truth_point = None
            prediction_point = None

            if int(
                row["ground_truth_visible"]
            ) == 1:
                ground_truth_point = scale_point(
                    int(row["ground_truth_x"]),
                    int(row["ground_truth_y"]),
                    image_width,
                    image_height,
                )

            if int(row["predicted_visible"]) == 1:
                prediction_point = scale_point(
                    int(row["predicted_x"]),
                    int(row["predicted_y"]),
                    image_width,
                    image_height,
                )

            raw_peak_point = scale_point(
                int(row["raw_peak_x"]),
                int(row["raw_peak_y"]),
                image_width,
                image_height,
            )
            ground_truth_trail.append(
                ground_truth_point
            )
            prediction_trail.append(
                prediction_point
            )

            draw_trail(
                image,
                ground_truth_trail,
                GROUND_TRUTH_TRAIL_COLOR,
            )
            draw_trail(
                image,
                prediction_trail,
                PREDICTION_TRAIL_COLOR,
            )

            if ground_truth_point is not None:
                draw_marker_with_label(
                    image,
                    ground_truth_point,
                    GROUND_TRUTH_COLOR,
                    "GROUND TRUTH",
                )

            if prediction_point is not None:
                draw_marker_with_label(
                    image,
                    prediction_point,
                    PREDICTION_COLOR,
                    "PREDICTION",
                )
            elif (
                str(row["classification"]) == "FN"
            ):
                draw_marker_with_label(
                    image,
                    raw_peak_point,
                    RAW_PEAK_COLOR,
                    "RAW PEAK BELOW THRESHOLD",
                )
            if (
                ground_truth_point is not None
                and prediction_point is not None
            ):
                cv2.line(
                    image,
                    ground_truth_point,
                    prediction_point,
                    CONNECTION_COLOR,
                    2,
                    cv2.LINE_AA,
                )

            draw_information_panel(
                image,
                row,
                args.threshold,
            )

            classification = str(
                row["classification"]
            )
            classification_counts[
                classification
            ] += 1

            output_image_path = (
                image_output_dir
                / (
                    f"frame_{source_frame:06d}_"
                    f"{classification}.png"
                )
            )

            if not cv2.imwrite(
                str(output_image_path),
                image,
            ):
                raise RuntimeError(
                    "確認画像を保存できません: "
                    f"{output_image_path}"
                )

            video_writer.write(image)

    finally:
        video_writer.release()

    if not video_path.is_file():
        raise RuntimeError(
            f"確認動画が作成されませんでした: "
            f"{video_path}"
        )

    print(
        "TrackNetV3しきい値予測の"
        "確認画像と動画を作成しました"
    )
    print(f"しきい値: {args.threshold:.2f}")
    print(f"保存枚数: {len(rows)}")

    for classification in (
        "TP",
        "TN",
        "FP1",
        "FP2",
        "FN",
    ):
        print(
            f"{classification}: "
            f"{classification_counts[classification]}"
        )

    print(f"画像出力先: {image_output_dir}")
    print(f"動画出力先: {video_path}")


if __name__ == "__main__":
    main()