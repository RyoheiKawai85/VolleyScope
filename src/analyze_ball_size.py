import csv
import re
import statistics
from pathlib import Path

import cv2


# このPythonファイルを基準に、VolleyScopeのルートフォルダを取得する
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_metrics"
    / "conf025_img1280_details.csv"
)
LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "evaluation_001_final"
    / "labels"
)
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "evaluation_001"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ball_size_analysis_001"
)
ANONYMIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "ball_size_analysis_001"
)

DETAIL_CSV = ANONYMIZED_DIR / "ball_size_by_outcome.csv"
SIZE_BIN_CSV = OUTPUT_DIR / "size_bin_summary.csv"

# 今回評価したYOLOの入力サイズ
INPUT_SIZE = 1280


def text_to_bool(value: str) -> bool:
    """CSVのTrue・Falseという文字列をbool型へ変換する。"""
    return value.strip().lower() == "true"


def extract_frame_stem(label_path: Path) -> str:
    """
    Label Studioの接頭辞を除き、frame_000000の部分を取り出す。

    例:
    732d22ff-frame_000000.txt
        → frame_000000
    """
    match = re.search(r"(frame_\d+)$", label_path.stem)

    if match is None:
        raise ValueError(
            f"画像名をラベル名から取得できません: {label_path.name}"
        )

    return match.group(1)


def create_label_map() -> dict[str, Path]:
    """元画像名とラベルファイルの対応表を作る。"""
    label_map: dict[str, Path] = {}

    for label_path in LABEL_DIR.glob("*.txt"):
        frame_stem = extract_frame_stem(label_path)

        if frame_stem in label_map:
            raise ValueError(
                f"同じ画像に複数のラベルがあります: {frame_stem}"
            )

        label_map[frame_stem] = label_path

    return label_map


def read_ball_box(
    label_path: Path,
) -> tuple[float, float, float, float]:
    """
    YOLO形式の正解枠から中心座標・幅・高さを読み取る。

    戻り値:
    center_x, center_y, normalized_width, normalized_height
    """
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise ValueError(
            f"正解枠が1個ではありません: "
            f"{label_path.name} ({len(lines)}個)"
        )

    values = lines[0].split()

    if len(values) != 5:
        raise ValueError(
            f"YOLO形式ではありません: {label_path.name}"
        )

    class_id = int(values[0])

    if class_id != 0:
        raise ValueError(
            f"ball以外のクラスです: {label_path.name}"
        )

    center_x = float(values[1])
    center_y = float(values[2])
    normalized_width = float(values[3])
    normalized_height = float(values[4])

    return (
        center_x,
        center_y,
        normalized_width,
        normalized_height,
    )


def classify_coco_size(area_px: float) -> str:
    """
    COCOの物体サイズ基準でsmall・medium・largeに分類する。

    small: 32×32未満
    medium: 32×32以上、96×96未満
    large: 96×96以上
    """
    if area_px < 32**2:
        return "small"

    if area_px < 96**2:
        return "medium"

    return "large"


def load_evaluation_outcomes() -> dict[str, str]:
    """
    評価CSVから、ボールが存在する画像のTP・FNを読み取る。
    """
    outcomes: dict[str, str] = {}

    with EVALUATION_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if not text_to_bool(
                row["ground_truth_present"]
            ):
                continue

            image_name = row["image_name"]
            matched = text_to_bool(
                row["matched_at_iou_050"]
            )

            outcomes[image_name] = (
                "TP" if matched else "FN"
            )

    return outcomes


def write_csv(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
    fieldnames: list[str],
) -> None:
    """集計結果をCSVとして保存する。"""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    label_map = create_label_map()
    outcomes = load_evaluation_outcomes()

    result_rows: list[
        dict[str, str | int | float]
    ] = []
    errors: list[str] = []

    for image_name, outcome in outcomes.items():
        try:
            frame_stem = Path(image_name).stem
            image_path = IMAGE_DIR / image_name

            if frame_stem not in label_map:
                raise FileNotFoundError(
                    f"ラベルがありません: {image_name}"
                )

            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(
                    f"画像を開けません: {image_name}"
                )

            image_height, image_width = image.shape[:2]

            (
                center_x,
                center_y,
                normalized_width,
                normalized_height,
            ) = read_ball_box(label_map[frame_stem])

            original_width_px = (
                normalized_width * image_width
            )
            original_height_px = (
                normalized_height * image_height
            )
            original_area_px = (
                original_width_px * original_height_px
            )

            normalized_area = (
                normalized_width * normalized_height
            )
            area_percentage = normalized_area * 100

            # 縦横比を維持して1280へ収める際の拡大率
            resize_scale = INPUT_SIZE / max(
                image_width,
                image_height,
            )

            input_width_px = (
                original_width_px * resize_scale
            )
            input_height_px = (
                original_height_px * resize_scale
            )
            input_area_px = (
                input_width_px * input_height_px
            )

            result_rows.append(
                {
                    "image_name": image_name,
                    "outcome": outcome,
                    "center_x": round(center_x, 6),
                    "center_y": round(center_y, 6),
                    "normalized_width": round(
                        normalized_width,
                        8,
                    ),
                    "normalized_height": round(
                        normalized_height,
                        8,
                    ),
                    "normalized_area": round(
                        normalized_area,
                        10,
                    ),
                    "area_percentage": round(
                        area_percentage,
                        6,
                    ),
                    "original_width_px": round(
                        original_width_px,
                        3,
                    ),
                    "original_height_px": round(
                        original_height_px,
                        3,
                    ),
                    "original_area_px": round(
                        original_area_px,
                        3,
                    ),
                    "input_width_px": round(
                        input_width_px,
                        3,
                    ),
                    "input_height_px": round(
                        input_height_px,
                        3,
                    ),
                    "input_area_px": round(
                        input_area_px,
                        3,
                    ),
                    "coco_size": classify_coco_size(
                        input_area_px
                    ),
                    "size_bin": "",
                }
            )

        except (
            FileNotFoundError,
            ValueError,
        ) as error:
            errors.append(str(error))

    # 小さい順に並べ、同数に近い4グループへ分ける
    result_rows.sort(
        key=lambda row: float(
            row["input_area_px"]
        )
    )

    size_bin_names = [
        "Q1_smallest",
        "Q2",
        "Q3",
        "Q4_largest",
    ]

    for index, row in enumerate(result_rows):
        bin_index = min(
            index * 4 // len(result_rows),
            3,
        )
        row["size_bin"] = size_bin_names[bin_index]

    detail_fieldnames = [
        "image_name",
        "outcome",
        "center_x",
        "center_y",
        "normalized_width",
        "normalized_height",
        "normalized_area",
        "area_percentage",
        "original_width_px",
        "original_height_px",
        "original_area_px",
        "input_width_px",
        "input_height_px",
        "input_area_px",
        "coco_size",
        "size_bin",
    ]

    write_csv(
        DETAIL_CSV,
        result_rows,
        detail_fieldnames,
    )

    size_summary_rows: list[
        dict[str, str | int | float]
    ] = []

    for size_bin in size_bin_names:
        bin_rows = [
            row
            for row in result_rows
            if row["size_bin"] == size_bin
        ]

        tp_count = sum(
            row["outcome"] == "TP"
            for row in bin_rows
        )
        fn_count = sum(
            row["outcome"] == "FN"
            for row in bin_rows
        )
        total_count = len(bin_rows)

        recall = (
            tp_count / total_count
            if total_count > 0
            else 0
        )

        areas = [
            float(row["input_area_px"])
            for row in bin_rows
        ]

        size_summary_rows.append(
            {
                "size_bin": size_bin,
                "total_count": total_count,
                "tp_count": tp_count,
                "fn_count": fn_count,
                "recall": round(recall, 4),
                "minimum_area_px": round(
                    min(areas),
                    3,
                ),
                "maximum_area_px": round(
                    max(areas),
                    3,
                ),
                "median_area_px": round(
                    statistics.median(areas),
                    3,
                ),
            }
        )

    write_csv(
        SIZE_BIN_CSV,
        size_summary_rows,
        [
            "size_bin",
            "total_count",
            "tp_count",
            "fn_count",
            "recall",
            "minimum_area_px",
            "maximum_area_px",
            "median_area_px",
        ],
    )

    print("ボールサイズ分析")
    print("-" * 72)
    print(f"評価対象数: {len(outcomes)}")
    print(f"分析成功数: {len(result_rows)}")
    print(f"エラー数: {len(errors)}")

    for outcome in ["TP", "FN"]:
        outcome_areas = [
            float(row["input_area_px"])
            for row in result_rows
            if row["outcome"] == outcome
        ]

        print()
        print(f"{outcome}: {len(outcome_areas)}件")
        print(
            "平均面積: "
            f"{statistics.mean(outcome_areas):.2f}px²"
        )
        print(
            "中央値: "
            f"{statistics.median(outcome_areas):.2f}px²"
        )

    print()
    print("大きさ別Recall")
    print("-" * 72)

    for row in size_summary_rows:
        print(
            f"{str(row['size_bin']):12} "
            f"件数={int(row['total_count']):3} "
            f"TP={int(row['tp_count']):3} "
            f"FN={int(row['fn_count']):3} "
            f"Recall={float(row['recall']):.3f} "
            f"面積={float(row['minimum_area_px']):.1f}"
            f"〜{float(row['maximum_area_px']):.1f}px²"
        )

    if errors:
        print()
        print("エラー内容")

        for error in errors:
            print(f"- {error}")

    print()
    print(f"画像別データ: {DETAIL_CSV}")
    print(f"大きさ別集計: {SIZE_BIN_CSV}")


if __name__ == "__main__":
    main()