import csv
import json
from collections import Counter
from pathlib import Path

from summarize_fn_causes import (
    FACTOR_NAMES,
    extract_image_name,
    extract_selected_factors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "fn_cause_analysis_001"
    / "label_studio_export_104.json"
)
DETAIL_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_metrics"
    / "conf025_img1280_details.csv"
)
SANITIZED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "detection_factor_analysis_001"
)
SANITIZED_CSV_PATH = (
    SANITIZED_OUTPUT_DIR
    / "factors_by_outcome.csv"
)
SUMMARY_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detection_factor_analysis_001"
)
COMPARISON_CSV_PATH = (
    SUMMARY_OUTPUT_DIR
    / "factor_comparison.csv"
)

EXPECTED_TP_COUNT = 32
EXPECTED_FN_COUNT = 72
EXPECTED_TOTAL_COUNT = 104


def parse_boolean(value: str) -> bool:
    """CSVのTrue・False文字列を真偽値へ変換する。"""
    return value.strip().lower() == "true"


def load_detection_outcomes() -> dict[str, str]:
    """各正例画像がTPかFNかを評価結果から取得する。"""
    outcomes = {}

    with DETAIL_CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as detail_file:
        reader = csv.DictReader(detail_file)

        for row in reader:
            if not parse_boolean(
                row["ground_truth_present"]
            ):
                continue

            if parse_boolean(
                row["matched_at_iou_050"]
            ):
                outcome = "TP"
            elif int(
                row["false_negative_count"]
            ) == 1:
                outcome = "FN"
            else:
                continue

            outcomes[row["image_name"]] = outcome

    return outcomes


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """0除算を避けて割合を計算する。"""
    if denominator == 0:
        return None

    return numerator / denominator


def format_number(value: float | str | None) -> str:
    """数値は小数第3位まで表示し、計算不能な値はN/Aと表示する。"""
    if value in (None, ""):
        return "N/A"

    return f"{float(value):.3f}"


def main() -> None:
    if not ANNOTATION_JSON_PATH.exists():
        print(
            f"分類JSONが見つかりません: "
            f"{ANNOTATION_JSON_PATH}"
        )
        return

    if not DETAIL_CSV_PATH.exists():
        print(
            f"評価結果CSVが見つかりません: "
            f"{DETAIL_CSV_PATH}"
        )
        return

    tasks = json.loads(
        ANNOTATION_JSON_PATH.read_text(
            encoding="utf-8",
        )
    )
    outcomes = load_detection_outcomes()

    errors = []
    rows = []
    image_names = []

    outcome_counts = Counter()
    factor_counts = {
        "TP": Counter(),
        "FN": Counter(),
    }

    if len(tasks) != EXPECTED_TOTAL_COUNT:
        errors.append(
            "分類タスク数が想定と異なります: "
            f"{len(tasks)} / {EXPECTED_TOTAL_COUNT}"
        )

    for task in tasks:
        image_name = extract_image_name(task)

        if image_name is None:
            errors.append(
                f"画像名を取得できません: "
                f"task={task.get('id')}"
            )
            continue

        outcome = outcomes.get(image_name)

        if outcome is None:
            errors.append(
                f"TP・FNを取得できません: {image_name}"
            )
            continue

        try:
            selected_factors = (
                extract_selected_factors(task)
            )
        except ValueError as error:
            errors.append(
                f"{image_name}: {error}"
            )
            continue

        unknown_factors = (
            set(selected_factors)
            - set(FACTOR_NAMES)
        )

        if unknown_factors:
            errors.append(
                f"{image_name}: 未定義の要因: "
                f"{sorted(unknown_factors)}"
            )
            continue

        if not selected_factors:
            errors.append(
                f"{image_name}: 要因が未選択です"
            )
            continue

        if (
            "no_obvious_manual_factor"
            in selected_factors
            and len(selected_factors) > 1
        ):
            errors.append(
                f"{image_name}: "
                "no_obvious_manual_factorが"
                "他要因と同時選択されています"
            )
            continue

        image_names.append(image_name)
        outcome_counts[outcome] += 1
        factor_counts[outcome].update(
            selected_factors
        )

        row = {
            "image_name": image_name,
            "outcome": outcome,
            "selected_factors": ";".join(
                selected_factors
            ),
        }

        for factor_name in FACTOR_NAMES:
            row[factor_name] = int(
                factor_name in selected_factors
            )

        rows.append(row)

    if len(set(image_names)) != len(image_names):
        errors.append(
            "同じ画像が複数登録されています"
        )

    if outcome_counts["TP"] != EXPECTED_TP_COUNT:
        errors.append(
            "TP数が想定と異なります: "
            f"{outcome_counts['TP']} / "
            f"{EXPECTED_TP_COUNT}"
        )

    if outcome_counts["FN"] != EXPECTED_FN_COUNT:
        errors.append(
            "FN数が想定と異なります: "
            f"{outcome_counts['FN']} / "
            f"{EXPECTED_FN_COUNT}"
        )

    rows.sort(
        key=lambda row: row["image_name"]
    )

    SANITIZED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    SUMMARY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanitized_fieldnames = [
        "image_name",
        "outcome",
        *FACTOR_NAMES,
        "selected_factors",
    ]

    with SANITIZED_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=sanitized_fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []

    for factor_name in FACTOR_NAMES:
        tp_present = factor_counts["TP"][
            factor_name
        ]
        fn_present = factor_counts["FN"][
            factor_name
        ]

        tp_absent = (
            outcome_counts["TP"] - tp_present
        )
        fn_absent = (
            outcome_counts["FN"] - fn_present
        )

        tp_prevalence = safe_divide(
            tp_present,
            outcome_counts["TP"],
        )
        fn_prevalence = safe_divide(
            fn_present,
            outcome_counts["FN"],
        )

        fn_rate_when_present = safe_divide(
            fn_present,
            fn_present + tp_present,
        )
        fn_rate_when_absent = safe_divide(
            fn_absent,
            fn_absent + tp_absent,
        )

        relative_risk = None

        if (
            fn_rate_when_present is not None
            and fn_rate_when_absent not in (
                None,
                0,
            )
        ):
            relative_risk = (
                fn_rate_when_present
                / fn_rate_when_absent
            )

        comparison_rows.append(
            {
                "factor": factor_name,
                "tp_count": tp_present,
                "tp_percentage": round(
                    (tp_prevalence or 0) * 100,
                    2,
                ),
                "fn_count": fn_present,
                "fn_percentage": round(
                    (fn_prevalence or 0) * 100,
                    2,
                ),
                "percentage_point_difference": (
                    round(
                        (
                            (fn_prevalence or 0)
                            - (tp_prevalence or 0)
                        )
                        * 100,
                        2,
                    )
                ),
                "fn_relative_risk": (
                    round(relative_risk, 3)
                    if relative_risk is not None
                    else ""
                ),
            }
        )

    with COMPARISON_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as comparison_file:
        writer = csv.DictWriter(
            comparison_file,
            fieldnames=comparison_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("TP・FN要因比較")
    print("-" * 86)
    print(f"入力タスク数: {len(tasks)}")
    print(f"有効TP数: {outcome_counts['TP']}")
    print(f"有効FN数: {outcome_counts['FN']}")
    print(f"エラー数: {len(errors)}")

    print(
        "\n"
        f"{'要因':<30}"
        f"{'TP':>8}"
        f"{'TP率':>10}"
        f"{'FN':>8}"
        f"{'FN率':>10}"
        f"{'差':>10}"
        f"{'RR':>10}"
    )

    for comparison_row in comparison_rows:
        print(
            f"{comparison_row['factor']:<30}"
            f"{comparison_row['tp_count']:>8}"
            f"{comparison_row['tp_percentage']:>9.2f}%"
            f"{comparison_row['fn_count']:>8}"
            f"{comparison_row['fn_percentage']:>9.2f}%"
            f"{comparison_row['percentage_point_difference']:>9.2f}"
            f"{format_number(comparison_row['fn_relative_risk']):>10}"
        )

    print(f"\n匿名化比較データ: {SANITIZED_CSV_PATH}")
    print(f"比較結果CSV: {COMPARISON_CSV_PATH}")

    if errors:
        print("\n確認が必要な内容")

        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()