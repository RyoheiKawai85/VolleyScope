import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "fn_cause_analysis_001"
    / "label_studio_export.json"
)
SANITIZED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "fn_cause_analysis_001"
)
SANITIZED_CSV_PATH = (
    SANITIZED_OUTPUT_DIR
    / "fn_factors.csv"
)
SUMMARY_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fn_cause_analysis_001"
)
FACTOR_SUMMARY_PATH = (
    SUMMARY_OUTPUT_DIR
    / "factor_summary.csv"
)
CO_OCCURRENCE_PATH = (
    SUMMARY_OUTPUT_DIR
    / "factor_co_occurrence.csv"
)

EXPECTED_TASK_COUNT = 72

FACTOR_NAMES = [
    "net_overlap",
    "player_or_hand_occlusion",
    "motion_blur",
    "frame_edge",
    "low_contrast",
    "other",
    "no_obvious_manual_factor",
]


def extract_image_name(task: dict) -> str | None:
    """Label Studioの接頭辞を除いて元画像名を取得する。"""
    file_upload = task.get("file_upload", "")
    match = re.search(r"(frame_\d{6}\.jpg)$", file_upload)

    if match is None:
        return None

    return match.group(1)


def extract_selected_factors(task: dict) -> list[str]:
    """有効なアノテーションから選択要因を取得する。"""
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]

    if len(annotations) != 1:
        raise ValueError(
            "有効なアノテーション数が1ではありません"
        )

    selected_factors = []

    for result in annotations[0].get("result", []):
        if result.get("type") != "choices":
            continue

        if result.get("from_name") != "failure_factors":
            continue

        selected_factors.extend(
            result.get("value", {}).get("choices", [])
        )

    return sorted(set(selected_factors))


def save_sanitized_rows(rows: list[dict]) -> None:
    """画像名と要因だけを含むCSVを保存する。"""
    SANITIZED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image_name",
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
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_factor_summary(
    factor_counts: Counter,
    total_count: int,
) -> None:
    """要因ごとの件数と割合をCSVへ保存する。"""
    with FACTOR_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "factor",
                "count",
                "percentage",
            ],
        )
        writer.writeheader()

        for factor_name in FACTOR_NAMES:
            count = factor_counts[factor_name]

            writer.writerow(
                {
                    "factor": factor_name,
                    "count": count,
                    "percentage": round(
                        count / total_count * 100,
                        2,
                    ),
                }
            )


def save_co_occurrences(
    co_occurrence_counts: Counter,
) -> None:
    """同時に選択された要因ペアをCSVへ保存する。"""
    with CO_OCCURRENCE_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "first_factor",
                "second_factor",
                "count",
            ],
        )
        writer.writeheader()

        for (
            first_factor,
            second_factor,
        ), count in co_occurrence_counts.most_common():
            writer.writerow(
                {
                    "first_factor": first_factor,
                    "second_factor": second_factor,
                    "count": count,
                }
            )


def main() -> None:
    if not INPUT_JSON_PATH.exists():
        print(f"JSONが見つかりません: {INPUT_JSON_PATH}")
        return

    tasks = json.loads(
        INPUT_JSON_PATH.read_text(
            encoding="utf-8",
        )
    )

    errors = []
    sanitized_rows = []
    image_names = []

    factor_counts = Counter()
    combination_counts = Counter()
    co_occurrence_counts = Counter()

    if len(tasks) != EXPECTED_TASK_COUNT:
        errors.append(
            "タスク数が想定と異なります: "
            f"{len(tasks)} / {EXPECTED_TASK_COUNT}"
        )

    for task in tasks:
        image_name = extract_image_name(task)

        if image_name is None:
            errors.append(
                f"画像名を取得できません: task={task.get('id')}"
            )
            continue

        try:
            selected_factors = extract_selected_factors(task)
        except ValueError as error:
            errors.append(f"{image_name}: {error}")
            continue

        unknown_factors = (
            set(selected_factors)
            - set(FACTOR_NAMES)
        )

        if unknown_factors:
            errors.append(
                f"{image_name}: 未定義の要因があります: "
                f"{sorted(unknown_factors)}"
            )
            continue

        if not selected_factors:
            errors.append(
                f"{image_name}: 要因が選択されていません"
            )
            continue

        if (
            "no_obvious_manual_factor" in selected_factors
            and len(selected_factors) > 1
        ):
            errors.append(
                f"{image_name}: no_obvious_manual_factorが"
                "他要因と同時選択されています"
            )
            continue

        image_names.append(image_name)
        factor_counts.update(selected_factors)

        combination_key = " + ".join(selected_factors)
        combination_counts[combination_key] += 1

        for factor_pair in combinations(
            selected_factors,
            2,
        ):
            co_occurrence_counts[
                tuple(sorted(factor_pair))
            ] += 1

        row = {
            "image_name": image_name,
            "selected_factors": ";".join(
                selected_factors
            ),
        }

        for factor_name in FACTOR_NAMES:
            row[factor_name] = int(
                factor_name in selected_factors
            )

        sanitized_rows.append(row)

    if len(set(image_names)) != len(image_names):
        errors.append("同じ画像名が複数回登録されています")

    sanitized_rows.sort(
        key=lambda row: row["image_name"]
    )

    SUMMARY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_sanitized_rows(sanitized_rows)
    save_factor_summary(
        factor_counts,
        len(sanitized_rows),
    )
    save_co_occurrences(co_occurrence_counts)

    print("見逃し要因の集計結果")
    print("-" * 60)
    print(f"入力タスク数: {len(tasks)}")
    print(f"有効データ数: {len(sanitized_rows)}")
    print(f"エラー数: {len(errors)}")

    print("\n要因別件数")

    for factor_name in FACTOR_NAMES:
        count = factor_counts[factor_name]
        percentage = (
            count / len(sanitized_rows) * 100
            if sanitized_rows
            else 0
        )

        print(
            f"{factor_name:<30}"
            f"{count:>4}件 "
            f"({percentage:>6.2f}%)"
        )

    print("\n上位の組み合わせ")

    for combination_name, count in (
        combination_counts.most_common(10)
    ):
        print(f"{count:>4}件: {combination_name}")

    print(f"\n匿名化CSV: {SANITIZED_CSV_PATH}")
    print(f"要因集計CSV: {FACTOR_SUMMARY_PATH}")
    print(f"同時発生CSV: {CO_OCCURRENCE_PATH}")

    if errors:
        print("\n確認が必要な内容")

        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()