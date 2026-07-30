import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIZE_DATA_CSV = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "ball_size_analysis_001"
    / "ball_size_by_outcome.csv"
)
FACTOR_DATA_CSV = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "detection_factor_analysis_001"
    / "factors_by_outcome.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "combined_detection_factor_analysis_001"
)
OUTPUT_CSV = OUTPUT_DIR / "combined_factor_summary.csv"


def read_csv_by_image(
    csv_path: Path,
) -> dict[str, dict[str, str]]:
    """
    CSVを読み込み、画像名から各行を検索できる辞書へ変換する。
    """
    rows_by_image: dict[str, dict[str, str]] = {}

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            image_name = row.get("image_name")

            if not image_name:
                raise ValueError(
                    f"image_nameがありません: {csv_path}"
                )

            if image_name in rows_by_image:
                raise ValueError(
                    f"画像名が重複しています: {image_name}"
                )

            rows_by_image[image_name] = row

    return rows_by_image


def flag_is_true(value: str) -> bool:
    """CSVに保存された0・1をbool型へ変換する。"""
    return value.strip() == "1"


def wilson_interval(
    success_count: int,
    total_count: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """
    成功率の95% Wilson信頼区間を計算する。

    success_countはTP数、total_countはグループの画像数。
    """
    if total_count == 0:
        return 0.0, 0.0

    proportion = success_count / total_count
    denominator = 1 + (z_value**2 / total_count)

    center = (
        proportion
        + z_value**2 / (2 * total_count)
    ) / denominator

    margin = (
        z_value
        * math.sqrt(
            (
                proportion
                * (1 - proportion)
                / total_count
            )
            + (
                z_value**2
                / (4 * total_count**2)
            )
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def summarize_group(
    group_name: str,
    records: list[dict[str, str]],
) -> dict[str, str | int | float]:
    """指定したグループのTP・FN・Recallを集計する。"""
    total_count = len(records)
    tp_count = sum(
        record["outcome"] == "TP"
        for record in records
    )
    fn_count = sum(
        record["outcome"] == "FN"
        for record in records
    )

    recall = (
        tp_count / total_count
        if total_count > 0
        else 0.0
    )

    ci_lower, ci_upper = wilson_interval(
        tp_count,
        total_count,
    )

    return {
        "group_name": group_name,
        "total_count": total_count,
        "tp_count": tp_count,
        "fn_count": fn_count,
        "recall": round(recall, 4),
        "ci95_lower": round(ci_lower, 4),
        "ci95_upper": round(ci_upper, 4),
    }


def main() -> None:
    size_rows = read_csv_by_image(SIZE_DATA_CSV)
    factor_rows = read_csv_by_image(FACTOR_DATA_CSV)

    size_image_names = set(size_rows)
    factor_image_names = set(factor_rows)

    if size_image_names != factor_image_names:
        only_in_size = sorted(
            size_image_names - factor_image_names
        )
        only_in_factors = sorted(
            factor_image_names - size_image_names
        )

        raise ValueError(
            "2つのCSVで画像名が一致しません。\n"
            f"サイズCSVだけ: {only_in_size}\n"
            f"要因CSVだけ: {only_in_factors}"
        )

    combined_records: list[dict[str, str]] = []

    for image_name, size_row in size_rows.items():
        factor_row = factor_rows[image_name]

        if size_row["outcome"] != factor_row["outcome"]:
            raise ValueError(
                f"TP・FNが一致しません: {image_name}"
            )

        size_bin = size_row["size_bin"]

        size_group = (
            "small_half"
            if size_bin in {"Q1_smallest", "Q2"}
            else "large_half"
        )

        net_overlap = flag_is_true(
            factor_row["net_overlap"]
        )
        player_occlusion = flag_is_true(
            factor_row["player_or_hand_occlusion"]
        )

        structural_obstruction = (
            net_overlap or player_occlusion
        )

        combined_records.append(
            {
                "image_name": image_name,
                "outcome": size_row["outcome"],
                "size_group": size_group,
                "structural_obstruction": (
                    "present"
                    if structural_obstruction
                    else "absent"
                ),
            }
        )

    small_records = [
        record
        for record in combined_records
        if record["size_group"] == "small_half"
    ]
    large_records = [
        record
        for record in combined_records
        if record["size_group"] == "large_half"
    ]

    no_obstruction_records = [
        record
        for record in combined_records
        if record["structural_obstruction"] == "absent"
    ]
    obstruction_records = [
        record
        for record in combined_records
        if record["structural_obstruction"] == "present"
    ]

    groups = [
        (
            "all_small_half",
            small_records,
        ),
        (
            "all_large_half",
            large_records,
        ),
        (
            "no_obstruction_small_half",
            [
                record
                for record in small_records
                if record["structural_obstruction"]
                == "absent"
            ],
        ),
        (
            "no_obstruction_large_half",
            [
                record
                for record in large_records
                if record["structural_obstruction"]
                == "absent"
            ],
        ),
        (
            "obstruction_small_half",
            [
                record
                for record in obstruction_records
                if record["size_group"]
                == "small_half"
            ],
        ),
        (
            "obstruction_large_half",
            [
                record
                for record in obstruction_records
                if record["size_group"]
                == "large_half"
            ],
        ),
        (
            "all_no_obstruction",
            no_obstruction_records,
        ),
        (
            "all_obstruction",
            obstruction_records,
        ),
    ]

    summary_rows = [
        summarize_group(group_name, records)
        for group_name, records in groups
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "group_name",
        "total_count",
        "tp_count",
        "fn_count",
        "recall",
        "ci95_lower",
        "ci95_upper",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("ボールサイズ・遮蔽の複合分析")
    print("-" * 92)
    print(f"統合画像数: {len(combined_records)}")
    print()

    print(
        f"{'グループ':34}"
        f"{'件数':>8}"
        f"{'TP':>8}"
        f"{'FN':>8}"
        f"{'Recall':>12}"
        f"{'95%信頼区間':>20}"
    )
    print("-" * 92)

    for row in summary_rows:
        print(
            f"{str(row['group_name']):34}"
            f"{int(row['total_count']):8}"
            f"{int(row['tp_count']):8}"
            f"{int(row['fn_count']):8}"
            f"{float(row['recall']):12.3f}"
            f"  "
            f"{float(row['ci95_lower']):.3f}"
            f"〜"
            f"{float(row['ci95_upper']):.3f}"
        )

    print()
    print(f"集計結果: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()