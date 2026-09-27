import argparse
import csv
import hashlib
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "yolo_diverse_v1"
)

EXPECTED_SOURCE_COUNTS = {
    ("train", "pilot_v2"): {
        "total": 128,
        "positive": 126,
        "negative": 2,
    },
    ("train", "match04"): {
        "total": 128,
        "positive": 120,
        "negative": 8,
    },
    ("val", "pilot_v2"): {
        "total": 120,
        "positive": 118,
        "negative": 2,
    },
    ("val", "match06"): {
        "total": 128,
        "positive": 122,
        "negative": 6,
    },
}

EXPECTED_SPLIT_COUNTS = {
    "train": {
        "total": 256,
        "positive": 246,
        "negative": 10,
    },
    "val": {
        "total": 248,
        "positive": 240,
        "negative": 8,
    },
}

SOURCE_CONFIGS = (
    {
        "split": "train",
        "source_group": "pilot_v2",
        "image_root": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "train"
            / "images"
        ),
        "label_root": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_train_final"
            / "extracted"
            / "labels"
        ),
    },
    {
        "split": "train",
        "source_group": "match04",
        "image_root": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "train"
        ),
        "label_root": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "train"
        ),
    },
    {
        "split": "val",
        "source_group": "pilot_v2",
        "image_root": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "val"
            / "images"
        ),
        "label_root": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_val_final"
            / "extracted"
            / "labels"
        ),
    },
    {
        "split": "val",
        "source_group": "match06",
        "image_root": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_diverse_v1"
            / "val"
        ),
        "label_root": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_diverse_v1_final"
            / "prepared"
            / "val"
        ),
    },
)

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "既存pilot_v2とmatch04・match06をまとめ、"
            "YOLO11n多様化学習用Datasetを作成する"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="新規作成するYOLO Datasetの出力先",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "入力検証と構成表示だけを行い、"
            "Datasetを作成しない"
        ),
    )
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def list_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def list_labels(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.txt")
        if path.is_file()
    )


def build_label_mapping(
    image_paths: list[Path],
    label_paths: list[Path],
    split: str,
    source_group: str,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    used_labels: set[Path] = set()

    for image_path in image_paths:
        image_stem = image_path.stem
        candidates = [
            label_path
            for label_path in label_paths
            if (
                label_path.stem == image_stem
                or label_path.stem.endswith(
                    f"-{image_stem}"
                )
            )
        ]

        if len(candidates) != 1:
            raise ValueError(
                "画像に対応するラベルが一意ではありません: "
                f"split={split}, "
                f"source={source_group}, "
                f"image={image_path}, "
                f"candidate_count={len(candidates)}"
            )

        label_path = candidates[0]

        if label_path in used_labels:
            raise ValueError(
                "同じラベルが複数画像へ割り当てられました: "
                f"{label_path}"
            )

        mapping[image_stem] = label_path
        used_labels.add(label_path)

    unused_labels = sorted(
        set(label_paths) - used_labels
    )

    if unused_labels:
        raise ValueError(
            "画像へ対応しないラベルがあります: "
            + ", ".join(
                str(path)
                for path in unused_labels[:10]
            )
        )

    return mapping


def validate_label(
    label_path: Path,
) -> tuple[bool, str]:
    text = label_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        return False, ""

    lines = text.splitlines()

    if len(lines) != 1:
        raise ValueError(
            "1画像に複数ラベルがあります: "
            f"{label_path}"
        )

    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(
            "YOLOラベルが5列ではありません: "
            f"{label_path}"
        )

    try:
        class_id = int(parts[0])
        values = [
            float(value)
            for value in parts[1:]
        ]
    except ValueError as error:
        raise ValueError(
            "YOLOラベルを数値として読めません: "
            f"{label_path}"
        ) from error

    x_center, y_center, width, height = values

    valid = (
        class_id == 0
        and all(
            math.isfinite(value)
            for value in values
        )
        and 0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )

    if not valid:
        raise ValueError(
            "YOLOラベル値が不正です: "
            f"{label_path}: {lines[0]}"
        )

    normalized = (
        "0 "
        + " ".join(
            format(value, ".12g")
            for value in values
        )
        + "\n"
    )

    return True, normalized


def collect_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for config in SOURCE_CONFIGS:
        split = str(config["split"])
        source_group = str(
            config["source_group"]
        )
        image_root = Path(config["image_root"])
        label_root = Path(config["label_root"])

        if not image_root.is_dir():
            raise FileNotFoundError(
                f"画像フォルダがありません: {image_root}"
            )

        if not label_root.is_dir():
            raise FileNotFoundError(
                f"ラベルフォルダがありません: {label_root}"
            )

        image_paths = list_images(image_root)
        label_paths = list_labels(label_root)

        label_mapping = build_label_mapping(
            image_paths,
            label_paths,
            split,
            source_group,
        )

        source_records: list[
            dict[str, object]
        ] = []

        for image_path in image_paths:
            image_stem = image_path.stem
            label_path = label_mapping[image_stem]
            positive, normalized_label = (
                validate_label(label_path)
            )

            source_records.append(
                {
                    "split": split,
                    "source_group": source_group,
                    "image_path": image_path,
                    "label_path": label_path,
                    "output_image_name": (
                        image_path.name
                    ),
                    "output_label_name": (
                        f"{image_stem}.txt"
                    ),
                    "positive": positive,
                    "normalized_label": (
                        normalized_label
                    ),
                    "image_sha256": (
                        calculate_sha256(
                            image_path
                        )
                    ),
                    "source_label_sha256": (
                        calculate_sha256(
                            label_path
                        )
                    ),
                }
            )

        expected = EXPECTED_SOURCE_COUNTS[
            (split, source_group)
        ]
        total = len(source_records)
        positive_count = sum(
            bool(record["positive"])
            for record in source_records
        )
        negative_count = (
            total - positive_count
        )

        if total != expected["total"]:
            raise ValueError(
                "入力件数が想定と一致しません: "
                f"{split}/{source_group}: "
                f"{total} != {expected['total']}"
            )

        if positive_count != expected["positive"]:
            raise ValueError(
                "正例数が想定と一致しません: "
                f"{split}/{source_group}: "
                f"{positive_count} != "
                f"{expected['positive']}"
            )

        if negative_count != expected["negative"]:
            raise ValueError(
                "負例数が想定と一致しません: "
                f"{split}/{source_group}: "
                f"{negative_count} != "
                f"{expected['negative']}"
            )

        records.extend(source_records)

    return records


def validate_records(
    records: list[dict[str, object]],
) -> None:
    for split in ("train", "val"):
        split_records = [
            record
            for record in records
            if record["split"] == split
        ]
        expected = EXPECTED_SPLIT_COUNTS[split]

        total = len(split_records)
        positive_count = sum(
            bool(record["positive"])
            for record in split_records
        )
        negative_count = total - positive_count

        if total != expected["total"]:
            raise ValueError(
                f"{split}件数が不正です: "
                f"{total} != {expected['total']}"
            )

        if positive_count != expected["positive"]:
            raise ValueError(
                f"{split}正例数が不正です: "
                f"{positive_count} != "
                f"{expected['positive']}"
            )

        if negative_count != expected["negative"]:
            raise ValueError(
                f"{split}負例数が不正です: "
                f"{negative_count} != "
                f"{expected['negative']}"
            )

        image_names = [
            str(record["output_image_name"])
            for record in split_records
        ]
        label_names = [
            str(record["output_label_name"])
            for record in split_records
        ]

        if len(image_names) != len(set(image_names)):
            raise ValueError(
                f"{split}の画像名が重複しています"
            )

        if len(label_names) != len(set(label_names)):
            raise ValueError(
                f"{split}のラベル名が重複しています"
            )

    train_records = [
        record
        for record in records
        if record["split"] == "train"
    ]
    val_records = [
        record
        for record in records
        if record["split"] == "val"
    ]

    train_image_names = {
        str(record["output_image_name"])
        for record in train_records
    }
    val_image_names = {
        str(record["output_image_name"])
        for record in val_records
    }

    name_overlap = sorted(
        train_image_names & val_image_names
    )

    if name_overlap:
        raise ValueError(
            "trainとvalで画像名が重複しています: "
            + ", ".join(name_overlap[:10])
        )

    train_hashes = {
        str(record["image_sha256"])
        for record in train_records
    }
    val_hashes = {
        str(record["image_sha256"])
        for record in val_records
    }

    hash_overlap = sorted(
        train_hashes & val_hashes
    )

    if hash_overlap:
        raise ValueError(
            "trainとvalで同一内容画像があります: "
            + ", ".join(hash_overlap[:10])
        )


def summarize(
    records: list[dict[str, object]],
) -> None:
    print("=== YOLO diverse_v1 Dataset構成 ===")

    for split in ("train", "val"):
        print(f"\n[{split}]")

        split_records = [
            record
            for record in records
            if record["split"] == split
        ]

        source_groups = sorted(
            {
                str(record["source_group"])
                for record in split_records
            }
        )

        for source_group in source_groups:
            group_records = [
                record
                for record in split_records
                if (
                    record["source_group"]
                    == source_group
                )
            ]
            positive_count = sum(
                bool(record["positive"])
                for record in group_records
            )

            print(
                f"{source_group}: "
                f"total={len(group_records)}, "
                f"positive={positive_count}, "
                f"negative="
                f"{len(group_records) - positive_count}"
            )

        positive_count = sum(
            bool(record["positive"])
            for record in split_records
        )
        print(
            f"split total: "
            f"total={len(split_records)}, "
            f"positive={positive_count}, "
            f"negative="
            f"{len(split_records) - positive_count}"
        )


def write_dataset(
    records: list[dict[str, object]],
    output_root: Path,
) -> None:
    output_root = output_root.resolve()

    if output_root.exists():
        raise FileExistsError(
            f"出力先が既に存在します: {output_root}"
        )

    temporary_root = output_root.with_name(
        f".{output_root.name}_tmp_"
        f"{uuid.uuid4().hex}"
    )

    if temporary_root.exists():
        raise FileExistsError(
            f"一時出力先が既に存在します: "
            f"{temporary_root}"
        )

    try:
        for split in ("train", "val"):
            (
                temporary_root
                / "images"
                / split
            ).mkdir(parents=True)
            (
                temporary_root
                / "labels"
                / split
            ).mkdir(parents=True)

        manifest_rows: list[
            dict[str, object]
        ] = []

        for record in records:
            split = str(record["split"])
            image_source = Path(
                record["image_path"]
            )
            image_output = (
                temporary_root
                / "images"
                / split
                / str(record["output_image_name"])
            )
            label_output = (
                temporary_root
                / "labels"
                / split
                / str(record["output_label_name"])
            )

            shutil.copy2(
                image_source,
                image_output,
            )

            copied_image_hash = calculate_sha256(
                image_output
            )

            if (
                copied_image_hash
                != record["image_sha256"]
            ):
                raise RuntimeError(
                    "画像コピー後のSHA-256が"
                    "一致しません: "
                    f"{image_output}"
                )

            label_text = str(
                record["normalized_label"]
            )
            label_output.write_text(
                label_text,
                encoding="utf-8",
            )

            manifest_rows.append(
                {
                    "split": split,
                    "source_group": (
                        record["source_group"]
                    ),
                    "output_image": (
                        image_output.name
                    ),
                    "output_label": (
                        label_output.name
                    ),
                    "positive": int(
                        bool(record["positive"])
                    ),
                    "image_sha256": (
                        record["image_sha256"]
                    ),
                    "source_image": str(
                        image_source.resolve()
                    ),
                    "source_label": str(
                        Path(
                            record["label_path"]
                        ).resolve()
                    ),
                    "source_label_sha256": (
                        record[
                            "source_label_sha256"
                        ]
                    ),
                }
            )

        data_yaml_path = (
            temporary_root / "data.yaml"
        )
        data_yaml_text = (
            f"path: "
            f"{output_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: ball\n"
        )
        data_yaml_path.write_text(
            data_yaml_text,
            encoding="utf-8",
        )

        manifest_path = (
            temporary_root / "manifest.csv"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    manifest_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        metadata = {
            "dataset": "yolo_diverse_v1",
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "class_count": 1,
            "class_names": ["ball"],
            "output_root": str(output_root),
            "data_yaml": str(
                output_root / "data.yaml"
            ),
            "preparation_script": str(
                Path(__file__).resolve()
            ),
            "preparation_script_sha256": (
                calculate_sha256(
                    Path(__file__).resolve()
                )
            ),
            "splits": {
                split: {
                    "total": (
                        EXPECTED_SPLIT_COUNTS[
                            split
                        ]["total"]
                    ),
                    "positive": (
                        EXPECTED_SPLIT_COUNTS[
                            split
                        ]["positive"]
                    ),
                    "negative": (
                        EXPECTED_SPLIT_COUNTS[
                            split
                        ]["negative"]
                    ),
                }
                for split in ("train", "val")
            },
            "source_groups": [
                {
                    "split": config["split"],
                    "source_group": (
                        config["source_group"]
                    ),
                    "image_root": str(
                        Path(
                            config["image_root"]
                        ).resolve()
                    ),
                    "label_root": str(
                        Path(
                            config["label_root"]
                        ).resolve()
                    ),
                    **EXPECTED_SOURCE_COUNTS[
                        (
                            str(config["split"]),
                            str(
                                config[
                                    "source_group"
                                ]
                            ),
                        )
                    ],
                }
                for config in SOURCE_CONFIGS
            ],
            "validation": {
                "image_label_correspondence": True,
                "output_name_uniqueness": True,
                "cross_split_name_overlap": 0,
                "cross_split_sha256_overlap": 0,
                "copy_sha256_verified": True,
            },
        }

        metadata_path = (
            temporary_root / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        temporary_root.replace(output_root)

    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()

    if output_root.exists():
        raise FileExistsError(
            f"出力先が既に存在します: {output_root}"
        )

    records = collect_records()
    validate_records(records)
    summarize(records)

    print("\n=== 検証結果 ===")
    print("train・val画像名重複: 0")
    print("train・val画像SHA-256重複: 0")
    print("画像・ラベル対応: OK")
    print("YOLOラベル値: OK")

    if args.dry_run:
        print(
            "\ndry-runのためDatasetは"
            "作成していません"
        )
        return

    write_dataset(records, output_root)

    print("\nYOLO diverse_v1を作成しました")
    print(f"出力先: {output_root}")
    print(
        f"data.yaml: "
        f"{output_root / 'data.yaml'}"
    )
    print(
        f"manifest: "
        f"{output_root / 'manifest.csv'}"
    )
    print(
        f"metadata: "
        f"{output_root / 'metadata.json'}"
    )


if __name__ == "__main__":
    main()