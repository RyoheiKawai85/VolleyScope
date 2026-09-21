import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import cv2

from convert_yolo_to_tracknet_csv import (
    convert_positive_label,
    read_classes,
    read_nonempty_lines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FRAME_ROOT = (
    PROJECT_ROOT
    / "data"
    / "frames"
    / "tracknet_diverse_v1"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_FRAME_ROOT / "manifest.csv"
)

DEFAULT_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_diverse_v1_final"
    / "extracted"
    / "labels"
)

DEFAULT_CLASSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_diverse_v1_final"
    / "extracted"
    / "classes.txt"
)

DEFAULT_RAW_EXPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_diverse_v1_final"
    / "raw_export"
    / "label_studio_yolo_export.zip"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "tracknet_diverse_v1_final"
    / "prepared"
)

EXPECTED_RAW_EXPORT_SHA256 = (
    "2D2AD3D6A5956404629DC547D04DEAF73"
    "F2B1111E659427B4F6B7177755557C3"
)

LABEL_STEM_PATTERN = re.compile(
    r"(match0[46]_r(\d{2})_frame_(\d{6}))$"
)

REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "match_name",
    "rally_id",
    "file_name",
    "relative_path",
    "local_frame_index",
    "source_video",
    "source_frame_index",
    "source_time_seconds",
}


def parse_args() -> argparse.Namespace:
    """ラベル整理に使用する入出力を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3多様化用のLabel Studioラベルを"
            "train・val・rally別に整理する"
        ),
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=DEFAULT_FRAME_ROOT,
        help="抽出画像データセットのルート",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="抽出画像と元動画の対応表",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Label Studioから展開したYOLOラベル",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES_PATH,
        help="Label Studioから展開したclasses.txt",
    )
    parser.add_argument(
        "--raw-export",
        type=Path,
        default=DEFAULT_RAW_EXPORT_PATH,
        help="正式保存したLabel Studio ZIP",
    )
    parser.add_argument(
        "--expected-raw-export-sha256",
        default=EXPECTED_RAW_EXPORT_SHA256,
        help="正式保存したZIPに期待するSHA-256",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="整理済みラベルとTrackNet CSVの新規出力先",
    )
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def validate_inputs(args: argparse.Namespace) -> None:
    """入力の存在、ZIPの同一性、上書き防止を確認する。"""
    args.frame_root = args.frame_root.resolve()
    args.manifest = args.manifest.resolve()
    args.label_dir = args.label_dir.resolve()
    args.classes = args.classes.resolve()
    args.raw_export = args.raw_export.resolve()
    args.output_root = args.output_root.resolve()

    if not args.frame_root.is_dir():
        raise FileNotFoundError(
            f"画像ルートがありません: {args.frame_root}"
        )

    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"manifestがありません: {args.manifest}"
        )

    if not args.label_dir.is_dir():
        raise FileNotFoundError(
            f"ラベルフォルダがありません: {args.label_dir}"
        )

    if not args.classes.is_file():
        raise FileNotFoundError(
            f"classes.txtがありません: {args.classes}"
        )

    if not args.raw_export.is_file():
        raise FileNotFoundError(
            f"正式保存したZIPがありません: {args.raw_export}"
        )

    actual_hash = calculate_sha256(args.raw_export)

    if (
        actual_hash
        != args.expected_raw_export_sha256.upper()
    ):
        raise RuntimeError(
            "Label Studio ZIPのSHA-256が一致しません: "
            f"期待={args.expected_raw_export_sha256}, "
            f"実際={actual_hash}"
        )

    if args.output_root.exists():
        raise FileExistsError(
            "上書きを防ぐため停止します: "
            f"{args.output_root}"
        )

    read_classes(args.classes)


def read_manifest(
    manifest_path: Path,
    frame_root: Path,
) -> list[dict[str, str]]:
    """manifestを読み、全画像との対応を検証する。"""
    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        if reader.fieldnames is None:
            raise ValueError(
                "manifestにヘッダーがありません"
            )

        missing_columns = sorted(
            REQUIRED_MANIFEST_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "manifestに必要な列がありません: "
                f"{missing_columns}"
            )

        rows = list(reader)

    if len(rows) != 256:
        raise ValueError(
            "manifest行数が256ではありません: "
            f"{len(rows)}"
        )

    seen_stems = set()
    grouped_local_indices = {}

    for row_number, row in enumerate(rows, start=2):
        split = row["split"].strip()
        match_name = row["match_name"].strip()
        file_name = row["file_name"].strip()
        relative_path = row["relative_path"].strip()

        if split not in {"train", "val"}:
            raise ValueError(
                "splitがtrainまたはvalではありません: "
                f"行={row_number}, 値={split}"
            )

        try:
            rally_id = int(row["rally_id"])
            local_frame_index = int(
                row["local_frame_index"]
            )
            source_frame_index = int(
                row["source_frame_index"]
            )
            float(row["source_time_seconds"])
        except ValueError as error:
            raise ValueError(
                "manifestの数値列を変換できません: "
                f"行={row_number}"
            ) from error

        if (
            rally_id <= 0
            or local_frame_index < 0
            or source_frame_index < 0
        ):
            raise ValueError(
                "manifestに負数または不正なIDがあります: "
                f"行={row_number}"
            )

        file_stem = Path(file_name).stem

        if file_stem in seen_stems:
            raise ValueError(
                "manifestの画像名が重複しています: "
                f"{file_name}"
            )

        seen_stems.add(file_stem)

        image_path = frame_root / relative_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"manifestの画像がありません: {image_path}"
            )

        name_match = LABEL_STEM_PATTERN.fullmatch(
            file_stem
        )

        if name_match is None:
            raise ValueError(
                "画像名が期待形式ではありません: "
                f"{file_name}"
            )

        name_rally_id = int(name_match.group(2))
        name_local_index = int(name_match.group(3))

        if name_rally_id != rally_id:
            raise ValueError(
                "画像名とmanifestのrally_idが異なります: "
                f"{file_name}"
            )

        if name_local_index != local_frame_index:
            raise ValueError(
                "画像名とmanifestのローカル番号が"
                "異なります: "
                f"{file_name}"
            )

        group_key = (
            split,
            match_name,
            rally_id,
        )
        grouped_local_indices.setdefault(
            group_key,
            [],
        ).append(local_frame_index)

    for group_key, local_indices in (
        grouped_local_indices.items()
    ):
        if local_indices != list(
            range(len(local_indices))
        ):
            raise ValueError(
                "ラリー内のローカル番号が"
                "0から連続していません: "
                f"{group_key}"
            )

    return rows


def build_label_mapping(
    label_dir: Path,
) -> dict[str, Path]:
    """Label Studio接頭辞付きラベルを元画像名へ対応付ける。"""
    label_mapping = {}

    for label_path in sorted(
        label_dir.glob("*.txt")
    ):
        match = LABEL_STEM_PATTERN.search(
            label_path.stem
        )

        if match is None:
            raise ValueError(
                "元画像名を復元できません: "
                f"{label_path.name}"
            )

        source_stem = match.group(1)

        if source_stem in label_mapping:
            raise ValueError(
                "同じ元画像に対応するラベルが"
                "複数あります: "
                f"{source_stem}"
            )

        label_mapping[source_stem] = label_path

    if len(label_mapping) != 256:
        raise ValueError(
            "復元できたラベル数が256ではありません: "
            f"{len(label_mapping)}"
        )

    return label_mapping


def convert_row(
    manifest_row: dict[str, str],
    frame_root: Path,
    label_path: Path,
) -> dict[str, int]:
    """1枚のYOLOラベルをTrackNet座標へ変換する。"""
    image_path = (
        frame_root
        / manifest_row["relative_path"]
    )
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"画像を開けません: {image_path}"
        )

    image_height, image_width = image.shape[:2]
    label_lines = read_nonempty_lines(label_path)

    if len(label_lines) > 1:
        raise ValueError(
            "1枚に複数の矩形があります: "
            f"{label_path}"
        )

    local_frame_index = int(
        manifest_row["local_frame_index"]
    )

    if not label_lines:
        return {
            "Frame": local_frame_index,
            "Visibility": 0,
            "X": 0,
            "Y": 0,
        }

    x_pixel, y_pixel = convert_positive_label(
        label_lines[0],
        image_width,
        image_height,
        label_path,
    )

    return {
        "Frame": local_frame_index,
        "Visibility": 1,
        "X": x_pixel,
        "Y": y_pixel,
    }


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """辞書の一覧をUTF-8 BOM付きCSVへ保存する。"""
    with path.open(
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


def main() -> None:
    """Label Studioラベルを整理してTrackNet CSVを作る。"""
    args = parse_args()
    validate_inputs(args)

    manifest_rows = read_manifest(
        args.manifest,
        args.frame_root,
    )
    label_mapping = build_label_mapping(
        args.label_dir
    )

    manifest_stems = {
        Path(row["file_name"]).stem
        for row in manifest_rows
    }
    label_stems = set(label_mapping)

    missing_label_stems = sorted(
        manifest_stems - label_stems
    )
    unexpected_label_stems = sorted(
        label_stems - manifest_stems
    )

    if missing_label_stems:
        raise ValueError(
            "ラベルがない画像があります: "
            f"{missing_label_stems}"
        )

    if unexpected_label_stems:
        raise ValueError(
            "対応画像がないラベルがあります: "
            f"{unexpected_label_stems}"
        )

    args.output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=".tracknet_diverse_labels_staging_",
            dir=args.output_root.parent,
        )
    )
    staging_root = (
        staging_parent / args.output_root.name
    )
    staging_root.mkdir()

    prepared_manifest_rows = []
    rally_tracknet_rows = {}
    rally_manifest_rows = {}
    split_counts = {
        "train": {
            "total": 0,
            "positive": 0,
            "empty": 0,
        },
        "val": {
            "total": 0,
            "positive": 0,
            "empty": 0,
        },
    }

    try:
        for manifest_row in manifest_rows:
            split = manifest_row["split"]
            match_name = manifest_row[
                "match_name"
            ]
            rally_id = int(
                manifest_row["rally_id"]
            )
            source_stem = Path(
                manifest_row["file_name"]
            ).stem
            label_path = label_mapping[
                source_stem
            ]

            tracknet_row = convert_row(
                manifest_row,
                args.frame_root,
                label_path,
            )

            group_key = (
                split,
                match_name,
                rally_id,
            )

            rally_root = (
                staging_root
                / split
                / match_name
                / f"rally_{rally_id:02d}"
            )
            normalized_label_dir = (
                rally_root / "labels"
            )
            normalized_label_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            normalized_label_path = (
                normalized_label_dir
                / f"{source_stem}.txt"
            )

            shutil.copy2(
                label_path,
                normalized_label_path,
            )

            output_relative_label = (
                normalized_label_path
                .relative_to(staging_root)
                .as_posix()
            )

            prepared_row = dict(manifest_row)
            prepared_row[
                "normalized_label_path"
            ] = output_relative_label
            prepared_row["visibility"] = (
                tracknet_row["Visibility"]
            )
            prepared_row["x_pixel"] = (
                tracknet_row["X"]
            )
            prepared_row["y_pixel"] = (
                tracknet_row["Y"]
            )

            prepared_manifest_rows.append(
                prepared_row
            )
            rally_tracknet_rows.setdefault(
                group_key,
                [],
            ).append(tracknet_row)
            rally_manifest_rows.setdefault(
                group_key,
                [],
            ).append(prepared_row)

            split_counts[split]["total"] += 1

            if tracknet_row["Visibility"] == 1:
                split_counts[split][
                    "positive"
                ] += 1
            else:
                split_counts[split][
                    "empty"
                ] += 1

        for group_key, tracknet_rows in (
            rally_tracknet_rows.items()
        ):
            split, match_name, rally_id = (
                group_key
            )
            rally_root = (
                staging_root
                / split
                / match_name
                / f"rally_{rally_id:02d}"
            )

            tracknet_rows.sort(
                key=lambda row: row["Frame"]
            )
            group_manifest_rows = (
                rally_manifest_rows[group_key]
            )
            group_manifest_rows.sort(
                key=lambda row: int(
                    row["local_frame_index"]
                )
            )

            expected_frames = list(
                range(len(tracknet_rows))
            )
            actual_frames = [
                row["Frame"]
                for row in tracknet_rows
            ]

            if actual_frames != expected_frames:
                raise RuntimeError(
                    "TrackNet CSVのFrameが"
                    "0から連続していません: "
                    f"{group_key}"
                )

            write_csv(
                rally_root
                / "tracknet_labels.csv",
                tracknet_rows,
                [
                    "Frame",
                    "Visibility",
                    "X",
                    "Y",
                ],
            )

            write_csv(
                rally_root / "manifest.csv",
                group_manifest_rows,
                list(
                    group_manifest_rows[0].keys()
                ),
            )

        write_csv(
            staging_root / "manifest.csv",
            prepared_manifest_rows,
            list(
                prepared_manifest_rows[0].keys()
            ),
        )

        total_count = len(
            prepared_manifest_rows
        )
        positive_count = sum(
            counts["positive"]
            for counts in split_counts.values()
        )
        empty_count = sum(
            counts["empty"]
            for counts in split_counts.values()
        )

        if total_count != 256:
            raise RuntimeError(
                "整理後の総数が256ではありません"
            )

        if positive_count != 242:
            raise RuntimeError(
                "整理後の正例数が242ではありません: "
                f"{positive_count}"
            )

        if empty_count != 14:
            raise RuntimeError(
                "整理後の空ラベル数が14ではありません: "
                f"{empty_count}"
            )

        summary = {
            "dataset": "tracknet_diverse_v1",
            "raw_export": str(args.raw_export),
            "raw_export_sha256": (
                calculate_sha256(
                    args.raw_export
                )
            ),
            "source_manifest": str(
                args.manifest
            ),
            "source_manifest_sha256": (
                calculate_sha256(
                    args.manifest
                )
            ),
            "preparation_script": str(
                Path(__file__).resolve()
            ),
            "preparation_script_sha256": (
                calculate_sha256(
                    Path(__file__).resolve()
                )
            ),
            "total": total_count,
            "positive": positive_count,
            "empty": empty_count,
            "splits": split_counts,
            "rallies": [
                {
                    "split": split,
                    "match_name": match_name,
                    "rally_id": rally_id,
                    "total": len(rows),
                    "positive": sum(
                        row["Visibility"] == 1
                        for row in rows
                    ),
                    "empty": sum(
                        row["Visibility"] == 0
                        for row in rows
                    ),
                }
                for (
                    split,
                    match_name,
                    rally_id,
                ), rows in sorted(
                    rally_tracknet_rows.items()
                )
            ],
        }

        with (
            staging_root / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(
                summary,
                summary_file,
                ensure_ascii=False,
                indent=2,
            )
            summary_file.write("\n")

        shutil.move(
            str(staging_root),
            str(args.output_root),
        )
    finally:
        shutil.rmtree(
            staging_parent,
            ignore_errors=True,
        )

    print(
        "TrackNetV3多様化用ラベルを整理しました"
    )
    print(f"出力先: {args.output_root}")
    print(
        "全体: "
        f"total={len(prepared_manifest_rows)}, "
        f"positive={positive_count}, "
        f"empty={empty_count}"
    )

    for split, counts in split_counts.items():
        print(
            f"{split}: "
            f"total={counts['total']}, "
            f"positive={counts['positive']}, "
            f"empty={counts['empty']}"
        )

    print()
    print("ラリー別:")

    for rally in summary["rallies"]:
        print(
            f"{rally['split']} / "
            f"{rally['match_name']} / "
            f"rally {rally['rally_id']}: "
            f"total={rally['total']}, "
            f"positive={rally['positive']}, "
            f"empty={rally['empty']}"
        )


if __name__ == "__main__":
    main()