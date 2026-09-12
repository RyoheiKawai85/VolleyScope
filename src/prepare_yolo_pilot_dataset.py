import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "yolo_pilot_v2"
)

SPLIT_CONFIGS = {
    "train": {
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "train"
            / "images"
        ),
        "label_dir": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_train_final"
            / "extracted"
            / "labels"
        ),
        "classes": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_train_final"
            / "extracted"
            / "classes.txt"
        ),
        "expected_total": 128,
        "expected_positive": 126,
        "expected_empty": 2,
    },
    "val": {
        "image_dir": (
            PROJECT_ROOT
            / "data"
            / "frames"
            / "tracknet_pilot_v2"
            / "val"
            / "images"
        ),
        "label_dir": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_val_final"
            / "extracted"
            / "labels"
        ),
        "classes": (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "tracknet_pilot_v2_val_final"
            / "extracted"
            / "classes.txt"
        ),
        "expected_total": 120,
        "expected_positive": 118,
        "expected_empty": 2,
    },
}

LABEL_STEM_PATTERN = re.compile(
    r"(frame_\d{6})$"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def parse_args() -> argparse.Namespace:
    """YOLO学習用データセットの出力先を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "既存のTrackNetV3パイロット画像と"
            "YOLOラベルからYOLO学習用データを作る"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="YOLO学習用データセットの新規出力先",
    )
    return parser.parse_args()


def read_classes(classes_path: Path) -> list[str]:
    """クラス定義を読み込んでball一種類か確認する。"""
    classes = [
        line.strip()
        for line in classes_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if classes != ["ball"]:
        raise ValueError(
            "クラス定義がball一種類ではありません: "
            f"{classes_path} -> {classes}"
        )

    return classes


def collect_images(
    image_dir: Path,
) -> dict[str, Path]:
    """画像をファイル名の幹へ対応付ける。"""
    images = {}

    for image_path in sorted(image_dir.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in SUPPORTED_IMAGE_EXTENSIONS
        ):
            continue

        if image_path.stem in images:
            raise ValueError(
                "同じ名前の画像が複数あります: "
                f"{image_path.stem}"
            )

        images[image_path.stem] = image_path

    if not images:
        raise RuntimeError(
            f"画像がありません: {image_dir}"
        )

    return images


def collect_labels(
    label_dir: Path,
) -> dict[str, Path]:
    """Label Studio接頭辞付きラベルを画像名へ対応付ける。"""
    labels = {}

    for label_path in sorted(
        label_dir.glob("*.txt")
    ):
        match = LABEL_STEM_PATTERN.search(
            label_path.stem
        )

        if match is None:
            raise ValueError(
                "ラベル名から画像名を取得できません: "
                f"{label_path.name}"
            )

        image_stem = match.group(1)

        if image_stem in labels:
            raise ValueError(
                "同じ画像に対応するラベルが複数あります: "
                f"{image_stem}"
            )

        labels[image_stem] = label_path

    if not labels:
        raise RuntimeError(
            f"ラベルがありません: {label_dir}"
        )

    return labels


def validate_label(
    label_path: Path,
) -> bool:
    """YOLOラベルを検証し、正例ならTrueを返す。"""
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    if len(lines) != 1:
        raise ValueError(
            "1画像に複数のラベルがあります: "
            f"{label_path}"
        )

    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(
            "YOLOラベルが5列ではありません: "
            f"{label_path}"
        )

    if parts[0] != "0":
        raise ValueError(
            "ball以外のクラスIDがあります: "
            f"{label_path} -> {parts[0]}"
        )

    values = [
        float(value)
        for value in parts[1:]
    ]
    x_center, y_center, width, height = values

    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(
            "YOLO座標が範囲外です: "
            f"{label_path} -> {values}"
        )

    return True


def prepare_split(
    split: str,
    config: dict,
    staging_root: Path,
) -> dict:
    """1 splitを検証してYOLO形式へコピーする。"""
    image_dir = config["image_dir"].resolve()
    label_dir = config["label_dir"].resolve()
    classes_path = config["classes"].resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"画像フォルダがありません: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"ラベルフォルダがありません: {label_dir}"
        )

    if not classes_path.is_file():
        raise FileNotFoundError(
            f"classes.txtがありません: {classes_path}"
        )

    read_classes(classes_path)
    images = collect_images(image_dir)
    labels = collect_labels(label_dir)

    missing_labels = sorted(
        set(images) - set(labels)
    )
    unexpected_labels = sorted(
        set(labels) - set(images)
    )

    if missing_labels:
        raise ValueError(
            "ラベルがない画像があります: "
            f"{missing_labels}"
        )

    if unexpected_labels:
        raise ValueError(
            "対応画像がないラベルがあります: "
            f"{unexpected_labels}"
        )

    output_image_dir = (
        staging_root / "images" / split
    )
    output_label_dir = (
        staging_root / "labels" / split
    )
    output_image_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    output_label_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    positive_count = 0
    empty_count = 0

    for image_stem in sorted(images):
        image_path = images[image_stem]
        label_path = labels[image_stem]
        is_positive = validate_label(label_path)

        if is_positive:
            positive_count += 1
        else:
            empty_count += 1

        shutil.copy2(
            image_path,
            output_image_dir / image_path.name,
        )
        shutil.copy2(
            label_path,
            output_label_dir / f"{image_stem}.txt",
        )

    total_count = len(images)

    expected_values = {
        "total": config["expected_total"],
        "positive": config["expected_positive"],
        "empty": config["expected_empty"],
    }
    actual_values = {
        "total": total_count,
        "positive": positive_count,
        "empty": empty_count,
    }

    if actual_values != expected_values:
        raise ValueError(
            f"{split}の件数が想定と異なります: "
            f"expected={expected_values}, "
            f"actual={actual_values}"
        )

    return {
        "image_source": str(image_dir),
        "label_source": str(label_dir),
        **actual_values,
    }


def main() -> None:
    """安全な一時出力を経由してデータセットを作る。"""
    args = parse_args()
    output_root = args.output_root.resolve()

    if output_root.exists():
        raise FileExistsError(
            "出力先が既に存在します: "
            f"{output_root}"
        )

    output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{output_root.name}_staging_"
            ),
            dir=output_root.parent,
        )
    )

    print(f"一時出力先: {staging_root}")

    try:
        split_results = {}

        for split, config in SPLIT_CONFIGS.items():
            split_results[split] = prepare_split(
                split,
                config,
                staging_root,
            )

        data_yaml = (
            f"path: {output_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: ball\n"
        )

        (
            staging_root / "data.yaml"
        ).write_text(
            data_yaml,
            encoding="utf-8",
        )

        metadata = {
            "dataset": "yolo_pilot_v2",
            "class_count": 1,
            "class_names": ["ball"],
            "splits": split_results,
        }

        (
            staging_root / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        staging_root.replace(output_root)

    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    print("YOLO学習用データセットを作成しました")
    print(f"出力先: {output_root}")

    for split, result in split_results.items():
        print(
            f"{split}: "
            f"total={result['total']}, "
            f"positive={result['positive']}, "
            f"empty={result['empty']}"
        )

    print(f"data.yaml: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()