import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import ultralytics
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "yolo_pilot_v2"
    / "data.yaml"
)

DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_training"
    / "pilot_v2_transfer_img1280_bs2_lr1e-4_seed13"
)

EXPECTED_WEIGHTS_SHA256 = (
    "0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1"
)


def parse_args() -> argparse.Namespace:
    """YOLO追加学習の固定条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "COCO事前学習済みYOLO11nを、"
            "TrackNetV3と同じバレーボール画像で"
            "3 epoch追加学習する"
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Ultralytics形式のdata.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="追加学習の初期値にするYOLO重み",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="学習結果の新規出力先",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="追加学習するepoch数",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="YOLOへ入力する正方形画像サイズ",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="学習とvalのbatch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adamの固定学習率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="乱数シード",
    )
    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def validate_inputs(
    data_yaml: Path,
    weights: Path,
    output_dir: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[Path, Path, Path]:
    """入力、出力先、学習条件を検証する。"""
    data_yaml = data_yaml.resolve()
    weights = weights.resolve()
    output_dir = output_dir.resolve()

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yamlがありません: {data_yaml}"
        )

    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO重みがありません: {weights}"
        )

    weights_hash = calculate_sha256(weights)

    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            "YOLO公開重みのSHA-256が"
            "想定と一致しません: "
            f"{weights_hash}"
        )

    if output_dir.exists():
        raise FileExistsError(
            "出力先が既に存在します: "
            f"{output_dir}"
        )

    if epochs <= 0:
        raise ValueError(
            "--epochsは1以上にしてください"
        )

    if image_size <= 0:
        raise ValueError(
            "--image-sizeは1以上にしてください"
        )

    if batch_size <= 0:
        raise ValueError(
            "--batch-sizeは1以上にしてください"
        )

    if learning_rate <= 0:
        raise ValueError(
            "--learning-rateは0より大きくしてください"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "この学習はCUDA GPUを前提とします"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return data_yaml, weights, output_dir


def main() -> None:
    """YOLO11nをバレーボール用データで追加学習する。"""
    args = parse_args()

    (
        data_yaml,
        weights,
        output_dir,
    ) = validate_inputs(
        args.data,
        args.weights,
        args.output_dir,
        args.epochs,
        args.image_size,
        args.batch_size,
        args.learning_rate,
    )

    script_path = Path(__file__).resolve()
    weights_hash = calculate_sha256(weights)
    data_yaml_hash = calculate_sha256(data_yaml)
    script_hash = calculate_sha256(script_path)

    training_arguments = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "device": 0,
        "workers": 0,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": False,
        "optimizer": "Adam",
        "lr0": args.learning_rate,
        "lrf": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "warmup_epochs": 0.0,
        "seed": args.seed,
        "deterministic": True,
        "fraction": 1.0,
        "val": True,
        "save": True,
        "save_period": 1,
        "plots": False,
        "cache": False,
        "amp": False,
        "augment": False,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "close_mosaic": 0,
    }

    print("YOLO11nパイロット追加学習を開始します")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"data.yaml: {data_yaml}")
    print(f"data.yaml SHA-256: {data_yaml_hash}")
    print(f"公開重み: {weights}")
    print(f"公開重みSHA-256: {weights_hash}")
    print(f"学習スクリプトSHA-256: {script_hash}")
    print(f"epoch数: {args.epochs}")
    print(f"画像サイズ: {args.image_size}")
    print(f"batch size: {args.batch_size}")
    print(f"学習率: {args.learning_rate:.8f}")
    print("optimizer: Adam")
    print("データ拡張: 使用しない")
    print("AMP: 使用しない")
    print(f"出力先: {output_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(weights))

    start_time = perf_counter()

    results = model.train(
        **training_arguments
    )

    elapsed_seconds = (
        perf_counter() - start_time
    )
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    if not output_dir.is_dir():
        raise RuntimeError(
            "予定した出力先が作成されませんでした: "
            f"{output_dir}"
        )

    weights_dir = output_dir / "weights"
    best_checkpoint = weights_dir / "best.pt"
    last_checkpoint = weights_dir / "last.pt"

    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"best.ptがありません: {best_checkpoint}"
        )

    if not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"last.ptがありません: {last_checkpoint}"
        )

    metadata = {
        "task": "yolo_pilot_transfer_learning",
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": data_yaml_hash,
        "initial_weights": str(weights),
        "initial_weights_sha256": weights_hash,
        "training_script": str(script_path),
        "training_script_sha256": script_hash,
        "training_arguments": training_arguments,
        "result_save_dir": str(results.save_dir),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_mib": (
            peak_allocated_mib
        ),
        "peak_reserved_vram_mib": (
            peak_reserved_mib
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
        "best_checkpoint_sha256": (
            calculate_sha256(best_checkpoint)
        ),
        "last_checkpoint": str(
            last_checkpoint
        ),
        "last_checkpoint_sha256": (
            calculate_sha256(last_checkpoint)
        ),
    }

    metadata_path = (
        output_dir / "training_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("YOLO11nパイロット追加学習が完了しました")
    print(f"学習・val時間: {elapsed_seconds:.2f}秒")
    print(
        "ピーク割当VRAM: "
        f"{peak_allocated_mib:.1f} MiB"
    )
    print(
        "ピーク予約VRAM: "
        f"{peak_reserved_mib:.1f} MiB"
    )
    print(
        "best.pt SHA-256: "
        f"{metadata['best_checkpoint_sha256']}"
    )
    print(
        "last.pt SHA-256: "
        f"{metadata['last_checkpoint_sha256']}"
    )
    print(f"出力先: {output_dir}")
    print(f"再現情報: {metadata_path}")


if __name__ == "__main__":
    main()