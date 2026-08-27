import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tracknet_official_pilot_v2"
)

DEFAULT_REFERENCE_ROOT = Path(
    r"C:\GitHub\TrackNetV3-reference"
)

DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_REFERENCE_ROOT
    / "ckpts"
    / "TrackNet_best.pt"
)

DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_training"
    / "pilot_v2_pretrained_baseline"
    / "metrics.json"
)

EXPECTED_REFERENCE_COMMIT = (
    "77c123ad4dd449b7d275f16cc43f316ba5b54042"
)

EXPECTED_SEQUENCE_LENGTH = 8
EXPECTED_BACKGROUND_MODE = "concat"
EXPECTED_TOLERANCE = 4
OFFICIAL_HEATMAP_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """学習前val評価の実行条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3公開重みを"
            "VolleyScopeのvalデータで評価する"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="公式互換データセットのルート",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="固定したTrackNetV3公式実装のルート",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="評価するTrackNetV3公開重み",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="評価結果JSONの保存先",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="val評価時のbatch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="再現性確認用の乱数シード",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="公式評価の進捗バーを表示する",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """入力パスと数値引数を検証する。"""
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            "公式互換データセットが"
            f"見つかりません: {args.dataset_root}"
        )

    if not args.reference_root.is_dir():
        raise FileNotFoundError(
            "TrackNetV3参照リポジトリが"
            f"見つかりません: {args.reference_root}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "公開チェックポイントが"
            f"見つかりません: {args.checkpoint}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-sizeには1以上を"
            "指定してください"
        )

    if args.output_json.exists():
        raise FileExistsError(
            "評価結果の上書きを防ぐため停止します: "
            f"{args.output_json}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを使用できません"
        )


def import_official_modules(
    reference_root: Path,
):
    """固定した公式実装から評価機能を読み込む。"""
    reference_path = str(
        reference_root.resolve()
    )

    if reference_path not in sys.path:
        sys.path.insert(
            0,
            reference_path,
        )

    from dataset import (
        Shuttlecock_Trajectory_Dataset,
    )
    from test import eval_tracknet
    from utils.general import get_model

    return (
        Shuttlecock_Trajectory_Dataset,
        eval_tracknet,
        get_model,
    )


def calculate_sha256(
    file_path: Path,
) -> str:
    """ファイル内容からSHA-256を計算する。"""
    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while True:
            chunk = input_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def get_reference_commit(
    reference_root: Path,
) -> str:
    """参照リポジトリの現在のcommitを取得する。"""
    completed_process = subprocess.run(
        [
            "git",
            "-C",
            str(reference_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return completed_process.stdout.strip()


def normalize_metrics(
    result: dict,
) -> dict:
    """NumPy由来の数値をJSONへ保存可能な型にする。"""
    count_keys = {
        "TP",
        "TN",
        "FP1",
        "FP2",
        "FN",
    }

    normalized = {}

    for key, value in result.items():
        if key in count_keys:
            normalized[key] = int(
                round(float(value))
            )
        else:
            normalized[key] = float(value)

    return normalized


def main() -> None:
    """公開重みの学習前val基準値を測定する。"""
    args = parse_args()

    args.dataset_root = (
        args.dataset_root.resolve()
    )
    args.reference_root = (
        args.reference_root.resolve()
    )
    args.checkpoint = (
        args.checkpoint.resolve()
    )
    args.output_json = (
        args.output_json.resolve()
    )

    validate_args(args)

    reference_commit = get_reference_commit(
        args.reference_root
    )

    if reference_commit != EXPECTED_REFERENCE_COMMIT:
        raise ValueError(
            "公式参照実装のcommitが"
            "固定値と一致しません: "
            f"期待={EXPECTED_REFERENCE_COMMIT}, "
            f"実際={reference_commit}"
        )

    (
        dataset_class,
        evaluate_tracknet,
        get_model,
    ) = import_official_modules(
        args.reference_root
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_parameters = checkpoint[
        "param_dict"
    ]

    model_name = checkpoint_parameters[
        "model_name"
    ]
    sequence_length = checkpoint_parameters[
        "seq_len"
    ]
    background_mode = checkpoint_parameters[
        "bg_mode"
    ]
    tolerance = checkpoint_parameters[
        "tolerance"
    ]

    if model_name != "TrackNet":
        raise ValueError(
            "チェックポイントがTrackNetでは"
            f"ありません: {model_name}"
        )

    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            "系列長が期待値と一致しません: "
            f"期待={EXPECTED_SEQUENCE_LENGTH}, "
            f"実際={sequence_length}"
        )

    if background_mode != EXPECTED_BACKGROUND_MODE:
        raise ValueError(
            "背景モードが期待値と一致しません: "
            f"期待={EXPECTED_BACKGROUND_MODE}, "
            f"実際={background_mode}"
        )

    if tolerance != EXPECTED_TOLERANCE:
        raise ValueError(
            "座標許容距離が期待値と一致しません: "
            f"期待={EXPECTED_TOLERANCE}, "
            f"実際={tolerance}"
        )

    validation_dataset = dataset_class(
        root_dir=str(args.dataset_root),
        split="val",
        seq_len=sequence_length,
        sliding_step=sequence_length,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model = get_model(
        model_name,
        sequence_length,
        background_mode,
    )

    load_result = model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model = model.to(device)

    evaluation_parameters = {
        "verbose": args.verbose,
        "tolerance": tolerance,
    }

    checkpoint_hash = calculate_sha256(
        args.checkpoint
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    validation_loss, raw_metrics = (
        evaluate_tracknet(
            model,
            validation_loader,
            evaluation_parameters,
        )
    )

    torch.cuda.synchronize()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    metrics = normalize_metrics(
        raw_metrics
    )

    classification_total = sum(
        metrics[key]
        for key in (
            "TP",
            "TN",
            "FP1",
            "FP2",
            "FN",
        )
    )

    expected_frame_count = (
        len(validation_dataset)
        * sequence_length
    )

    if classification_total != expected_frame_count:
        raise ValueError(
            "評価分類数が期待フレーム数と"
            "一致しません: "
            f"分類数={classification_total}, "
            f"期待={expected_frame_count}"
        )

    if not torch.isfinite(
        torch.tensor(validation_loss)
    ):
        raise ValueError(
            "val lossにNaNまたは"
            "無限値があります"
        )

    peak_allocated_mib = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    result = {
        "schema_version": 1,
        "evaluation_name": (
            "tracknet_pilot_v2_"
            "pretrained_baseline"
        ),
        "model": {
            "model_name": model_name,
            "sequence_length": (
                sequence_length
            ),
            "background_mode": (
                background_mode
            ),
            "heatmap_threshold": (
                OFFICIAL_HEATMAP_THRESHOLD
            ),
            "tolerance": tolerance,
        },
        "data": {
            "dataset_root": str(
                args.dataset_root
            ),
            "split": "val",
            "sequence_count": len(
                validation_dataset
            ),
            "evaluated_frame_count": (
                classification_total
            ),
            "batch_size": args.batch_size,
        },
        "source": {
            "reference_root": str(
                args.reference_root
            ),
            "reference_commit": (
                reference_commit
            ),
            "checkpoint_path": str(
                args.checkpoint
            ),
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "missing_keys": len(
                load_result.missing_keys
            ),
            "unexpected_keys": len(
                load_result.unexpected_keys
            ),
        },
        "environment": {
            "torch_version": (
                torch.__version__
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "gpu": (
                torch.cuda.get_device_name(
                    device
                )
            ),
            "seed": args.seed,
        },
        "result": {
            "validation_loss": (
                float(validation_loss)
            ),
            **metrics,
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "peak_allocated_vram_mib": (
                peak_allocated_mib
            ),
            "peak_reserved_vram_mib": (
                peak_reserved_mib
            ),
        },
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_json.open(
        "x",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            result,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    print("TrackNetV3公開重みのval評価が完了しました")
    print(f"デバイス: {device}")
    print(
        "GPU: "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(
        "参照commit: "
        f"{reference_commit}"
    )
    print(
        "チェックポイントSHA-256: "
        f"{checkpoint_hash}"
    )
    print(
        "重み読込missing keys: "
        f"{len(load_result.missing_keys)}"
    )
    print(
        "重み読込unexpected keys: "
        f"{len(load_result.unexpected_keys)}"
    )
    print(
        "val系列数: "
        f"{len(validation_dataset)}"
    )
    print(
        "評価フレーム数: "
        f"{classification_total}"
    )
    print(
        "val loss: "
        f"{validation_loss:.8f}"
    )

    for metric_name in (
        "TP",
        "TN",
        "FP1",
        "FP2",
        "FN",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "miss_rate",
    ):
        print(
            f"{metric_name}: "
            f"{metrics[metric_name]}"
        )

    print(
        "評価時間: "
        f"{elapsed_seconds:.4f}秒"
    )
    print(
        "ピーク割当VRAM: "
        f"{peak_allocated_mib:.1f} MiB"
    )
    print(
        "ピーク予約VRAM: "
        f"{peak_reserved_mib:.1f} MiB"
    )
    print(
        "結果保存先: "
        f"{args.output_json}"
    )


if __name__ == "__main__":
    main()