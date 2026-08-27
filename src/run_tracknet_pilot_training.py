import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracknet_training"
    / "pilot_v2_transfer_lr1e-4_bs2_seed13_fresh_dataset"
)

EXPECTED_REFERENCE_COMMIT = (
    "77c123ad4dd449b7d275f16cc43f316ba5b54042"
)

EXPECTED_MODEL_NAME = "TrackNet"
EXPECTED_SEQUENCE_LENGTH = 8
EXPECTED_BACKGROUND_MODE = "concat"
EXPECTED_TOLERANCE = 4
OFFICIAL_HEATMAP_THRESHOLD = 0.5
EXPECTED_VALIDATION_VISIBLE_FRAMES = 118
EXPECTED_VALIDATION_INVISIBLE_FRAMES = 2


def parse_args() -> argparse.Namespace:
    """パイロット追加学習の実行条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "公開重みからTrackNetV3を"
            "バレーボール用データで3 epoch追加学習する"
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
        help="転移学習の初期値にする公開重み",
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
        "--batch-size",
        type=int,
        default=2,
        help="trainとvalのbatch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam optimizerの固定学習率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="乱数シード",
    )
    parser.add_argument(
        "--verbose-evaluation",
        action="store_true",
        help="公式val評価の進捗バーを表示する",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """入力パス、数値、出力先を検証する。"""
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

    if args.output_dir.exists():
        raise FileExistsError(
            "学習結果の上書きを防ぐため停止します: "
            f"{args.output_dir}"
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochsには1以上を指定してください"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-sizeには1以上を指定してください"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rateには正の値を"
            "指定してください"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを使用できません"
        )


def import_official_modules(
    reference_root: Path,
):
    """固定した公式実装から必要な機能を読み込む。"""
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
    from utils.metric import WBCELoss

    return (
        Shuttlecock_Trajectory_Dataset,
        eval_tracknet,
        get_model,
        WBCELoss,
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


def get_git_commit(
    repository_root: Path,
) -> str:
    """指定したGitリポジトリのcommitを取得する。"""
    completed_process = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
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
    """公式評価結果をJSON保存可能な型へ変換する。"""
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


def write_json_atomic(
    output_path: Path,
    value,
) -> None:
    """一時ファイルを経由してJSONを更新する。"""
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            value,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    temporary_path.replace(
        output_path
    )


def write_history_csv(
    output_path: Path,
    history: list[dict],
) -> None:
    """epoch履歴を表形式のCSVへ保存する。"""
    fieldnames = [
        "epoch",
        "phase",
        "learning_rate",
        "train_loss",
        "train_sequence_count",
        "train_seconds",
        "validation_loss",
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
        "evaluation_seconds",
        "peak_allocated_vram_mib",
        "peak_reserved_vram_mib",
        "checkpoint_file",
        "checkpoint_sha256",
    ]

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)

    temporary_path.replace(
        output_path
    )

def create_dataset_and_loader(
    dataset_class,
    dataset_root: Path,
    split: str,
    sequence_length: int,
    background_mode: str,
    batch_size: int,
    shuffle: bool,
    shuffle_seed: int,
):
    """破壊的変更を持ち越さないDatasetを生成する。"""
    if split == "train":
        sliding_step = 1
    elif split == "val":
        sliding_step = sequence_length
    else:
        raise ValueError(
            "未対応のsplitです: "
            f"{split}"
        )

    dataset = dataset_class(
        root_dir=str(dataset_root),
        split=split,
        seq_len=sequence_length,
        sliding_step=sliding_step,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            shuffle_seed
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )

    return dataset, data_loader

def check_gradients_are_finite(
    model: torch.nn.Module,
) -> None:
    """全パラメータの勾配が有限値か確認する。"""
    for parameter_name, parameter in (
        model.named_parameters()
    ):
        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all().item():
            raise ValueError(
                "勾配にNaNまたは無限値があります: "
                f"{parameter_name}"
            )


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    weighted_bce_loss,
    device: torch.device,
    epoch: int,
) -> dict:
    """全train系列を使用して1 epoch学習する。"""
    model.train()

    weighted_loss_sum = 0.0
    processed_sequence_count = 0
    batch_count = len(data_loader)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        (
            _,
            model_input,
            target_heatmaps,
            _,
            _,
        ) = batch

        current_batch_size = (
            model_input.shape[0]
        )

        model_input = (
            model_input
            .float()
            .to(
                device,
                non_blocking=True,
            )
        )
        target_heatmaps = (
            target_heatmaps
            .float()
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            model_input
        )

        if (
            predictions.shape
            != target_heatmaps.shape
        ):
            raise ValueError(
                "予測shapeと正解shapeが"
                "一致しません: "
                f"予測={tuple(predictions.shape)}, "
                f"正解={tuple(target_heatmaps.shape)}"
            )

        if not torch.isfinite(
            predictions
        ).all().item():
            raise ValueError(
                "予測にNaNまたは無限値があります"
            )

        loss = weighted_bce_loss(
            predictions,
            target_heatmaps,
        )

        if not torch.isfinite(
            loss
        ).item():
            raise ValueError(
                "train lossにNaNまたは"
                "無限値があります"
            )

        loss.backward()

        check_gradients_are_finite(
            model
        )

        optimizer.step()

        weighted_loss_sum += (
            loss.item()
            * current_batch_size
        )
        processed_sequence_count += (
            current_batch_size
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == batch_count
        ):
            print(
                f"Epoch {epoch}: "
                f"batch {batch_index}/{batch_count}, "
                f"loss={loss.item():.8f}"
            )

    torch.cuda.synchronize()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    if processed_sequence_count <= 0:
        raise RuntimeError(
            "train系列を1件も処理できませんでした"
        )

    average_loss = (
        weighted_loss_sum
        / processed_sequence_count
    )

    return {
        "loss": average_loss,
        "processed_sequence_count": (
            processed_sequence_count
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024**2
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024**2
        ),
    }


def evaluate_validation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    evaluate_tracknet,
    tolerance: int,
    verbose: bool,
    expected_frame_count: int,
) -> dict:
    """公式評価関数でvalを評価する。"""
    evaluation_parameters = {
        "verbose": verbose,
        "tolerance": tolerance,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    validation_loss, raw_metrics = (
        evaluate_tracknet(
            model,
            data_loader,
            evaluation_parameters,
        )
    )

    torch.cuda.synchronize()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    if not torch.isfinite(
        torch.tensor(validation_loss)
    ).item():
        raise ValueError(
            "val lossにNaNまたは"
            "無限値があります"
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

    if classification_total != expected_frame_count:
        raise ValueError(
            "評価分類数が期待フレーム数と"
            "一致しません: "
            f"分類数={classification_total}, "
            f"期待={expected_frame_count}"
        )

    visible_classification_total = (
        metrics["TP"]
        + metrics["FP1"]
        + metrics["FN"]
    )
    invisible_classification_total = (
        metrics["TN"]
        + metrics["FP2"]
    )

    if (
        visible_classification_total
        != EXPECTED_VALIDATION_VISIBLE_FRAMES
    ):
        raise ValueError(
            "valのボールあり分類数が"
            "期待値と一致しません: "
            f"実際={visible_classification_total}, "
            "期待="
            f"{EXPECTED_VALIDATION_VISIBLE_FRAMES}"
        )

    if (
        invisible_classification_total
        != EXPECTED_VALIDATION_INVISIBLE_FRAMES
    ):
        raise ValueError(
            "valのボールなし分類数が"
            "期待値と一致しません: "
            f"実際={invisible_classification_total}, "
            "期待="
            f"{EXPECTED_VALIDATION_INVISIBLE_FRAMES}"
        )

    return {
        "validation_loss": (
            float(validation_loss)
        ),
        **metrics,
        "evaluation_seconds": (
            elapsed_seconds
        ),
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated()
            / 1024**2
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved()
            / 1024**2
        ),
    }


def save_epoch_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    epoch: int,
    validation_result: dict,
    saved_parameters: dict,
    source_checkpoint_hash: str,
) -> str:
    """epoch終了時のモデル重みを保存する。"""
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "param_dict": saved_parameters,
        "validation_result": (
            validation_result
        ),
        "source_checkpoint_sha256": (
            source_checkpoint_hash
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )

    return calculate_sha256(
        output_path
    )


def create_history_row(
    epoch: int,
    phase: str,
    learning_rate: float,
    training_result: dict | None,
    validation_result: dict,
    checkpoint_file: str | None,
    checkpoint_hash: str | None,
) -> dict:
    """baselineまたはepoch結果を1行へまとめる。"""
    return {
        "epoch": epoch,
        "phase": phase,
        "learning_rate": learning_rate,
        "train_loss": (
            None
            if training_result is None
            else training_result["loss"]
        ),
        "train_sequence_count": (
            None
            if training_result is None
            else training_result[
                "processed_sequence_count"
            ]
        ),
        "train_seconds": (
            None
            if training_result is None
            else training_result[
                "elapsed_seconds"
            ]
        ),
        "validation_loss": (
            validation_result[
                "validation_loss"
            ]
        ),
        "TP": validation_result["TP"],
        "TN": validation_result["TN"],
        "FP1": validation_result["FP1"],
        "FP2": validation_result["FP2"],
        "FN": validation_result["FN"],
        "accuracy": (
            validation_result["accuracy"]
        ),
        "precision": (
            validation_result["precision"]
        ),
        "recall": (
            validation_result["recall"]
        ),
        "f1": validation_result["f1"],
        "miss_rate": (
            validation_result["miss_rate"]
        ),
        "evaluation_seconds": (
            validation_result[
                "evaluation_seconds"
            ]
        ),
        "peak_allocated_vram_mib": max(
            (
                0.0
                if training_result is None
                else training_result[
                    "peak_allocated_vram_mib"
                ]
            ),
            validation_result[
                "peak_allocated_vram_mib"
            ],
        ),
        "peak_reserved_vram_mib": max(
            (
                0.0
                if training_result is None
                else training_result[
                    "peak_reserved_vram_mib"
                ]
            ),
            validation_result[
                "peak_reserved_vram_mib"
            ],
        ),
        "checkpoint_file": (
            checkpoint_file
        ),
        "checkpoint_sha256": (
            checkpoint_hash
        ),
    }


def print_validation_result(
    label: str,
    result: dict,
) -> None:
    """val評価結果を比較しやすい形式で表示する。"""
    print(f"=== {label} ===")
    print(
        "val loss: "
        f"{result['validation_loss']:.8f}"
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
            f"{result[metric_name]}"
        )

    print(
        "評価時間: "
        f"{result['evaluation_seconds']:.4f}秒"
    )


def main() -> None:
    """公開重みから3 epoch追加学習する。"""
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
    args.output_dir = (
        args.output_dir.resolve()
    )

    validate_args(args)

    (
        dataset_class,
        evaluate_tracknet,
        get_model,
        weighted_bce_loss,
    ) = import_official_modules(
        args.reference_root
    )

    volley_scope_commit = get_git_commit(
        PROJECT_ROOT
    )
    reference_commit = get_git_commit(
        args.reference_root
    )

    if reference_commit != EXPECTED_REFERENCE_COMMIT:
        raise ValueError(
            "公式参照実装のcommitが"
            "固定値と一致しません: "
            f"期待={EXPECTED_REFERENCE_COMMIT}, "
            f"実際={reference_commit}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    source_checkpoint_hash = (
        calculate_sha256(
            args.checkpoint
        )
    )

    training_script_path = Path(
        __file__
    ).resolve()
    training_script_hash = (
        calculate_sha256(
            training_script_path
        )
    )

    source_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_parameters = (
        source_checkpoint["param_dict"]
    )

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

    if model_name != EXPECTED_MODEL_NAME:
        raise ValueError(
            "モデル名が期待値と一致しません: "
            f"期待={EXPECTED_MODEL_NAME}, "
            f"実際={model_name}"
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

    train_dataset = dataset_class(
        root_dir=str(args.dataset_root),
        split="train",
        seq_len=sequence_length,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    validation_dataset = dataset_class(
        root_dir=str(args.dataset_root),
        split="val",
        seq_len=sequence_length,
        sliding_step=sequence_length,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    model = get_model(
        model_name,
        sequence_length,
        background_mode,
    )

    load_result = model.load_state_dict(
        source_checkpoint["model"],
        strict=True,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    saved_parameters = dict(
        checkpoint_parameters
    )
    saved_parameters.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": (
                args.learning_rate
            ),
            "optim": "Adam",
            "lr_scheduler": "",
            "alpha": -1,
            "frame_alpha": -1,
            "seed": args.seed,
            "dataset_root": str(
                args.dataset_root
            ),
            "reference_commit": (
                reference_commit
            ),
            "volley_scope_commit": (
                volley_scope_commit
            ),
            "drop_last": False,
        }
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoints_dir = (
        args.output_dir
        / "checkpoints"
    )
    checkpoints_dir.mkdir()

    configuration = {
        "schema_version": 1,
        "experiment_name": (
            "tracknet_pilot_v2_transfer"
        ),
        "volley_scope_commit": (
            volley_scope_commit
        ),
        "training_script": str(
            training_script_path
        ),
        "training_script_sha256": (
            training_script_hash
        ),
        "reference_root": str(
            args.reference_root
        ),
        "reference_commit": (
            reference_commit
        ),
        "dataset_root": str(
            args.dataset_root
        ),
        "source_checkpoint": str(
            args.checkpoint
        ),
        "source_checkpoint_sha256": (
            source_checkpoint_hash
        ),
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": (
            args.learning_rate
        ),
        "optimizer": "Adam",
        "scheduler": None,
        "mixup_alpha": -1,
        "frame_mixup_alpha": -1,
        "drop_last": False,
        "dataset_refresh_policy": (
            "create a new Dataset before "
            "every train and validation pass"
        ),
        "official_dataset_mutates_coordinates": True,
        "seed": args.seed,
        "train_sequence_count": len(
            train_dataset
        ),
        "validation_sequence_count": len(
            validation_dataset
        ),
        "expected_validation_frames": (
            len(validation_dataset)
            * sequence_length
        ),
        "torch_version": (
            torch.__version__
        ),
        "gpu": (
            torch.cuda.get_device_name(
                device
            )
        ),
        "missing_keys": len(
            load_result.missing_keys
        ),
        "unexpected_keys": len(
            load_result.unexpected_keys
        ),
    }

    write_json_atomic(
        args.output_dir / "configuration.json",
        configuration,
    )

    history = []
    history_json_path = (
        args.output_dir / "history.json"
    )
    history_csv_path = (
        args.output_dir / "history.csv"
    )

    print("TrackNetV3パイロット追加学習を開始します")
    print(f"GPU: {configuration['gpu']}")
    print(
        "VolleyScope commit: "
        f"{volley_scope_commit}"
    )
    print(
        "学習スクリプトSHA-256: "
        f"{training_script_hash}"
    )
    print(
        "公式参照commit: "
        f"{reference_commit}"
    )
    print(
        "公開重みSHA-256: "
        f"{source_checkpoint_hash}"
    )
    print(
        "train系列数: "
        f"{len(train_dataset)}"
    )
    print(
        "val系列数: "
        f"{len(validation_dataset)}"
    )
    print(f"epoch数: {args.epochs}")
    print(f"batch size: {args.batch_size}")
    print(
        "学習率: "
        f"{args.learning_rate:.8f}"
    )
    print("mixup: 使用しない")
    print("scheduler: 使用しない")
    print("drop_last: False")
    print(
        "Dataset再生成: "
        "train・valの各走査前に実施"
    )
    expected_validation_frames = (
        len(validation_dataset)
        * sequence_length
    )
    (
        _,
        baseline_validation_loader,
    ) = create_dataset_and_loader(
        dataset_class=dataset_class,
        dataset_root=args.dataset_root,
        split="val",
        sequence_length=sequence_length,
        background_mode=background_mode,
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_seed=args.seed,
    )

    baseline_result = evaluate_validation(
        model,
        baseline_validation_loader,
        evaluate_tracknet,
        tolerance,
        args.verbose_evaluation,
        expected_validation_frames,
    )

    del baseline_validation_loader

    print_validation_result(
        "学習前baseline",
        baseline_result,
    )

    baseline_row = create_history_row(
        epoch=0,
        phase="pretrained_baseline",
        learning_rate=args.learning_rate,
        training_result=None,
        validation_result=baseline_result,
        checkpoint_file=None,
        checkpoint_hash=None,
    )
    history.append(
        baseline_row
    )

    write_json_atomic(
        history_json_path,
        history,
    )
    write_history_csv(
        history_csv_path,
        history,
    )

    best_accuracy = float("-inf")
    best_epoch = None
    best_checkpoint_path = None
    best_checkpoint_hash = None
    epoch_checkpoint_records = []

    total_start_time = perf_counter()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            f"=== Epoch {epoch}/{args.epochs} ==="
        )

        (
            epoch_train_dataset,
            epoch_train_loader,
        ) = create_dataset_and_loader(
            dataset_class=dataset_class,
            dataset_root=args.dataset_root,
            split="train",
            sequence_length=sequence_length,
            background_mode=background_mode,
            batch_size=args.batch_size,
            shuffle=True,
            shuffle_seed=(
                args.seed + epoch
            ),
        )

        training_result = train_one_epoch(
            model,
            optimizer,
            epoch_train_loader,
            weighted_bce_loss,
            device,
            epoch,
        )

        if (
            training_result[
                "processed_sequence_count"
            ]
            != len(epoch_train_dataset)
        ):
            raise ValueError(
                "処理train系列数が"
                "Dataset件数と一致しません"
            )

        del epoch_train_loader
        del epoch_train_dataset

        (
            epoch_validation_dataset,
            epoch_validation_loader,
        ) = create_dataset_and_loader(
            dataset_class=dataset_class,
            dataset_root=args.dataset_root,
            split="val",
            sequence_length=sequence_length,
            background_mode=background_mode,
            batch_size=args.batch_size,
            shuffle=False,
            shuffle_seed=(
                args.seed + epoch
            ),
        )

        validation_result = evaluate_validation(
            model,
            epoch_validation_loader,
            evaluate_tracknet,
            tolerance,
            args.verbose_evaluation,
            expected_validation_frames,
        )

        if (
            len(epoch_validation_dataset)
            * sequence_length
            != expected_validation_frames
        ):
            raise ValueError(
                "valの期待フレーム数が"
                "一致しません"
            )

        del epoch_validation_loader
        del epoch_validation_dataset

        print(
            "train loss: "
            f"{training_result['loss']:.8f}"
        )
        print(
            "処理train系列数: "
            f"{training_result['processed_sequence_count']}"
        )
        print(
            "train時間: "
            f"{training_result['elapsed_seconds']:.4f}秒"
        )

        print_validation_result(
            f"Epoch {epoch} val",
            validation_result,
        )

        checkpoint_path = (
            checkpoints_dir
            / f"epoch_{epoch:03d}.pt"
        )

        checkpoint_hash = (
            save_epoch_checkpoint(
                checkpoint_path,
                model,
                epoch,
                validation_result,
                saved_parameters,
                source_checkpoint_hash,
            )
        )

        checkpoint_record = {
            "epoch": epoch,
            "file": str(
                checkpoint_path
            ),
            "sha256": (
                checkpoint_hash
            ),
            "accuracy": (
                validation_result[
                    "accuracy"
                ]
            ),
        }
        epoch_checkpoint_records.append(
            checkpoint_record
        )

        history_row = create_history_row(
            epoch=epoch,
            phase="fine_tuned",
            learning_rate=(
                optimizer.param_groups[0]["lr"]
            ),
            training_result=training_result,
            validation_result=(
                validation_result
            ),
            checkpoint_file=str(
                checkpoint_path
            ),
            checkpoint_hash=(
                checkpoint_hash
            ),
        )
        history.append(
            history_row
        )

        write_json_atomic(
            history_json_path,
            history,
        )
        write_history_csv(
            history_csv_path,
            history,
        )

        current_accuracy = (
            validation_result["accuracy"]
        )

        if current_accuracy > best_accuracy:
            best_accuracy = (
                current_accuracy
            )
            best_epoch = epoch
            best_checkpoint_path = (
                checkpoint_path
            )
            best_checkpoint_hash = (
                checkpoint_hash
            )

    total_elapsed_seconds = (
        perf_counter() - total_start_time
    )

    final_checkpoint_record = (
        epoch_checkpoint_records[-1]
    )

    baseline_accuracy = (
        baseline_result["accuracy"]
    )

    improved_over_baseline = (
        best_accuracy
        > baseline_accuracy
    )

    if improved_over_baseline:
        overall_best_source = (
            "fine_tuned"
        )
        overall_best_accuracy = (
            best_accuracy
        )
        overall_best_checkpoint = str(
            best_checkpoint_path
        )
    else:
        overall_best_source = (
            "pretrained_baseline"
        )
        overall_best_accuracy = (
            baseline_accuracy
        )
        overall_best_checkpoint = str(
            args.checkpoint
        )

    summary = {
        "schema_version": 1,
        "experiment_name": (
            configuration[
                "experiment_name"
            ]
        ),
        "baseline": {
            "accuracy": (
                baseline_accuracy
            ),
            "TP": baseline_result["TP"],
            "TN": baseline_result["TN"],
            "FP1": baseline_result["FP1"],
            "FP2": baseline_result["FP2"],
            "FN": baseline_result["FN"],
            "validation_loss": (
                baseline_result[
                    "validation_loss"
                ]
            ),
        },
        "best_fine_tuned": {
            "epoch": best_epoch,
            "accuracy": best_accuracy,
            "checkpoint": str(
                best_checkpoint_path
            ),
            "checkpoint_sha256": (
                best_checkpoint_hash
            ),
        },
        "final_fine_tuned": {
            "epoch": (
                final_checkpoint_record[
                    "epoch"
                ]
            ),
            "accuracy": (
                final_checkpoint_record[
                    "accuracy"
                ]
            ),
            "checkpoint": (
                final_checkpoint_record[
                    "file"
                ]
            ),
            "checkpoint_sha256": (
                final_checkpoint_record[
                    "sha256"
                ]
            ),
        },
        "improved_over_baseline": (
            improved_over_baseline
        ),
        "overall_best_source": (
            overall_best_source
        ),
        "overall_best_accuracy": (
            overall_best_accuracy
        ),
        "overall_best_checkpoint": (
            overall_best_checkpoint
        ),
        "total_training_and_validation_seconds": (
            total_elapsed_seconds
        ),
        "epoch_checkpoints": (
            epoch_checkpoint_records
        ),
    }

    write_json_atomic(
        args.output_dir / "summary.json",
        summary,
    )

    print("=== 学習完了 ===")
    print(
        "baseline accuracy: "
        f"{baseline_accuracy:.6f}"
    )
    print(
        "最良追加学習epoch: "
        f"{best_epoch}"
    )
    print(
        "最良追加学習accuracy: "
        f"{best_accuracy:.6f}"
    )
    print(
        "baselineより改善: "
        f"{improved_over_baseline}"
    )
    print(
        "全体の最良source: "
        f"{overall_best_source}"
    )
    print(
        "学習・評価合計時間: "
        f"{total_elapsed_seconds:.2f}秒"
    )
    print(
        "出力先: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()