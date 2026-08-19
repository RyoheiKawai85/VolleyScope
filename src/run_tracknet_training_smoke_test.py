import argparse
import math
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


def parse_args() -> argparse.Namespace:
    """1バッチ学習の実行条件を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "公開重みとVolleyScopeデータを使い、"
            "TrackNetV3の1バッチ学習を検証する"
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
        "--batch-size",
        type=int,
        default=1,
        help="スモークテストのbatch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="新しいAdam optimizerの学習率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="乱数シード",
    )
    return parser.parse_args()


def import_official_modules(
    reference_root: Path,
):
    """固定した公式実装から必要な機能を読み込む。"""
    if not reference_root.is_dir():
        raise FileNotFoundError(
            "TrackNetV3参照リポジトリが"
            f"見つかりません: {reference_root}"
        )

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
    from utils.general import get_model
    from utils.metric import WBCELoss

    return (
        Shuttlecock_Trajectory_Dataset,
        get_model,
        WBCELoss,
    )


def validate_args(
    args: argparse.Namespace,
) -> None:
    """パスと数値引数を検証する。"""
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            "公式互換データセットが"
            f"見つかりません: {args.dataset_root}"
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

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rateには正の値を"
            "指定してください"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを使用できません"
        )


def calculate_gradient_norm(
    model: torch.nn.Module,
) -> tuple[float, bool]:
    """全パラメータの勾配ノルムと有限性を確認する。"""
    squared_sum = 0.0
    all_finite = True

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            all_finite = False

        squared_sum += (
            gradient
            .float()
            .pow(2)
            .sum()
            .item()
        )

    return math.sqrt(squared_sum), all_finite


def select_updated_parameter(
    model: torch.nn.Module,
) -> tuple[
    str,
    torch.nn.Parameter,
    torch.Tensor,
]:
    """非ゼロ勾配を持つパラメータを1つ選ぶ。"""
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if torch.count_nonzero(
            parameter.grad
        ).item() == 0:
            continue

        return (
            name,
            parameter,
            parameter.detach().clone(),
        )

    raise RuntimeError(
        "非ゼロ勾配を持つパラメータが"
        "見つかりません"
    )


def main() -> None:
    """公開重みから1バッチだけ転移学習する。"""
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

    validate_args(args)

    (
        dataset_class,
        get_model,
        weighted_bce_loss,
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

    if model_name != "TrackNet":
        raise ValueError(
            "チェックポイントがTrackNetでは"
            f"ありません: {model_name}"
        )

    if sequence_length != 8:
        raise ValueError(
            "系列長が期待値8ではありません: "
            f"{sequence_length}"
        )

    if background_mode != "concat":
        raise ValueError(
            "背景モードが期待値concatでは"
            f"ありません: {background_mode}"
        )

    dataset = dataset_class(
        root_dir=str(args.dataset_root),
        split="train",
        seq_len=sequence_length,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode=background_mode,
    )

    data_loader = DataLoader(
        dataset,
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
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    batch = next(iter(data_loader))

    (
        _,
        model_input,
        target_heatmaps,
        _,
        _,
    ) = batch

    print(f"デバイス: {device}")
    print(
        "GPU: "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(f"データセット系列数: {len(dataset)}")
    print(f"batch size: {args.batch_size}")
    print(
        "学習率: "
        f"{args.learning_rate:.8f}"
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
        "転送前入力: "
        f"shape={tuple(model_input.shape)}, "
        f"dtype={model_input.dtype}"
    )
    print(
        "転送前正解: "
        f"shape={tuple(target_heatmaps.shape)}, "
        f"dtype={target_heatmaps.dtype}"
    )

    model_input = model_input.float().to(
        device,
        non_blocking=True,
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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    predictions = model(model_input)

    if predictions.shape != target_heatmaps.shape:
        raise ValueError(
            "予測shapeと正解shapeが"
            "一致しません: "
            f"予測={tuple(predictions.shape)}, "
            f"正解={tuple(target_heatmaps.shape)}"
        )

    if not torch.isfinite(predictions).all():
        raise ValueError(
            "予測にNaNまたは無限値があります"
        )

    loss = weighted_bce_loss(
        predictions,
        target_heatmaps,
    )

    if not torch.isfinite(loss):
        raise ValueError(
            "lossが有限値ではありません"
        )

    loss.backward()

    (
        gradient_norm,
        gradients_are_finite,
    ) = calculate_gradient_norm(model)

    if not gradients_are_finite:
        raise ValueError(
            "勾配にNaNまたは無限値があります"
        )

    (
        updated_parameter_name,
        updated_parameter,
        parameter_before_update,
    ) = select_updated_parameter(model)

    optimizer.step()

    torch.cuda.synchronize()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    parameter_update_max = (
        updated_parameter.detach()
        - parameter_before_update
    ).abs().max().item()

    peak_allocated_mib = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    print(
        "GPU転送後入力: "
        f"shape={tuple(model_input.shape)}, "
        f"dtype={model_input.dtype}"
    )
    print(
        "予測: "
        f"shape={tuple(predictions.shape)}, "
        f"dtype={predictions.dtype}"
    )
    print(f"loss: {loss.item():.8f}")
    print(
        "勾配がすべて有限値: "
        f"{gradients_are_finite}"
    )
    print(
        "勾配L2ノルム: "
        f"{gradient_norm:.8f}"
    )
    print(
        "更新確認パラメータ: "
        f"{updated_parameter_name}"
    )
    print(
        "最大パラメータ更新量: "
        f"{parameter_update_max:.10f}"
    )
    print(
        "1バッチ学習時間: "
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
        "チェックポイントは保存していません"
    )
    print(
        "TrackNetV3の1バッチ学習に"
        "成功しました"
    )


if __name__ == "__main__":
    main()