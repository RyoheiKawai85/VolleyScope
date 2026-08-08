import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKNET_ROOT = PROJECT_ROOT.parent / "TrackNetV3-reference"
DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "clips"
    / "ball_challenge_002.mp4"
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_TRACKNET_ROOT
    / "ckpts"
    / "TrackNet_best.pt"
)


def parse_args() -> argparse.Namespace:
    """スモークテストで使用するパスと背景サンプル数を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "TrackNetV3公式実装の前処理とモデルを接続し、"
            "実動画1系列の入出力shapeを確認する"
        )
    )
    parser.add_argument(
        "--tracknet-root",
        type=Path,
        default=DEFAULT_TRACKNET_ROOT,
        help="TrackNetV3公式コードのルート",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="入力動画",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="TrackNetのチェックポイント",
    )
    parser.add_argument(
        "--max-sample-num",
        type=int,
        default=100,
        help="背景中央値の生成に使う最大フレーム数",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help=(
            "非重複8フレーム系列の番号。"
            "0はフレーム0〜7、4は32〜39"
        ),
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """必要なファイルと設定値を推論前に検証する。"""
    if not args.tracknet_root.is_dir():
        raise FileNotFoundError(
            f"TrackNetV3公式コードが見つかりません: "
            f"{args.tracknet_root}"
        )

    if not args.video.is_file():
        raise FileNotFoundError(
            f"入力動画が見つかりません: {args.video}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: "
            f"{args.checkpoint}"
        )

    if args.max_sample_num <= 0:
        raise ValueError(
            "--max-sample-numには1以上を指定してください"
        )
    if args.sequence_index < 0:
        raise ValueError(
            "--sequence-indexには0以上を指定してください"
        )



def import_tracknet_modules(tracknet_root: Path):
    """参照用公式リポジトリから必要な処理だけを読み込む。"""
    tracknet_root_text = str(tracknet_root.resolve())

    if tracknet_root_text not in sys.path:
        sys.path.insert(0, tracknet_root_text)

    from dataset import Video_IterableDataset
    from test import predict_location
    from utils.general import get_model, to_img

    return (
        Video_IterableDataset,
        get_model,
        predict_location,
        to_img,
    )


def main() -> None:
    """実動画1系列を前処理し、TrackNetで推論する。"""
    args = parse_args()
    validate_inputs(args)

    (
    VideoIterableDataset,
    get_model,
    predict_location,
    to_img,
) = import_tracknet_modules(args.tracknet_root)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"デバイス: {device}")
    print(f"入力動画: {args.video}")
    print(f"チェックポイント: {args.checkpoint}")
    print(f"背景サンプル上限: {args.max_sample_num}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    parameters = checkpoint["param_dict"]
    sequence_length = parameters["seq_len"]
    background_mode = parameters["bg_mode"]

    print(f"系列長: {sequence_length}")
    print(f"背景モード: {background_mode}")

    model = get_model(
        "TrackNet",
        sequence_length,
        background_mode,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    dataset = VideoIterableDataset(
        video_file=str(args.video),
        seq_len=sequence_length,
        sliding_step=sequence_length,
        bg_mode=background_mode,
        max_sample_num=args.max_sample_num,
    )

    dataset_iterator = iter(dataset)

    try:
        for _ in range(args.sequence_index + 1):
            frame_indices, frames = next(
                dataset_iterator
            )
    except StopIteration as error:
        raise ValueError(
            "指定した系列番号が動画範囲を超えています: "
            f"{args.sequence_index}"
        ) from error

    input_tensor = (
        torch.from_numpy(frames)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = perf_counter()

    with torch.inference_mode():
        output = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = perf_counter() - start_time

    print(f"フレーム番号: {frame_indices[:, 1].tolist()}")
    print(f"入力shape: {tuple(input_tensor.shape)}")
    print(f"入力dtype: {input_tensor.dtype}")
    print(
        "入力値範囲: "
        f"{input_tensor.min().item():.4f}〜"
        f"{input_tensor.max().item():.4f}"
    )
    print(f"出力shape: {tuple(output.shape)}")
    output_is_finite = torch.isfinite(output).all().item()

    print(f"出力がすべて有限値: {output_is_finite}")
    print(
        "出力値範囲: "
        f"{output.min().item():.4f}〜"
        f"{output.max().item():.4f}"
    )

    binary_heatmaps = (
        output > 0.5
    ).detach().cpu().numpy()

    print(
        "Frame  Visibility  X_original  Y_original  "
        "Peak_X  Peak_Y  Heatmap_max"
    )

    for sequence_index in range(sequence_length):
        frame_index = int(
            frame_indices[sequence_index][1]
        )
        heatmap = binary_heatmaps[0][sequence_index]
        heatmap_image = to_img(heatmap)

        x, y, width, height = predict_location(
            heatmap_image
        )

        center_x = int(x + width / 2)
        center_y = int(y + height / 2)

        visibility = (
            0
            if center_x == 0 and center_y == 0
            else 1
        )

        original_x = int(
            center_x * dataset.w_scaler
        )
        original_y = int(
            center_y * dataset.h_scaler
        )

        frame_heatmap = output[0][sequence_index]
        heatmap_max = frame_heatmap.max().item()

        flat_peak_index = (
            frame_heatmap.argmax().item()
        )
        heatmap_width = frame_heatmap.shape[1]

        peak_y, peak_x = divmod(
            flat_peak_index,
            heatmap_width,
        )

        original_peak_x = int(
            peak_x * dataset.w_scaler
        )
        original_peak_y = int(
            peak_y * dataset.h_scaler
        )

        print(
            f"{frame_index:>5}"
            f"{visibility:>12}"
            f"{original_x:>12}"
            f"{original_y:>12}"
            f"{original_peak_x:>8}"
            f"{original_peak_y:>8}"
            f"{heatmap_max:>13.4f}"
        )
    print(f"推論時間: {elapsed_seconds:.4f}秒")

    if device.type == "cuda":
        peak_vram_mib = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
        print(f"ピーク割当VRAM: {peak_vram_mib:.1f} MiB")


if __name__ == "__main__":
    main()