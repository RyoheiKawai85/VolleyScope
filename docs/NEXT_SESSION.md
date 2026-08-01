# 次回作業：TrackNetV3導入可能性の調査

## 現在地

最新コミットは`3498d24`。

事前学習済みYOLO11nを入力サイズ1280で評価した。

- TP：32
- FN：72
- Precision：0.914
- Recall：0.308
- F1：0.460

104個の正解ボールは、1280入力換算ですべてCOCO形式のsmall基準に該当した。

小さい半分のRecallは9.6%、大きい半分は51.9%だった。

構造的遮蔽なしに限定しても、小さい半分は21.7%、大きい半分は61.4%だった。

この結果から、最初に視認可能な小物体の基本検出性能を改善する。

## モデル候補

- バレーボールデータで追加学習したYOLO
- 小物体向け検出ヘッド
- TrackNetV2・TrackNetV3
- SAHIは主軸ではなく比較候補

TrackNetV3の追跡モジュールをYOLOの比較候補とし、軌道補完用InpaintNetは基本検出性能を確認した後に扱う。

## PC移行

現在のPCにはNVIDIA GPUがなく、PyTorchはCPU版である。

ゲーミングノートPCを主開発環境にする予定。最初に以下を確認する。

```powershell
nvidia-smi

Get-CimInstance Win32_ComputerSystem |
    Select-Object TotalPhysicalMemory

Get-CimInstance Win32_VideoController |
    Select-Object Name, AdapterRAM, DriverVersion

Get-PSDrive C |
    Select-Object Used, Free

git --version

py --list-paths
```

## TrackNetV3調査項目

1. NVIDIA GPUとVRAM
2. 公式リポジトリのライセンス
3. 公開チェックポイント
4. Python・PyTorch・CUDA依存関係
5. Windowsでの実行可能性
6. 公開重みからファインチューニングできるか
7. `Frame, Visibility, X, Y`形式へのデータ変換
8. YOLOとTrackNetで共通利用できる評価指標
9. 現行PyTorchへの移植コスト
10. 学習用連続フレームのアノテーション計画

## データ移行

GitHubから取得できないものは別途移行する。

- `data/raw/`
- `data/clips/`
- `data/frames/`
- Git管理外のアノテーション
- 必要な`outputs/`

`.venv`はコピーせず、ゲーミングPCで再作成する。

現在の評価データは学習へ使用しない。

## ゲーミングPCでCodexへ送る開始指示

GitHubからVolleyScopeをクローンした後、新しいCodexタスクで次の指示を送る。

```text
このVolleyScopeプロジェクトを継続します。AGENTS.mdとdocs/NEXT_SESSION.md、README.md、docs/DEVELOPMENT_PLAN.md、docs/EXPERIMENT_LOG.mdを読んで、TrackNetV3の導入可能性調査から再開してください。
```