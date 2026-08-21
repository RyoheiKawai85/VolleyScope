# 次回作業：TrackNetV3パイロット追加学習の実装

## 現在地

TrackNetV3パイロットのtrainとvalのアノテーション、CSV変換、公式Dataset互換形式への変換が完了した。

### データセット

- 出力先：`data/tracknet_official_pilot_v2`
- train画像：128枚
- val画像：120枚
- train系列：121
- val系列：15
- 系列長：8
- 背景中央値shape：`(1080, 2340, 3)`
- 背景中央値SHA-256：`3B618FA13BF201BB9CB1C3C8AE061303FB0F52BB3E9731867A374763B0F13A0F`
- コピー画像のハッシュ不一致：0
- 一時フォルダの残留：0

### 作成済みコード

- `src/prepare_tracknet_official_dataset.py`
- `src/validate_tracknet_official_dataset.py`
- `src/run_tracknet_training_smoke_test.py`

### 1バッチ学習

batch size 1と2の両方で次を確認した。

- 公開重みを完全一致で読み込める
- 入力と正解を`float32`でGPUへ転送できる
- 順伝播できる
- lossを計算できる
- 勾配が有限値である
- パラメータが実際に更新される
- GTX 1650のVRAM内で実行できる

batch size 2のピーク予約VRAMは2912 MiBだった。

## 次に検証する仮説

バドミントン用TrackNetV3公開重みを初期値とし、バレーボール用trainデータで3 epoch追加学習することで、数値的に安定した学習を継続でき、学習前よりval指標が改善する可能性がある。

## 実験条件

最初のパイロットでは次の条件を使用する。

- 初期重み：公開`TrackNet_best.pt`
- 更新対象：全層
- optimizer：Adam
- batch size：2
- 学習率：`0.0001`
- epoch数：3
- seed：13
- train sliding step：1
- val sliding step：8
- チェックポイント：epochごととval最良時
- CUDAメモリ不足時の代替：batch size 1
- ヒートマップしきい値：`0.5`
- 座標許容距離：ヒートマップ空間で4px

## 次回の作業順序

1. 公開重みをval 120フレームで評価し、学習前の基準値を保存する
2. 学習前の公開重みをval 15系列で評価する
3. `src/run_tracknet_pilot_training.py`を作成する
4. 引数、入力パス、出力先の検証を実装する
5. 公開重みだけをモデルへ読み込む
6. Adam optimizerを新規作成する
7. 3 epochのtrainとvalを実装する
8. epochごとのlossと評価指標をCSVへ保存する
9. 最良モデルと最終モデルを保存する
10. 実行条件をJSONへ保存する
11. 学習後モデルを同じval条件で評価する
12. 学習前後の結果を比較する

## 保存する情報

- Git commit
- 公式参照実装のcommit
- Python、PyTorch、CUDA、GPU
- データセットのパスと件数
- 背景中央値のハッシュ
- checkpointの入力ハッシュ
- seed
- batch size
- 学習率
- epoch数
- train loss
- val loss
- val評価指標
- epochごとの処理時間
- ピーク割当VRAM
- ピーク予約VRAM
- 最良epoch
- 出力モデルのSHA-256

## 完了条件

- 3 epochがエラーなく終了する
- lossと勾配に`NaN`または無限大がない
- 各epochの結果がCSVへ記録される
- 学習前後を同じval条件で比較できる
- 最良モデルと最終モデルを区別して保存できる
- 実験条件を後から再現できる
- 結果から言えることと言えないことを分けて記録できる

## 解釈上の注意

train lossの低下だけでは、未学習データに対する性能向上を示せない。

valはモデル選択と開発判断に使用する。valへ合わせて設計を繰り返すほど、valに対する過適合の可能性が増える。

今回のtrainとvalは同じ元動画と撮影環境から取得している。そのため、同一環境内の追加学習経路は検証できるが、未知の試合、体育館、カメラ条件への汎化性能は評価できない。

公開重みから改善しなかった場合も、TrackNetV3構造が不適切だと直ちに結論づけない。データ量、ドメイン差、学習率、epoch数、ラベル品質、負の転移を代替説明として検討する。

YOLOとの比較は、同じテスト対象と同じ判定基準を用意した後に行う。