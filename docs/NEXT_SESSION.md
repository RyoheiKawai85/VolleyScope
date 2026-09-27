# 次回作業：第四外部testによる固定2モデルの最終未知環境評価

## 現在地

TrackNetV3とYOLO11nの多様化学習、val選定、第三外部testの診断が完了した。

第四外部testはまだ推論へ使用していない。

## 固定TrackNetV3

- checkpoint：多様化学習Epoch 7
- threshold：0.28
- SHA-256：`23ADAEAFC5B0159815B99F466904132BA9541E520CF09BAF9D0842E79C131CFE`

## 固定YOLO11n

- checkpoint：多様化学習Epoch 6の`epoch5.pt`
- confidence：0.15
- imgsz：1280
- class ID：0
- SHA-256：`C99C1C5D0D2EAFA04533DED473D844A0E70D4F798F4910E1983B6233FF354A06`

## 第四外部test

- 元動画：未使用match08
- 高校生の試合
- 未使用会場
- エンドライン側画角
- 約60fps
- 第三外部testよりモーションブラーが少ない
- 評価画像：120枚
- 正例：103枚
- 負例：17枚
- 判別不能除外：0枚
- 正式保存先：`data/annotations/external_evaluation_004_final`
- Label Studio ZIP SHA-256：`2ED8BD400A3179559C467D0B1A29745FC0F02409774F2E5B4C7BBD601B7A31E4`

## 次回の作業順序

1. manifest・画像・ラベル対応を再確認する
2. YOLOラベルをTrackNetV3用CSVへ変換する
3. 正解座標を再描画して確認する
4. 固定TrackNetV3をthreshold 0.28で一度だけ評価する
5. 固定YOLO11nをconfidence 0.15、imgsz 1280で一度だけ評価する
6. 共通中心距離基準で比較する
7. TPの重なりと補完関係を集計する
8. 代表FP・FNを可視化する
9. 第三外部testとの差を解釈する
10. 結果を文書へ記録する

## 完了条件

- 両checkpointのSHA-256が固定値と一致する
- 第四外部testを一度だけ評価する
- 第四外部testを使ってモデルやしきい値を再選定しない
- 両モデルを共通中心距離基準で比較する
- 次の変更前に新しい未使用testを確保する

## 現時点で行わないこと

- 第四外部testを使った再調整
- TrackNetV3とYOLOの単純統合
- 客席や画面下部の一律除外
- 第三外部testだけへ合わせた再学習