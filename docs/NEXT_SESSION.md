# 次回作業：TrackNetV3公式互換データセットの作成

## 現在地

TrackNetV3パイロットのtrainとvalが完成した。

### train

- 元フレーム：2977〜3104
- 総数：128枚
- 可視：126枚
- 不可視：2枚
- 不可視フレーム：2984、2985
- YOLO完成版、TrackNetV3用CSV、座標QC：完了

### val

- 元フレーム：62〜181
- 総数：120枚
- 可視：118枚
- 不可視：2枚
- 不可視フレーム：122、133
- YOLO完成版、TrackNetV3用CSV、座標QC：完了

val完成版ZIPのSHA-256は`6BB57A0359479E6FFED742FF4600A860B0FCD64EC697AC446D2892AE67745189`である。

### test

既存の`evaluation_001_final`を使用する。

## 公式実装から確認した仕様

- データルート既定値：`data`
- 試合フォルダ：`match{番号}`
- ラリーフォルダ：`frame/{rally_id}`
- CSV：`csv/{rally_id}_ball.csv`
- 画像形式：PNG
- 最初の画像：`0.png`
- CSVの`Frame`値と画像名が対応する
- trainのsliding step：1
- valのsliding step：8
- 背景中央値：`median.npz`の`median`キー
- 公式前処理ではtrainの試合中央値をvalへコピーする

## 次に検証する仮説

元画像と完成CSVを公式形式へコピー・変換すれば、公式コードを変更せずにtrain 121系列とval 15系列を読み出せる。

## 実装前に確認する設計

出力先は`data/tracknet_official_pilot_v2`とする。

想定構造は次のとおりである。

    tracknet_official_pilot_v2/
      train/
        match1/
          csv/
            1_ball.csv
          frame/
            1/
              0.png〜127.png
          median.npz
      val/
        match1/
          csv/
            1_ball.csv
          frame/
            1/
              0.png〜119.png
          median.npz
      frame_mapping.csv

番号は次のように変換する。

| split | 元フレーム | 公式用フレーム |
|---|---:|---:|
| train | 2977〜3104 | 0〜127 |
| val | 62〜181 | 0〜119 |

## 次回の作業順序

1. 整形方針を再確認する
2. `src/prepare_tracknet_official_dataset.py`を新規作成する
3. train・valの完成CSVとPNGを検証する
4. 0始まりの画像とCSVを一時出力先へ生成する
5. `frame_mapping.csv`を生成する
6. train画像から中央値背景を生成する
7. 同じ`median.npz`をvalへコピーする
8. 完成後に一時出力先を正式な出力先へ確定する
9. 件数、番号、CSV、画像、中央値を検証する
10. 公式Datasetから系列を読み出すスモークテストを行う

## 新規スクリプトの安全方針

- 元画像と完成ラベルを変更しない
- 既存出力先がある場合は上書きせず停止する
- 途中失敗時は完成データとして扱わない
- 元番号とローカル番号の対応を必ず保存する
- 公式参照リポジトリを変更しない
- メモリ不足を避けるため中央値を分割計算する

## 次回の完了条件

- train画像が0〜127の128枚である
- val画像が0〜119の120枚である
- train CSVが0〜127の128行である
- val CSVが0〜119の120行である
- 可視・不可視数が変換前と一致する
- `frame_mapping.csv`から元番号へ戻せる
- trainとvalの`median.npz`が一致する
- 公式Datasetがtrain 121系列、val 15系列を生成する
- 最初の系列shapeと可視ラベルが期待値に一致する

## 解釈上の注意

0始まりへの変換は公式互換性のためであり、元動画上のフレーム番号を置き換えるものではない。

121個のtrain系列は大きく重複しているため、独立した121種類の映像サンプルとは解釈しない。

trainとvalは同じ撮影環境のパイロットであり、未知環境への汎化性能は評価できない。