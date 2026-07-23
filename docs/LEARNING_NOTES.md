# VolleyScope 学習ノート

## この文書について

VolleyScopeの開発中に学んだPython、Git、機械学習開発の基礎知識を記録する。

単なるコマンド集ではなく、「何をしているのか」「なぜ必要なのか」を後から説明できる状態にすることを目的とする。

---

# 1. GitHubリポジトリのクローン

GitHub上のリポジトリをPCで編集するには、リポジトリをローカル環境へクローンする。

```text
GitHub上のリポジトリ
        ↓ clone
PC上のVolleyScopeフォルダ
```

クローン後は、VS Codeからファイルを編集し、変更をコミット、プッシュすることでGitHubへ反映できる。

状態確認には次を使用する。

```powershell
git status
```

主な表示内容：

- 現在のブランチ
- GitHub上のブランチとの差
- 変更されたファイル
- 新しく追加されたファイル
- コミット対象になっているファイル

---

# 2. `.gitignore`

`.gitignore`は、Gitの管理対象に含めないファイルやフォルダを指定する設定ファイルである。

VolleyScopeでは、以下をGitHubへアップロードしない。

```gitignore
.venv/
__pycache__/
*.pyc

data/raw/
data/clips/
data/frames/
outputs/

*.mp4
*.mov
*.avi
*.mkv

*.pt
*.onnx
```

対象外にする理由：

- `.venv`：PCごとに作り直せるPython環境
- `data/raw`：公開許可や個人情報の問題がある試合動画
- `data/frames`：動画から大量生成される画像
- `outputs`：再生成できる解析結果
- 動画ファイル：容量が大きい
- 学習済みモデル：容量が大きい

`.gitignore`はGitHubへの登録を防ぐが、OneDriveへの同期は防がない。

---

# 3. Pythonの仮想環境

## 仮想環境とは

`.venv`は、VolleyScope専用のPythonとライブラリを保存するフォルダである。

仮想マシンや別のWindowsを作っているわけではない。

```text
VolleyScope/
└── .venv/
    ├── Scripts/
    │   └── python.exe
    └── Lib/
        └── site-packages/
```

仮想環境を利用することで、別プロジェクトとライブラリの種類やバージョンが衝突することを防ぐ。

作成コマンド：

```powershell
python -m venv .venv
```

PowerShellの設定によって`Activate.ps1`を実行できない場合でも、仮想環境内のPythonを直接指定できる。

```powershell
.\.venv\Scripts\python.exe
```

---

# 4. Python Interpreter

Interpreterは、Pythonコードを読み取り、実行するプログラムである。

PC全体のPythonと、VolleyScope専用のPythonは別の実行環境である。

```text
PC全体のPython
C:\...\Python\python.exe

VolleyScope専用Python
VolleyScope\.venv\Scripts\python.exe
```

VS Codeの`Python: Select Interpreter`では、コードの実行、デバッグ、入力補完などに使用するPythonを選択する。

VolleyScopeでは次を選択する。

```text
.venv\Scripts\python.exe
```

---

# 5. ライブラリ

ライブラリは、他の開発者が作成した、再利用可能なプログラムの部品である。

VolleyScopeでは次を使用する。

| ライブラリ | 用途 |
|---|---|
| Ultralytics | YOLOの学習・推論・評価 |
| OpenCV | 動画・画像の読み込みと加工 |
| pandas | 検出結果や戦術データの集計 |
| PyTorch | 機械学習モデルの実行・学習 |
| NumPy | 数値や配列の処理 |

インストールしたライブラリは、主に次へ保存される。

```text
VolleyScope/.venv/Lib/site-packages/
```

---

# 6. `pip`

`pip`は、Pythonのライブラリをインストール・管理するためのツールである。

通常はPyPIという公開パッケージ置き場からライブラリを取得する。

```text
PyPI
 ↓
pip
 ↓
.venv/Lib/site-packages/
```

インストール例：

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
```

インストール済みライブラリの確認：

```powershell
.\.venv\Scripts\python.exe -m pip list
```

特定ライブラリの確認：

```powershell
.\.venv\Scripts\python.exe -m pip show ultralytics
```

---

# 7. `-m`の意味

`-m`は、Pythonに「指定したモジュールをPythonの機能として実行する」と伝えるオプションである。

```powershell
python -m pip
```

は、

> 現在指定しているPython環境に所属するpipを実行する

という意味になる。

単に次のように実行すると、

```powershell
pip install ultralytics
```

PC内に複数のPythonがある場合、別のPythonに所属するpipが使われる可能性がある。

そのため、次のようにPythonを明示する方が安全である。

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
```

---

# 8. `import`

`import`は、別のライブラリやPythonファイルに用意された機能を利用可能にする処理である。

```python
import cv2
```

OpenCVの機能を`cv2`という名前で使用可能にする。

```python
from pathlib import Path
```

`pathlib`の中から`Path`を読み込む。

Pythonはモジュールを最初にimportするとき、そのファイルのトップレベルを上から順番に実行し、関数・クラス・変数を利用可能な状態にする。

---

# 9. パス

パスは、PC内にあるファイルやフォルダの住所である。

## 絶対パス

PCの先頭からすべて記載した住所。

```text
C:\Users\orasa\OneDrive\Vscode\GitHub\VolleyScope\data\raw\match01.mp4
```

## 相対パス

現在位置やプロジェクトを基準にした住所。

```text
data/raw/match01.mp4
```

絶対パスは保存場所が変わると使用できなくなる。相対的に組み立てたパスは、プロジェクトを別のPCへ移しても利用しやすい。

今回のコード：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "match01.mp4"
```

`__file__`は、現在実行しているPythonファイル自身の場所を表す。

`inspect_video.py`から2階層上へ移動して、VolleyScopeのルートフォルダを取得している。

---

# 10. Pythonのインデント

Pythonでは、インデントによって処理の所属範囲を表す。

```python
def main():
    print("mainの中")
    print("これもmainの中")

print("これはmainの外")
```

`def main():`の下で同じ深さのインデントが付いている部分が、`main()`の処理内容になる。

インデントがなくなった場所で、その処理範囲が終了する。

---

# 11. 関数

関数は、複数の処理を一つの名前でまとめたものである。

```python
def main():
    print("動画を確認します")
```

`def main():`は、`main`という処理を定義しているだけで、まだ実行していない。

実際に処理を実行するには、関数を呼び出す。

```python
main()
```

`main`という名前自体に特別な動作はない。「プログラムの中心処理」という意味で一般的に使用される名前である。

---

# 12. `__name__`と`__main__`

`__name__`は、Pythonがモジュールへ自動設定する特別な変数である。

Pythonファイルを直接実行した場合：

```python
__name__ == "__main__"
```

別のPythonファイルからimportした場合：

```python
__name__ == "ファイル名"
```

そのため、次のコードを利用する。

```python
if __name__ == "__main__":
    main()
```

日本語にすると、

> このファイルがプログラム本体として直接実行された場合だけ、main関数を実行する

という意味になる。

別ファイルからimportされた場合は条件が成立せず、`main()`は実行されない。

---

# 13. `=`と`==`

`=`は値を変数へ代入する。

```python
name = "VolleyScope"
```

`==`は左右の値が同じか比較する。

```python
name == "VolleyScope"
```

次のコードは、`__name__`へ値を入れているのではなく、現在の値を確認している。

```python
if __name__ == "__main__":
```

---

# 14. `return`

`return`は関数の処理を終了し、呼び出し元へ戻る。

```python
if not VIDEO_PATH.exists():
    print("動画が見つかりません")
    return
```

動画が存在しない場合、その後の動画読み込み処理を行わずに`main()`を終了する。

異常が起きた状態で後続処理を続けないために使用する。

---

# 15. OpenCVによる動画読み込み

動画を開く処理：

```python
video = cv2.VideoCapture(str(VIDEO_PATH))
```

ファイルがOpenCVで正常に開けたか確認する。

```python
if not video.isOpened():
    print("動画を開けませんでした")
    return
```

動画情報を取得する。

```python
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
```

| 項目 | 意味 |
|---|---|
| width | 横方向の画素数 |
| height | 縦方向の画素数 |
| FPS | 1秒間に含まれるフレーム数 |
| frame_count | 動画全体のフレーム数 |

動画時間は、総フレーム数をFPSで割って求める。

```python
duration_seconds = frame_count / fps
```

処理後は動画ファイルを解放する。

```python
video.release()
```

---

# 16. 今回発生した問題

## PowerShellで仮想環境を有効化できなかった

原因：

PowerShellの実行ポリシーによって`Activate.ps1`が拒否された。

対応：

セキュリティ設定を恒久的に変更せず、仮想環境内のPythonを直接指定した。

```powershell
.\.venv\Scripts\python.exe
```

## 動画が見つからなかった

原因：

フォルダ名を`data`ではなく`date`として作成していた。

エラーメッセージに表示されたパスと、実際のフォルダ構成を比較して原因を特定した。

学び：

エラーが出た場合は「Python全体が壊れている」と判断せず、エラーメッセージが示す場所、ファイル名、処理段階を確認する。

---

# 17. 現在までにできたこと

- GitHubリポジトリをローカル環境へクローン
- VS Codeでリポジトリを開く
- プロジェクト用フォルダを作成
- `.gitignore`を設定
- Python仮想環境を作成
- Ultralytics、OpenCV、pandasをインストール
- VS CodeのPython Interpreterを設定
- Python環境の動作確認
- 試合動画を`data/raw`へ配置
- OpenCVで動画を読み込み
- 解像度、FPS、総フレーム数、動画時間を取得

## 次回の予定

1. 元動画から短い検証用クリップを作成する
2. 事前学習済みYOLOを検証用クリップへ適用する
3. 選手とボールの検出結果付き動画を生成する
4. 独自学習前の見逃し・誤検出を記録する