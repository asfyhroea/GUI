# 手書き数字分類・精度判定アプリ(2025/02/20)
### 概要
このアプリは、手書きした数字を AI が分類し、その認識結果と信頼度（確率）を表示するものです。
PyTorch を用いて畳み込みニューラルネットワーク（CNN）を構築し、MNIST データセットの手書き数字を学習させました。
PySide を使用して GUI を実装し、画面上に手書きした 0〜9 の数字を認識・分類します。

きれいに書かれた数字ほど、AI の認識精度が向上し、信頼度が高くなります。
逆に、崩れた数字や判別しづらい形の数字は、信頼度が低くなることがあります。
このアプリを使って、自分の書いた数字がどれだけ「正しく認識されるか」試してみてください！

### 使い方
requirements.txt のライブラリをインストールします。
predict.py を実行すると、手書き入力用の GUI が表示されます。
キャンバス上に 0〜9 の数字を書き、「決定」ボタンを押すと、認識結果と信頼度（確率）が表示されます。
きれいに書いた数字ほど高い確率で認識されます。
文字を消したい場合は、「消す」ボタンを押してください。
### 必要なライブラリ
以下のコマンドを実行し、必要なライブラリをインストールしてください。
```
pip install -r requirements.txt
```
