# Stock Price Prediction (Graduation Research) — 4 Models (Trinary Classification)

このリポジトリは、**株価の方向性を三値分類（trinary classification）**する卒業研究用の実験コードです。  
次の4つのモデルを同一のデータパイプラインで比較します。

- **Transformer**（時系列ウィンドウ入力）
- **LSTM**（時系列ウィンドウ入力）
- **Logistic Regression**（ウィンドウを集約して2D特徴にする）
- **LightGBM**（ウィンドウを集約して2D特徴にする）

> ✅ 重要：このリポジトリは「動作・結果を変えずに」構造だけ整理したものです。  
> そのため **実行入口（旧パス互換）** と、整理後の **本体（src/）** の2層構造になっています。

---

## 目次

- [1. まず何をすればいい？（最短で動かす）](#1-まず何をすればいい最短で動かす)
- [2. 4つのモデルを実行する](#2-4つのモデルを実行する)
- [3. 実行すると何が出力される？（結果の見方）](#3-実行すると何が出力される結果の見方)
- [4. フォルダ構成とファイルの役割](#4-フォルダ構成とファイルの役割)
- [5. 銘柄・期間・窓幅などを変えたい（設定変更の場所）](#5-銘柄期間窓幅などを変えたい設定変更の場所)
- [6. テスト（回帰テスト）を走らせる](#6-テスト回帰テストを走らせる)
- [7. 旧コードと結果が同じか比較したい](#7-旧コードと結果が同じか比較したい)
- [8. 再現性（seed / TFの注意）](#8-再現性seed--tfの注意)
- [9. よくあるつまずき（FAQ）](#9-よくあるつまずきfaq)

---

## 1. まず何をすればいい？（最短で動かす）

### 1) リポジトリ直下へ移動

> **必ず**「このREADMEがある階層（リポジトリ直下）」で実行してください。  
> 相対パスで出力する設計のため、別ディレクトリから実行すると出力先がズレます。

```bash
cd stockpred_refactored
```

### 2) 仮想環境を作って依存関係を入れる

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows（PowerShell）
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> TensorFlow のインストールが環境依存で失敗する場合があります。  
> その場合は [FAQ](#9-よくあるつまずきfaq) を見てください。

---

## 2. 4つのモデルを実行する

このリポジトリでは、**実行用の入口（旧パス互換）**が用意されています。  
初心者はまずここを実行してください（これが一番安全で迷いません）。

### Transformer
```bash
python transformer/trinary_transformer_old.py
```

### LSTM
```bash
python LSTM/trinary_LSTM.py
```

### Logistic Regression
```bash
python Logistic_Regression/trinary_logistic_regression.py
```

### LightGBM
```bash
python LightGBM/trinary_LightGBM.py
```

---

## 3. 実行すると何が出力される？（結果の見方）

各モデルを実行すると、基本的に「そのモデルのフォルダ配下」に成果物が出ます。

例（Transformerの場合）：

- `transformer/run_summary.json`  
  実行結果の要約（fold平均のスコア、設定値など）が入ります。  
  **まずはこれを見るのが最短**です。

- `transformer/figs/...`  
  学習曲線、混同行列、指標の図、Permutation Importance（MDA）の出力などが入ります。

他のモデルも同様に、例えば：

- `LSTM/run_summary.json`, `LSTM/figs/...`
- `Logistic_Regression/run_summary.json`, `Logistic_Regression/figs/...`
- `LightGBM/run_summary.json`, `LightGBM/figs/...`

> ✅ “出力先パス”は、元コードと同じ相対パスになるように設計されています。

---

## 4. フォルダ構成とファイルの役割

このリポジトリは **2層構造**です。

### A) 実行入口（旧パス互換）

以下のフォルダは「入口」です。  
中身は薄いラッパーで、**本体（src/）の実験モジュールをそのまま起動**します。

- `transformer/trinary_transformer_old.py`
- `LSTM/trinary_LSTM.py`
- `Logistic_Regression/trinary_logistic_regression.py`
- `LightGBM/trinary_LightGBM.py`

> ✅ 共同開発や比較実験で「昔の実行手順がそのまま動く」ことを優先しています。

---

### B) 本体コード（整理後の正本）

`src/stock_pred/` が「本体」です。今後のレビューや拡張はここを見ます。

- `src/stock_pred/common_shared.py`  
  4モデル共通の処理（例：seed固定、評価、分割、図保存、run_summary保存など）

- `src/stock_pred/dataset_pipeline.py`  
  4モデル共通のデータ処理（例：株価取得、特徴量、ラベル作成、窓化/2D化など）

- `src/stock_pred/models/`  
  実験スクリプト本体（4モデル分）
  - `transformer_trinary_transformer_old.py`
  - `lstm_trinary_LSTM.py`
  - `logistic_regression_trinary_logistic_regression.py`
  - `lightgbm_trinary_LightGBM.py`

---

### C) 互換 shim（importを壊さないための橋渡し）

ルート直下にある以下は「互換用」です。旧コードの import を壊しません。

- `common_shared.py` → `src/stock_pred/common_shared.py` を再export
- `dataset_pipeline.py` → `src/stock_pred/dataset_pipeline.py` を再export

---

### D) テスト / ツール

- `tests/`  
  回帰テスト（前処理や分割などの “壊れていないか” を確認する軽量テスト）

- `tools/compare_run_summaries.py`  
  `run_summary.json` 同士を比較して、差分を見やすくする補助スクリプト

---

## 5. 銘柄・期間・窓幅などを変えたい（設定変更の場所）

このプロジェクトは、基本的に **コマンド引数ではなく「スクリプト内の設定（dataclass）」を編集**して実験します。

編集場所（例：Transformer）  
- `src/stock_pred/models/transformer_trinary_transformer_old.py`

各モデルのファイル内にはだいたい以下の設定クラスがあります：

- `DataConfig`：データ関連（ticker、期間、horizon、win、k_tau、pooling、出力先など）
- `SplitConfig`：分割（PWFE、embargoなど）
- `TrainConfig`：学習関連（epoch、batch、early stopping等がある場合）
- `ModelConfig`：モデル構造/ハイパーパラメータ（各モデル固有）

例（よく触る項目のイメージ）：
- `ticker`：銘柄（例 `"7203.T"`）
- `start`, `end`：期間（例 `"2001-01-01"`, `"2024-12-31"`）
- `horizon`：何日先を予測するか
- `win`：ウィンドウ長（時系列入力の長さ）
- `k_tau`：ラベル境界の係数（分類の閾値に関係）
- `output_root`：図の出力先（モデルごとに既存パス維持のため固定されがち）

> ✅ 初心者向けおすすめ手順  
> 1) まずデフォルトのまま実行して動作確認  
> 2) 次に `ticker` と `start/end` だけ変えて再実行  
> 3) 最後に `win` や `k_tau` を触る

---

## 6. テスト（回帰テスト）を走らせる

テストは「学習をフルで回す」ものではなく、主に

- 前処理の出力形状が変わっていないか
- 分割が壊れていないか
- seed固定関数が呼べるか

などを軽く検証します。

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 7. 旧コードと結果が同じか比較したい

旧コードで出した `run_summary.json` と、新コードで出した `run_summary.json` を比較できます。

```bash
python tools/compare_run_summaries.py path/to/old/run_summary.json transformer/run_summary.json
```

> TensorFlow/GPU を使う場合、完全一致が難しい環境もあります。  
> 詳細は [再現性の注意](#8-再現性seed--tfの注意) を参照してください。

---

## 8. 再現性（seed / TFの注意）

- 共通ユーティリティで seed 固定を行います（元コードに合わせるため）
- ただし **深層学習（TensorFlow）＋GPU** の場合、CUDA/cuDNNやカーネル実装により  
  **完全一致が崩れることがあります**（一般に起こり得ます）

何が保証できて、何が環境依存かは：
- `tests/test_reproducibility_notes.py`

にまとめています。

---

## 9. よくあるつまずき（FAQ）

### Q1. `pip install -r requirements.txt` で TensorFlow が入らない
A. OS / Python バージョン / CPU/GPU に依存します。  
まずは以下を確認してください：

- 使っている Python のバージョン（例：3.10 など）
- Apple Silicon / NVIDIA GPU など、環境差

対処の方向性：
- CPUのみで動かす（TensorFlow CPU版）
- PythonバージョンをTensorFlowが対応しているものに合わせる

---

### Q2. 実行したのに `run_summary.json` が見つからない
A. 「リポジトリ直下」で実行しているか確認してください。

✅ 正しい例：
```bash
cd stockpred_refactored
python transformer/trinary_transformer_old.py
```

---

### Q3. どのファイルを読めばロジックが分かる？
A. 基本はここです：

- 前処理：`src/stock_pred/dataset_pipeline.py`
- 共通処理：`src/stock_pred/common_shared.py`
- 各モデル本体：`src/stock_pred/models/*.py`

入口の `transformer/` や `LSTM/` の `.py` は「起動するだけ」なので、ロジックはほぼありません。

---

## ライセンス / 免責
- データ取得は外部サービス（例：株価データ）に依存します。取得結果は時点により変動する可能性があります。
- 研究目的のコードであり、投資助言を目的としません。
