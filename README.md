# MIMO Gray-Code Modulation (Python)

Python用の変復調プログラム

- BPSK，QPSK，QAM対応（すべてグレイ符号）
- 複素モデル，実数等価モデルどちらにも対応

## 依存関係

- Python 3.10+（動作確認: Python 3.13.5）
- Numpy

## インストール方法

```bash
pip install git+https://github.com/Ichihashi0317/MIMO_GrayCode_Modulation_py.git
```

補足:

- [ソースコード](MIMO_GrayCode_Modulation/modulator.py)をそのまま自分のプロジェクトへコピーして使うこともできる．

## 使い方

### インスタンス生成時の引数

```python
import MIMO_GrayCode_Modulation as mgm
Q_ant = 2 # 2: BPSK, 4: QPSK, 16: 16QAM, 64: 64QAM ...
mod = mgm.Modulator(Q_ant, complex=True)
```

- `Q_ant`: 1アンテナあたりの多値数． $2$ または $4^n$ ($n$は任意の自然数) で指定．
- `complex`:
  - `True`: 複素モデル（規定値）
  - `False`: 実数等価モデル

### ビット列生成メソッド

```python
bits = mod.gen_bits(M, K)
```

入力：

- `M`: アンテナ本数．任意の自然数で指定．
- `K`: 1フレーム中の送信ベクトル数．任意の自然数で指定．

出力：

- `bits`: 乱数生成された，変調入力用ビット列の2次元配列[^bits-note]．`shape=(log2(Q_ant) * M, K)`．

### 変調メソッド

```python
syms = mod.modulate(bits)
```

入力：

- `bits`: 送信ビットの2次元配列．1送信ベクトルに対応するビット列を列ベクトルとして含む．`shape=(log2(Q_ant) * M, K)`．[^bits-note]

出力：

- `syms`: 送信行列．送信ベクトルを列ベクトルとして含む．`complex=True` のとき `shape=(M, K)`，`complex=False` のとき `shape=(2M, K)`．シンボルマッピングは平均電力が 1 になるように正規化される．

### 復調メソッド

```python
bits = mod.demodulate(syms)
```

入力：

- `syms`: 受信行列．送信行列と同じサイズ．

出力：

- `bits`: 受信ビットの2次元配列．各アンテナの実部・虚部単位で硬判定される．送信ビット配列と同じサイズ．

[^bits-note]: `modulate()` の入力は `gen_bits()` の出力に限らず，符号化器などから出力したビット列を整形して入力しても問題ない．
