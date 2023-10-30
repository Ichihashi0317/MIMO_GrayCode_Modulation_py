# MIMO_GrayCode_Modulation_py
Python用の変復調プログラム
- BPSK，QPSK，QAM対応（すべてグレイ符号）
- 複素モデル，実数等価モデルどちらにも対応

## インストール方法
```
pip install git+https://github.com/Ichihashi0317/MIMO_GrayCode_Modulation_py.git
```

## 使い方

  ### インスタンス生成時の引数
  ```
  import MIMO_GrayCode_Modulation_py as mgm
  Q_ant = 2 # 2: BPSK, 4: QPSK, 16: 16QAM, 64: 64QAM ...
  mod = mgm.Modulator(Q_ant)
  ```
  - Q_ant (int) - 1アンテナあたりの多値数． $2$ または $4^n$ ($n$は自然数) で指定．
  - complex (bool) - Trueの場合は複素モデルを扱う．Falseの場合は実数等価モデルを扱う．規定値はTrue．
  
  ### ビット列生成メソッド
  ```
  bits = mod.gen_bits(M, K)
  ```
  入力：
  - M (int) - アンテナ本数．任意の自然数で指定．
  - K (int) - 1フレーム中の送信ベクトル数．任意の自然数で指定．
  
  出力：
  - bits (ndarray) - 変調メソッド入力に合わせたサイズのビット列．乱数で生成される．
  
  [^1]
  
  ### 変調メソッド
  ```
  syms = mod.modulate(bits)
  ```
  入力：
  - bits (ndarray) - 送信ビットの2次元配列．1送信ベクトルに対応するビット列を列ベクトルとして含み，それを行方向へ並べた行列．1送信ベクトルあたりのビット数を $q = M \log_2{Q_\mathrm{ant}}$ とすると，サイズは $(q \times K)$．

  出力：
  - syms (ndarray) - 送信行列．送信ベクトルを列ベクトルとして含む．サイズは，複素モデルの場合は $(M \times K)$，実数等価モデルの場合は $(2M \times K)$．シンボルマッピングは平均電力が1に規格化される．

  ### 復調メソッド
  ```
  bits = mod.demodulate(syms)
  ```
  入力：
  - syms (ndarray) - 受信行列．サイズは送信行列と同じ．


  出力：
  - bits (ndarray) - 受信ビットの2次元配列．各アンテナの実部・虚部単位で硬判定される．サイズは送信ビットの配列と同じ．

[^1]: 変調メソッドへ入力する送信ビット列は，必ずしもビット列生成メソッドで作る必要はなく，自分で作ったものでも問題ない．<br>例えば，符号化器などから出力したビット列を，サイズを整えて変調メソッドへ入力することは可能．a
