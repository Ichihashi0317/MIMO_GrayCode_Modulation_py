import math
import numpy as np

class Modulator():
    ## 引数 ##
    # Q_ant: 1アンテナあたりの多値数（2または4の累乗）
    # real_eq: Falseなら複素モデル，Trueなら実数等価モデルを扱える
    def __init__(self, Q_ant, M=1, complex=True):
        self.Q_ant = Q_ant
        self.M = M
        self.complex = complex
        ## QPSKのとき
        if Q_ant == 4:
            Q_dim = int(math.sqrt(Q_ant))   # 1次元あたりの多値数
            # データラベル -> シンボル 変換用 変数
            label_dim = np.arange(Q_dim, dtype=float)
            self.mean = label_dim.mean()
            self.k_demod = np.sqrt(2 * np.mean((label_dim - self.mean) ** 2))   # シンボル間隔を1にするための係数
            self.k_mod = 1 / self.k_demod   # 1次元あたりの電力を1/2にするための係数
        ## QAMのとき
        elif Q_ant > 4:
            Q_dim = int(math.sqrt(Q_ant))   # 1次元あたりの多値数
            self.Q_dim = Q_dim
            q_dim = int(math.log2(Q_dim))   # 1次元あたりのビット長
            self.q_dim = q_dim
            self.q_ant = q_dim * 2          # 1アンテナあたりのビット長
            self.q_all = M * self.q_ant     # アンテナ全体のビット長
            # ビット列 -> データラベル 変換用 重みベクトル
            self.weight = 2 ** np.arange(q_dim)
            # データラベル -> シンボル 変換用 配列
            label_dim = np.arange(Q_dim)
            self.mean = label_dim.mean()
            self.k_demod = np.sqrt(2 * np.mean((label_dim - self.mean) ** 2))   # シンボル間隔を1にするための係数
            k_mod = 1 / self.k_demod        # 1次元あたりの電力を1/2にするための係数
            self.symtab_dim = (self._gray2binary(label_dim) - self.mean) * k_mod
            # シンボル位置ラベル -> ビット列 変換用 配列
            bittab_dim = np.empty([q_dim, Q_dim], dtype=int)
            tmp = np.arange(Q_dim)
            for i in range(q_dim):
                bittab_dim[i, :] = tmp % 2
                tmp //= 2
            self.bittab_dim_ = bittab_dim[:, self._binary2gray(label_dim)]
            #（binary符号が1増減すると，それに対応するgray符号は1bitだけ変化する．シンボル位置が1つ変化したときにデータは1bitだけの変化にしたいから，シンボル位置をbinary符号，データラベルをgray符号に対応させる．）
    
    ### 変調メソッド
    # 入力：送信ビットの2次元配列
    # 出力：送信行列
    def modulate(self, bits):
        ### BPSK
        if self.Q_ant == 2:
            # シンボル生成
            syms = 2 * bits - 1
            # 複素数型or実数等価モデルに合わせる
            if self.complex:
                syms = syms.astype(complex)
            else:
                K = syms.shape[1]
                syms = np.concatenate([syms, np.zeros([self.M, K])], axis=0) # zerosが実数型だから，symsは実数型として出力される
        ### QPSK
        elif self.Q_ant == 4:
            # シンボル生成
            syms = (bits - self.mean) * self.k_mod
            # 複素モデル
            if self.complex:
                syms = syms[:self.M, :] + 1j * syms[self.M:, :]
        ### QAM
        else:
            # 変数取り込み
            M = self.M
            K = bits.shape[1]
            # ビット列 -> 各次元のデータラベル
            labels_dim = self.weight @ bits.reshape(2*M, self.q_dim, K) # (2*M, K) = (q_dim,) @ (2*M*q_dim, K).reshape(2*M, q_dim, K)
            # シンボル生成
            syms = self.symtab_dim[labels_dim]
            # 複素モデル
            if self.complex:
                syms = syms[:M, :] + 1j * syms[M:, :]
        return syms
    
    ### 復調メソッド
    # 入力：受信行列
    # 出力：受信ビットの2次元配列
    def demodulate(self, syms):
        ### BPSK or QPSK
        if self.Q_ant <= 4:
            # BPSKなら実部だけ抜き出す
            if self.Q_ant == 2:
                if self.complex:
                    syms = syms.real
                else:
                    syms = syms[:self.M]
            # QPSKなら実数等価
            elif self.complex:
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            # bit計算
            bits = (syms > 0.0).astype(int)
        ### QAM
        else:
            # 実数等価
            if self.complex:
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            # 変数取り込み
            K = syms.shape[1]
            # 規格化・整数化（整数間隔になるように係数をかけて，0以上になるように定数を足して，整数に丸める）
            syms_ = np.around(syms * self.k_demod + self.mean).astype(int).clip(0, self.Q_dim - 1)
            # bit計算
            bits = self.bittab_dim_[:, syms_].transpose(1, 0, 2).reshape(self.q_all, K) # (q_all, K) = (q_dim, 2*M, K).transpose(0 <-> 1).reshape(q_all, K), (q_dim, 2*M, K) = (q_dim, Q_dim)[:, (2*M, K)]
        return bits
    
    @staticmethod
    def _binary2gray(binary):
        return binary ^ (binary >> 1)
    
    @staticmethod
    def _gray2binary(gray):
        tmp = gray.copy()
        mask = gray >> 1
        while mask.any():
            tmp ^= mask
            mask >>= 1
        return tmp