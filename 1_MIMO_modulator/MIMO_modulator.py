import math
import numpy as np

class MIMO_modulator():
    ## 引数 ##
    # Q_ant: 1アンテナあたりの多値数
    # M: 送信アンテナ本数
    # real_eq: Trueにすると実数等価モデルを扱える
    def __init__(self, Q_ant, M=1, real_eq=False):
        ### 変調多値数計算
        Q = Q_ant**M    # 全多値数
        if Q_ant == 2:  # BPSK
            Q_dim = 2
        else:           # QPSK or QAM
            Q_dim = int(math.sqrt(Q_ant))   # 1次元あたりの多値数
        
        ### 変調準備
        # ベクトルのラベル -> 各次元のラベル 変換用 2次元配列生成
        label_tab = np.empty([2*M, Q], dtype=int)
        tmp = np.arange(Q)
        for i in range(2*M):
            label_tab[i,:] = tmp % Q_dim
            tmp //= Q_dim
        # データラベル -> シンボル位置 変換
        #（binary符号が1増減すると，それに対応するgray符号は1bitだけ変化する．シンボル位置が1つ変化したときにデータは1bitだけの変化にしたいから，シンボル位置をbinary符号，データラベルをgray符号に対応させる．）
        symtab_  = self.gray2binary(label_tab)
        # 平均 0
        mean_vec = symtab_.mean(axis=1, keepdims=True)  # 次元ごとの平均値
        symtab_  = symtab_ - mean_vec
        # 平均電力 1
        std = np.sqrt((symtab_**2).sum()/(M*Q))         # 平均電力のルート
        symtab_R = symtab_ / std
        # 複素化
        symtab_C = symtab_R[:M, :] + 1j * symtab_R[M:, :]
        ## 変調用配列選択
        if real_eq == True:
            self.symtab = symtab_R
        else:
            self.symtab = symtab_C
        
        ### 復調準備
        self.Q_ant = Q_ant
        self.complex = not real_eq
        if   Q_ant == 2:
            self.M = M
            self.weight = Q_dim**np.arange(M)   # BPSKのときは虚部を無視する
        elif Q_ant == 4:
            self.weight = Q_dim**np.arange(2*M)
        else:
            self.Q_dim = Q_dim
            self.mean = mean_vec[0]
            self.std = std
            self.weight = Q_dim**np.arange(2*M)
            # bittab計算
            q = int(math.log2(Q))
            self.bittab = np.empty([q, Q], dtype=int)
            tmp = np.arange(Q)
            for i in range(q):
                self.bittab[i, :] = tmp % 2
                tmp //= 2
    
    ### 変調メソッド
    # 入力：送信データを表す整数の1次元配列
    # 出力：送信行列
    def modulate(self, label):
        return self.symtab[:, label]
    
    ### 復調メソッド
    # 入力：受信行列
    # 出力：受信データを表す整数の1次元配列と，そのビット表現の2次元配列
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
            bits = syms > 0.0
            # データラベル計算
            label = self.weight @ bits
        ### QAM
        else:
            # 実数等価
            if self.complex:
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            # 規格化・整数化（整数間隔になるように係数をかけて，0以上になるように定数を足して，丸める）
            syms_ = np.around(syms * self.std + self.mean).astype(int).clip(0, self.Q_dim - 1)
            # 次元ごとのデータラベル
            labels_dim = self.binary2gray(syms_)
            # シンボルベクトル全体のデータラベル（各アンテナのデータラベルの重み付け和）
            label = self.weight @ labels_dim
            # bit計算
            bits = self.bittab[:, label]
        return label, bits
    
    @staticmethod
    def binary2gray(binary):
        return binary ^ (binary >> 1)
    
    @staticmethod
    def gray2binary(gray):
        tmp = gray.copy()
        mask = gray >> 1
        while mask.any():
            tmp ^= mask
            mask >>= 1
        return tmp