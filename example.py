import math
import numpy as np
from modulator import Modulator


def main():
    # パラメータ設定
    M = 16          # アンテナ本数
    Q_ant = 64      # 1アンテナあたりの多値数
    SNR_dB = 30     # SNR [dB]
    K = 1024        # シンボル長
    nloops = 100    # ループ数

    # 雑音レベル計算
    SNR = 10.0 ** (SNR_dB / 10.0)
    N0 = M / SNR
    sigma_noise = math.sqrt(N0 / 2.0)

    # 変復調器インスタンス生成
    mod = Modulator(Q_ant, M, K)

    # ビットエラー数
    BE = 0

    # テストループ
    for _ in range(nloops):

        # 送信ビット生成
        bits = mod.make_bits()

        # 変調
        X = mod.modulate(bits)

        # AWGN通信路
        Z = np.random.normal(0.0, sigma_noise, size=[M, K]) + 1j * np.random.normal(0.0, sigma_noise, size=[M, K])
        Y = X + Z

        # 復調
        bits_hat = mod.demodulate(Y)

        # ビットエラー数計算
        BE += (bits != bits_hat).sum()

    # ビット誤り率計算
    BER = BE / (nloops * K * mod.q)

    # 結果表示
    print('BER = ', BER)


if __name__ == '__main__':
    main()
