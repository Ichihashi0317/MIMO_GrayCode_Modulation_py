import math

import MIMO_GrayCode_Modulation as mgm
import numpy as np


def main():
    # Setup parameters
    complex_model = True
    M = 16  # Number of transmitting and receiving antennas
    Q = 64  # Modulation level per antenna
    K = 1024  # Symbol length
    SNR_dB = 20  # SNR [dB]
    nloops = 100  # Number of loops

    # Create modulator instance
    mod = mgm.Modulator(Q, complex_model)

    # Calculate noise level
    SNR = 10.0 ** (SNR_dB / 10.0)
    N0 = 1.0 / SNR  # In case of no interference between antennas
    sigma_noise = math.sqrt(N0 / 2.0)

    # Test loop
    BE = 0
    for _ in range(nloops):
        # Generate transmitted bit sequence
        bits = mod.gen_bits(M, K)

        # Modulate
        X = mod.modulate(bits)

        # AWGN channel
        if complex_model:
            # Complex model
            Z = (np.random.normal(0.0, sigma_noise, size=[M, K])
                 + 1j * np.random.normal(0.0, sigma_noise, size=[M, K]))  # fmt: skip
        else:
            # Real number equivalent model
            Z = np.random.normal(0.0, sigma_noise, size=[2 * M, K])

        Y = X + Z

        # Demodulate
        bits_hat = mod.demodulate(Y)

        # Calculate number of bit errors
        BE += (bits != bits_hat).sum()

    # Calculate bit error rate
    BER = BE / (nloops * K * M * math.log2(Q))

    # Display results
    print("BER =", BER)


if __name__ == "__main__":
    main()
