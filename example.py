import math
import numpy as np
import MIMO_GrayCode_Modulation as mgm


def main():
    # Setup parameters
    M = 16  # Number of transmitting and receiving antennas
    Q = 64  # Modulation level per antenna
    K = 1024  # Symbol length
    SNR_dB = 20  # SNR [dB]
    nloops = 100  # Number of loops

    # Create modulator instance
    mod = mgm.Modulator(Q)  # Complex model
    # mod = mgm.Modulator(Q, complex=False)   # Real number equivalent model

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
        Z = np.random.normal(0.0, sigma_noise, size=[M, K])
        +1j * np.random.normal(0.0, sigma_noise, size=[M, K])  # Complex model
        # Z = np.random.normal(0.0, sigma_noise, size=[2*M, K])   # Real number equivalent model
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
