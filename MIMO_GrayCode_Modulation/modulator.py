import math
import numpy as np


class Modulator:
    def __init__(self, Q_ant: int, complex=True):
        """
        Initialize a modulator.

        Parameters:
        - Q_ant (int): Modulation level per antenna.
        - complex (bool): If True, handles complex model. If False, handles real equivalent model.
        """
        self.Q_ant = Q_ant
        self.complex = complex
        self.q_ant = int(math.log2(Q_ant))  # Bit length per antenna.
        if Q_ant == 4:
            # For QPSK.
            Q_dim = int(math.sqrt(Q_ant))  # Modulation level per dimension.
            # Data label array -> Symbol array
            label_dim = np.arange(Q_dim, dtype=float)  # Bit length per dimension.
            self.mean = label_dim.mean()
            self.k_demod = np.sqrt(
                2 * np.mean((label_dim - self.mean) ** 2)
            )  # Coefficient to set symbol spacing to 1.
            self.k_mod = (
                1 / self.k_demod
            )  # Coefficient to set power per dimension to 1/2.
        elif Q_ant > 4:
            # For QAM.
            Q_dim = int(math.sqrt(Q_ant))  # Modulation level per dimension.
            q_dim = self.q_ant // 2  # Bit length per dimension.
            self.Q_dim = Q_dim
            self.q_dim = q_dim
            # Conversion Weight Vector (bit array -> label array)
            self.weight = 2 ** np.arange(q_dim)
            # Data label array -> Symbol array
            label_dim = np.arange(Q_dim)
            self.mean = label_dim.mean()
            self.k_demod = np.sqrt(
                2 * np.mean((label_dim - self.mean) ** 2)
            )  # Coefficient to set symbol spacing to 1.
            k_mod = 1 / self.k_demod  # Coefficient to set symbol spacing to 1.
            self.symtab_dim = (self._gray2binary(label_dim) - self.mean) * k_mod
            # Conversion array (Symbol label array -> Bit array)
            bittab_dim = np.empty([q_dim, Q_dim], dtype=int)
            tmp = np.arange(Q_dim)
            for i in range(q_dim):
                bittab_dim[i, :] = tmp % 2
                tmp //= 2
            self.bittab_dim_ = bittab_dim[:, self._binary2gray(label_dim)]
            # When the binary code increases or decreases by one, the corresponding gray code changes by one bit. Since we want the data to change by only one bit when the symbol position changes by one, we map the symbol position to the binary code and the data label to the gray code.
            # binary符号が1増減すると，それに対応するgray符号は1bitだけ変化する．シンボル位置が1つ変化したときにデータは1bitだけの変化にしたいから，シンボル位置をbinary符号，データラベルをgray符号に対応させる．

    def gen_bits(self, M, K):
        """
        Generate bit sequence.

        Parameters:
        - M (int): Number of transmitting antennas.
        - K (int): Symbol length.

        Returns:
        - ndarray: Transmit bit 2D array.
        """
        return np.random.randint(2, size=[self.q_ant * M, K])

    def modulate(self, bits):
        """
        Modulate the provided bits into symbols.

        Parameters:
        - bits (ndarray): 2D array of input bits.

        Returns:
        - ndarray: Transmitted symbols 2D array.
        """
        if self.Q_ant == 2:
            # BPSK
            # Generate symbols
            syms = 2 * bits - 1
            # Adjust to complex type or real equivalent model
            if self.complex:
                syms = syms.astype(complex)
            else:
                M, K = syms.shape
                syms = np.concatenate(
                    [syms, np.zeros([M, K])], axis=0
                )  # Because zeros is a real number type, syms is output as a real number type
        elif self.Q_ant == 4:
            # QPSK
            # Generate symbols
            syms = (bits - self.mean) * self.k_mod
            if self.complex:
                # Complex model
                M = syms.shape[0] // 2
                syms = syms[:M, :] + 1j * syms[M:, :]
        else:
            # QAM
            # Import variables
            M = bits.shape[0] // self.q_ant
            K = bits.shape[1]
            # Bit array -> Data label array
            labels_dim = self.weight @ bits.reshape(2 * M, self.q_dim, K)
            # Generate symbols
            syms = self.symtab_dim[labels_dim]
            if self.complex:
                # Complex model
                syms = syms[:M, :] + 1j * syms[M:, :]
        return syms

    def demodulate(self, syms):
        """
        Demodulate the received symbols into bits.

        Parameters:
        - syms (ndarray): Received symbols 2D array.

        Returns:
        - ndarray: Received bits 2D array.
        """
        if self.Q_ant <= 4:
            # BPSK or QPSK
            if self.Q_ant == 2:
                # If BPSK, only the real part is extracted.
                if self.complex:
                    syms = syms.real
                else:
                    M = syms.shape[0] // 2
                    syms = syms[:M]
            elif self.complex:
                # If QPSK, convert to real-valued equivalents.
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            # Calculate bit
            bits = (syms > 0.0).astype(int)
        else:
            # QAM
            if self.complex:
                # Convert to real-valued equivalents.
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            # Import variables
            M = syms.shape[0] // 2
            K = syms.shape[1]
            # Normalization/integer (multiply by a factor to obtain an integer interval, add a constant so that it is greater than or equal to zero, and round to the nearest integer)
            # 規格化・整数化（整数間隔になるように係数をかけて，0以上になるように定数を足して，整数に丸める）
            syms_ = (
                np.around(syms * self.k_demod + self.mean)
                .astype(int)
                .clip(0, self.Q_dim - 1)
            )
            # Calculate bit
            bits = (
                self.bittab_dim_[:, syms_]
                .transpose(1, 0, 2)
                .reshape(2 * M * self.q_dim, K)
            )
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
