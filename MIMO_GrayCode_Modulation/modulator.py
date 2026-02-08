"""MIMO Gray-Code Modulation (Python).

This module provides Gray-coded modulation / demodulation utilities:

- BPSK, QPSK, and square QAM (Gray labeling)
- Complex baseband model and real-valued equivalent model

The main entry point is :class:`Modulator`.
"""

from typing import Any, TypeAlias

import numpy as np
from numpy.random import randint
from numpy.typing import NDArray

NDArrayInt: TypeAlias = NDArray[np.integer[Any]]
NDArrayFloat: TypeAlias = NDArray[np.floating[Any]]
NDArrayComplex: TypeAlias = NDArray[np.complexfloating[Any, Any]]


class Modulator:
    """Gray-coded modulator / demodulator for MIMO simulations.

    Parameters
    ----------
    Q_ant:
        Modulation order per antenna. Must be either ``2`` (BPSK) or ``4**n`` for any
        positive integer ``n`` (QPSK, 16QAM, 64QAM, ...).
    complex:
        If ``True`` (default), use a complex baseband model.
        If ``False``, use a real-valued equivalent model (I/Q stacked).

    Notes
    -----
    - Bit mapping uses Gray labeling so that adjacent constellation points differ by
      one bit (within each dimension).
    - The constellation is normalized so that the *average transmit power per complex
      symbol* becomes 1 in the complex model. Equivalently, each real dimension has
      average power 1/2.
    """

    def __init__(self, Q_ant: int, complex: bool = True) -> None:
        """Initialize a modulator.

        See the class docstring for details.
        """
        if type(Q_ant) is not int:
            raise TypeError("Q_ant must be an integer")

        q_ant = Q_ant.bit_length() - 1
        if not (Q_ant == 2 or Q_ant >= 4 and Q_ant.bit_count() == 1 and q_ant % 2 == 0):
            raise ValueError("Q_ant must be 2 or 4**n")

        self._Q_ant = Q_ant
        self.complex = complex
        self._q_ant = q_ant
        if Q_ant == 2:
            return

        ## QPSK / QAM
        q_dim = self._q_ant // 2
        Q_dim = 1 << q_dim
        label_dim = np.arange(Q_dim)
        self._mean: np.floating = label_dim.mean()
        # Scaling factors:
        # - k_demod makes the minimum symbol spacing equal to 1.
        # - k_mod normalizes power so that each real dimension has average power 1/2.
        k_demod: np.floating = np.sqrt(2.0 * np.mean((label_dim - self._mean) ** 2))
        k_mod: np.floating = 1.0 / k_demod
        if Q_ant == 4:
            self._k_mod = k_mod
            return

        ## QAM
        self._q_dim = q_dim
        self._Q_dim = Q_dim
        self._k_demod = k_demod
        self._weight = 2 ** np.arange(q_dim)
        self._symtab_dim = (self._gray2binary(label_dim) - self._mean) * k_mod
        bittab_dim = np.empty([q_dim, Q_dim], dtype=int)
        tmp = np.arange(Q_dim)
        for i in range(q_dim):
            bittab_dim[i, :] = tmp & 1
            tmp >>= 1
        self._bittab_dim_ = bittab_dim[:, self._binary2gray(label_dim)]
        # If the binary index changes by ±1, the corresponding Gray code changes by
        # exactly one bit. This is why we associate:
        #   - constellation position: binary index
        #   - data label (bits): Gray index

    def gen_bits(self, M: int, K: int) -> NDArrayInt:
        """Generate random transmit bits.

        Parameters
        ----------
        M:
            Number of antennas (positive integer).
        K:
            Number of transmit vectors per frame (positive integer).

        Returns
        -------
        bits:
            A 2-D array of bits with shape ``(log2(Q_ant) * M, K)``.
        """
        return randint(2, size=[self._q_ant * M, K])

    def modulate(self, bits: NDArrayInt) -> NDArrayComplex | NDArrayFloat:
        """Map bits to modulation symbols.

        Parameters
        ----------
        bits:
            A 2-D bit array with shape ``(log2(Q_ant) * M, K)``, where each column is
            the bit vector for one transmit vector.

        Returns
        -------
        syms:
            Transmit matrix whose columns are transmit vectors.

            - If ``complex=True``: shape ``(M, K)`` and dtype is complex.
            - If ``complex=False``: shape ``(2M, K)`` (I/Q stacked) and dtype is float.

        Notes
        -----
        The constellation is normalized so that the average power becomes 1 in the
        complex model (or 1/2 per real dimension).
        """
        if bits.shape[0] % self._q_ant != 0:
            raise ValueError("bits.shape[0] must be a multiple of log2(Q_ant)")

        if self._Q_ant == 2:
            ## BPSK
            syms = 2 * bits - 1
            if self.complex:
                syms = syms.astype(complex)
            else:
                M, K = syms.shape
                syms = np.concatenate([syms, np.zeros([M, K])], axis=0)
                # Since zeros is real-valued, the concatenated output stays real.

        elif self._Q_ant == 4:
            ## QPSK
            syms = (bits - self._mean) * self._k_mod
            if self.complex:
                M = syms.shape[0] // 2
                syms = syms[:M, :] + 1j * syms[M:, :]

        else:
            ## QAM
            M = bits.shape[0] // self._q_ant
            K = bits.shape[1]
            labels_dim = self._weight @ bits.reshape(2 * M, self._q_dim, K)
            syms = self._symtab_dim[labels_dim]
            if self.complex:
                syms = syms[:M, :] + 1j * syms[M:, :]

        return syms

    def demodulate(self, syms: NDArrayComplex | NDArrayFloat) -> NDArrayInt:
        """Hard-decision demodulation.

        Parameters
        ----------
        syms:
            Received matrix. Must have the same shape as the output of
            :meth:`modulate`.

        Returns
        -------
        bits:
            A 2-D bit array with the same shape as the input to :meth:`modulate`.
            Decisions are made independently for each real dimension (I and Q).
        """
        if self._Q_ant <= 4:
            ### BPSK / QPSK
            if self._Q_ant == 2:
                ## BPSK
                if self.complex:
                    syms = syms.real
                else:
                    M = syms.shape[0] // 2
                    syms = syms[:M]
            elif self.complex:
                ## QPSK
                syms = np.concatenate([syms.real, syms.imag], axis=0)
            return (syms > 0.0).astype(int)

        ## QAM
        if self.complex:
            syms = np.concatenate([syms.real, syms.imag], axis=0)
        M = syms.shape[0] // 2
        K = syms.shape[1]
        # Scale so the grid spacing becomes 1, shift to be non-negative, round to the
        # nearest integer label, and clip to the valid range.
        syms_ = (
            (syms * self._k_demod + self._mean)
            .round()
            .astype(int)
            .clip(0, self._Q_dim - 1)
        )
        return (
            self._bittab_dim_[:, syms_]
            .transpose(1, 0, 2)
            .reshape(2 * M * self._q_dim, K)
        )

    @staticmethod
    def _binary2gray(binary: NDArrayInt) -> NDArrayInt:
        return binary ^ (binary >> 1)

    @staticmethod
    def _gray2binary(gray: NDArrayInt) -> NDArrayInt:
        tmp = gray.copy()
        mask = gray >> 1
        while mask.any():
            tmp ^= mask
            mask >>= 1
        return tmp
