"""Pure Python fallback for zernike_moments Cython extension."""
import numpy as np


def _zernike_yiq(A, Ipi, Yiq, N, deg):
    """Row-integral accumulation over the ink pixels of A."""
    for i in range(N):
        for j in range(N):
            if not A[i, j]:
                for q in range(deg + 1):
                    Yiq[q, i] += Ipi[q, j]


def _zernike_mpqs(Yiq, Ipi, Mpqs, N, deg):
    """Geometric moments Mpqs from the row integrals."""
    for p in range(deg + 1):
        for q in range(deg + 1):
            mpq = 0.0
            for i in range(N):
                mpq += Yiq[q, i] * Ipi[p, i]
            Mpqs[p, q] = mpq


def _zernike_rpq(D, ws, Mpqs, Rpq, deg):
    """Radial moments Rpq (only for even p−q)."""
    for p in range(deg + 1):
        for q in range(deg + 1):
            diff = p - q
            if diff % 2 != 0:
                continue
            S = diff // 2
            rpq = 0.0j
            for j in range(S + 1):
                for m in range(q + 1):
                    rpq += ws[m % 4] * D[S, j] * D[q, m] * Mpqs[p - 2 * j - m, 2 * j + m]
            Rpq[p, q] = rpq


def _zernike_zpq(Bpqk, Rpq, Zpq, o_PI, deg):
    """Zernike moment magnitudes Zpq from the radial moments."""
    i = 0
    for p in range(deg + 1):
        pp1 = p + 1
        for q in range(p % 2, pp1, 2):
            zpk = 0.0j
            for k in range(q, pp1, 2):
                zpk += Bpqk[p, q, k] * Rpq[k, q]
            Zpq[i] = abs(pp1 * o_PI * zpk)
            i += 1


def zernike_features(A, D, Bpqk, Ipi, Mpqs, Rpq, Yiq, ws, Zpq):
    N = 32
    o_PI = 1.0 / np.pi
    deg = 17
    Yiq[:] = 0.0
    _zernike_yiq(A, Ipi, Yiq, N, deg)
    _zernike_mpqs(Yiq, Ipi, Mpqs, N, deg)
    _zernike_rpq(D, ws, Mpqs, Rpq, deg)
    _zernike_zpq(Bpqk, Rpq, Zpq, o_PI, deg)
