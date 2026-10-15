"""
Pure Python fallback for fast_utils Cython extension.
Used when fast_utils.so/.pyd is not available (e.g. frozen PyInstaller builds on Mac).
"""
import math
import numpy as np
import cv2


def gausslogprob(mean, std, x):
    pi = math.pi
    df = x - mean
    dn = math.log(math.sqrt(2 * pi))
    return (-(df * df) / (2 * std * std)) - dn - math.log(std)


def scale_transform(x, mean, o_std, size):
    for i in range(size):
        x[i] = (x[i] - mean[i]) * o_std[i]


def fadd_padding(arr, padding):
    h, w = arr.shape
    nh = h + 2 * padding
    nw = w + 2 * padding
    newarr = np.ones((nh, nw), dtype=np.uint8)
    newarr[padding:padding + h, padding:padding + w] = arr
    return newarr


def _first_ink_row(arr, indices):
    """First index in `indices` whose row contains ink (a 0), or None."""
    for i in indices:
        if np.any(arr[i, :] == 0):
            return i
    return None


def _first_ink_col(arr, indices):
    """First index in `indices` whose column contains ink (a 0), or None."""
    for i in indices:
        if np.any(arr[:, i] == 0):
            return i
    return None


def _trim_vertical(arr, sides, rows):
    top, bottom, oft, ofb = 0, rows, 0, 0
    if 't' in sides:
        r = _first_ink_row(arr, range(rows))
        if r is not None:
            top = oft = r
    if 'b' in sides:
        r = _first_ink_row(arr, range(bottom - 1, 0, -1))
        if r is not None:
            ofb = -(bottom - r)
            bottom = r
    return top, bottom, oft, ofb


def _trim_horizontal(arr, sides, right):
    left, ofl, ofr = 0, 0, 0
    if 'l' in sides:
        c = _first_ink_col(arr, range(right))
        if c is not None:
            left = ofl = c
    if 'r' in sides:
        c = _first_ink_col(arr, range(right - 1, 0, -1))
        if c is not None:
            ofr = -(right - c)
            right = c
    return left, right, ofl, ofr


def ftrim(arr, sides='trbl', new_offset=False):
    rows, cols = arr.shape
    top, bottom, oft, ofb = _trim_vertical(arr, sides, rows)
    left, right, ofl, ofr = _trim_horizontal(arr, sides, cols)
    if not new_offset:
        return arr[top:bottom, left:right]
    return arr[top:bottom, left:right], {'top': oft, 'bottom': ofb, 'right': ofr, 'left': ofl}


def to255(a):
    a[:] = a * 255
    return a


def _place_normalized_rows(c, b, starti, endi, LL):
    """Center the resized image `b` vertically into the 32×32 output `c`, padding
    the top `starti` and bottom `endi` rows with 1 (background)."""
    for i in range(LL):
        for j in range(LL):
            c[i, j] = 1 if (i < starti or i >= LL - endi) else b[i - starti, j]


def _place_normalized_cols(c, b, starti, endi, LL):
    """Center the resized image `b` horizontally into the 32×32 output `c`, padding
    the left `starti` and right `endi` columns with 1 (background)."""
    for i in range(LL):
        for j in range(LL):
            c[i, j] = 1 if (j < starti or j >= LL - endi) else b[i, j - starti]


def fnormalize(a, c):
    h, w = float(a.shape[0]), float(a.shape[1])
    L = 32.0
    LL = 32

    if h >= w:
        bg = h
        sm = w
        smi = 1
    else:
        bg = w
        sm = h
        smi = 0

    R1 = sm / bg
    R2 = math.sqrt(R1)

    if sm == h:
        H2 = L * R2
        W2 = L
    else:
        H2 = L
        W2 = L * R2

    alpha = W2 / w
    beta = H2 / h
    b = cv2.resize(a, (0, 0), fy=beta, fx=alpha, interpolation=cv2.INTER_CUBIC)
    smn = b.shape[smi]
    df = L - smn

    offset = math.floor(df * 0.5)
    if df % 2 == 1.0:
        start = offset + 1
        end = offset
    else:
        start = end = offset

    starti = int(start)
    endi = int(end)

    if sm == h:
        _place_normalized_rows(c, b, starti, endi, LL)
    else:
        _place_normalized_cols(c, b, starti, endi, LL)
