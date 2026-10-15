"""Pure Python fallback for transitions Cython extension."""
import numpy as np


def horizontal_transitions(a):
    """Count white-to-black transitions per row. Returns uint8 array of length imgh."""
    imgh = a.shape[0]
    imgw = a.shape[1]
    result = np.zeros(imgh, dtype=np.uint8)
    for i in range(imgh):
        row = a[i]
        prev = 1
        trs = 0
        j = 1
        for k in range(imgw):
            j = int(row[k])
            if j == 1 and prev == 0:
                trs += 1
            prev = j
        if j == 0:
            trs += 1
        result[i] = min(trs, 255)
    return result


def _count_transitions(seq):
    """Count black(0)→white(1) transitions across `seq`, plus one if it ends black."""
    trs = 0
    prev = 1
    last = 1
    for x in seq:
        v = int(x)
        if v == 1 and prev == 0:
            trs += 1
        prev = last = v
    if last == 0:
        trs += 1
    return trs


def transition_features(a, allv):
    imgh = a.shape[0]
    imgw = a.shape[1]
    b = a.T
    for i in range(imgh):
        allv[i] = _count_transitions(a[i, :imgw])          # horizontal (row i)
        allv[i + imgw] = _count_transitions(b[i, :imgw])   # vertical (column i)
