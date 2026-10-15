"""Pure Python fallback for sobel_features Cython extension."""
import math
from cv2 import Sobel


# Gradient-direction bins: (low, high, (cls1, cls2)) — a direction strictly inside
# a 30° band contributes to both adjacent classes.
_DEGREE_BINS = [
    (0.0, 30.0, (0, 1)), (30.0, 60.0, (1, 2)), (60.0, 90.0, (2, 3)),
    (90.0, 120.0, (3, 4)), (120.0, 150.0, (4, 5)), (150.0, 180.0, (5, 6)),
    (-30.0, 0.0, (0, 11)), (-60.0, -30.0, (11, 10)), (-90.0, -60.0, (10, 9)),
    (-120.0, -90.0, (9, 8)), (-150.0, -120.0, (8, 7)), (-180.0, -150.0, (7, 6)),
]


def _region(ix):
    """4×4 spatial-cell index (0-3) for a 0-31 coordinate."""
    return min(ix // 8, 3)


def _gradient_bins(curdeg):
    """The two adjacent direction classes for a degree strictly inside a band, or None."""
    for lo, hi, bins in _DEGREE_BINS:
        if lo < curdeg < hi:
            return bins
    return None


def _sobel_magnitudes(sx, sy, magnitude, direction, imgh, imgw):
    """Fill magnitude + direction from the Sobel gradients; return the magnitude sum."""
    msum = 0.0
    for i in range(imgh):
        for j in range(imgw):
            dx = sx[i, j]
            dy = sy[i, j]
            mg = math.sqrt(dx * dx + dy * dy)
            magnitude[i, j] = mg
            msum += mg
            direction[i, j] = math.atan2(dy, dx)
    return msum


def sobel_features(a, magnitude, direction, sx, sy, vector):
    imgh = 32
    imgw = 32
    o_tsize = 1.0 / (imgh * imgw)
    d_30 = 1.0 / 30.0
    degc = 180.0 / math.pi

    Sobel(a, dst=sx, ddepth=-1, dx=1, dy=0, ksize=3)
    Sobel(a, dst=sy, ddepth=-1, dx=0, dy=1, ksize=3)

    msum = _sobel_magnitudes(sx, sy, magnitude, direction, imgh, imgw)
    rbar = msum * o_tsize
    for k in range(imgh):
        for l in range(imgh):
            if magnitude[k, l] < rbar:
                continue
            curdeg = direction[k, l] * degc
            sec = _region(k) * 4 + _region(l)   # spatial cell (from ix0=k, ix1=l)
            if math.fmod(curdeg, 30.0) != 0.0:
                bins = _gradient_bins(curdeg)
                if bins is None:
                    continue
                base = sec * 12
                vector[base + bins[0]] += 1
                vector[base + bins[1]] += 1
            else:
                vector[sec * 12 + int(curdeg * d_30)] += 1
