from cv2 import GaussianBlur
from cv2 import HuMoments, moments, GaussianBlur
try:
    from .fast_utils import fnormalize, scale_transform
    from .sobel_features import sobel_features
    from .transitions import transition_features
    from .zernike_moments import zernike_features
    from .utils import local_file
except ImportError:
    from fast_utils import fnormalize, scale_transform
    from sobel_features import sobel_features
    from transitions import transition_features
    from zernike_moments import zernike_features
    from utils import local_file
import numpy as np
import joblib
import os
try:
    from .config_manager import default_config
except ImportError:
    from config_manager import default_config
import cv2


SCALER_PATH = 'zernike_scaler-latest'
scaler_full_path = local_file(SCALER_PATH)
if os.path.exists(scaler_full_path):
    scaler = joblib.load(scaler_full_path)
    transform = scaler.transform
    try:
        sc_o_std = 1.0/scaler.scale_
    except AttributeError:
        sc_o_std = 1.0/scaler.std_
    sc_mean = scaler.mean_
    SCALER_DEFINED = True
else:
    SCALER_DEFINED = False

FEAT_SIZE = 346
hstack = np.hstack

NORM_SIZE = 32
ARR_SHAPE = (NORM_SIZE, NORM_SIZE)
x3 = np.empty(NORM_SIZE*2, dtype=np.uint8)
newarr = np.empty(ARR_SHAPE, dtype=np.uint8)

magnitude = np.empty(ARR_SHAPE, np.double)
direction = np.empty(ARR_SHAPE, np.double)
sx = np.empty(ARR_SHAPE, np.double)
sy = np.empty(ARR_SHAPE, np.double)
# Use np.intp consistently for Cython compatibility
# This matches the DTYPE_t definition in sobel_features.pyx
x2 = np.zeros((192), dtype=np.intp)

# Suppress NumPy warnings during feature loading
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        # numpy .npz (not pickle, not skops) — the three Zernike matrices are plain
        # float64 arrays, so np.load with allow_pickle=False is data-only (nothing can
        # execute on load) AND drags in no third-party deserializer. The previous
        # skops-based load made `import skops` a hard, module-level requirement of the
        # whole engine: when it was absent from a frozen build the ImportError cascaded
        # through line_breaker -> namsel -> daemon and the daemon never started.
        # zernike_features.npz is byte-identical to the former features/*.skops trio.
        _zf = np.load(local_file(os.path.join('features', 'zernike_features.npz')),
                      allow_pickle=False)
        D = _zf['D_matrix']
        Bpqk = _zf['Bpqk17']
        Ipi = _zf['Ipi32']
        _zf.close()
    except Exception as e:
        # Loud, single-line marker: the dummies below keep the engine importable but
        # Zernike features are meaningless, so recognition accuracy collapses. This
        # must never be silent in a shipped build.
        print(f"ERROR: Zernike feature matrices unavailable ({e}) — "
              f"recognition accuracy will be severely degraded.")
        # Create dummy data with correct dimensions
        D = np.eye(100)
        Bpqk = np.zeros((18, 18, 32))
        Ipi = np.zeros((32, 32))

Ipi = np.array(Ipi, Ipi.dtype, order='F')
deg = 17
Mpqs = np.zeros((deg+1, deg+1), np.double, order='F')
Rpq = np.empty((deg+1, deg+1), complex)
ws = np.array([1, -1j, -1, 1j], complex)
Zpq = np.empty((90), np.double)
Yiq = np.zeros((deg+1, NORM_SIZE), np.double, order='F')


def normalize_and_extract_features(arr, debug_char_idx=None, debug_coordinates=None, debug_line=None):
    global newarr, x3, Zpq
    try:
        if not hasattr(arr, 'shape'):
            return None
        newarr = newarr.astype(np.uint8)
        try:
            fnormalize(arr, newarr)
        except Exception:
            # Fallback: copy arr to newarr directly (or bail on shape mismatch).
            if arr.shape == newarr.shape:
                newarr[:] = arr.astype(np.uint8)
            else:
                return None
        result = extract_features(newarr)
        if result is None or not hasattr(result, 'shape'):
            return None
        return result
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _rate_limited_warn(fn, msg, limit):
    """Print msg at most `limit` times, counting on the fn object (spam guard)."""
    fn.warning_count = getattr(fn, 'warning_count', 0) + 1
    if fn.warning_count <= limit:
        print(msg)


def _extract_zernike(arr):
    """Zernike features into the module-level Zpq; zero-fill on dimension mismatch."""
    try:
        zernike_features(arr, D, Bpqk, Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
    except ValueError as e:
        _rate_limited_warn(zernike_features, f"Warning: Zernike feature extraction failed: {e}", 5)
        Zpq.fill(0.0)


def _extract_sobel(arr):
    """Sobel features into module-level x2, retrying with alternative int dtypes on
    a dtype mismatch (np.long was removed in NumPy 1.20+)."""
    x2.fill(0)
    try:
        sobel_features(arr, magnitude, direction, sx, sy, x2)
        return
    except (ValueError, TypeError) as e:
        _rate_limited_warn(sobel_features, f"Warning: Sobel feature extraction failed: {e}", 3)
    alt_dtypes = [np.int64, np.int32, np.intp, np.long if hasattr(np, 'long') else int]
    for alt_dtype in alt_dtypes:
        try:
            x2_temp = np.zeros((192), dtype=alt_dtype)
            sobel_features(arr, magnitude, direction, sx, sy, x2_temp)
            x2[:] = x2_temp[:]
            return
        except (ValueError, TypeError):
            continue
    x2.fill(0.0)


def _maybe_scale(x1):
    """Apply the zernike scaler in place, skipping when it's undefined or the
    feature/scaler dimensions disagree."""
    if not SCALER_DEFINED:
        if not hasattr(extract_features, 'scaler_warning_count'):
            extract_features.scaler_warning_count = 0
        if extract_features.scaler_warning_count < 5:
            print("Warning: Scaler not defined, skipping scaling")
            extract_features.scaler_warning_count += 1
        return
    if len(x1) == len(sc_mean):
        scale_transform(x1, sc_mean, sc_o_std, FEAT_SIZE)


def extract_features(arr, scale=True):  # Re-enable scaling but need to fix corruption
    global x3, Zpq
    transition_features(arr, x3)
    arr = arr.astype(np.double)
    Yiq.fill(0.0)
    _extract_zernike(arr)
    GaussianBlur(arr, ksize=(5, 5), sigmaX=1, dst=newarr)
    _extract_sobel(arr)
    x1 = hstack((Zpq, x2, x3))
    if scale:
        _maybe_scale(x1)
    return x1


def invert_binary_image(arr):
    '''
    Invert a binary image so that zero-pixels are considered as background.
    This is assumed by various functions in OpenCV and other libraries.
    
    Parameters:
    -----------
    arr: 2D numpy array containing only 1s and 0s
    
    Returns:
    --------
    2d inverted array
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, np.uint8)  # Ensure input is a numpy array
    
    if np.max(arr) == 255:
        return (arr / -255) + 1
    else:
        return (arr * -1) + 1


def get_zernike_moments(arr):
    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    
    zernike_features(arr, D, Bpqk, Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
    return Zpq


def get_hu_moments(arr):
    arr = invert_binary_image(arr)
    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    m = moments(arr.astype(np.float64), binaryImage=True)
    hu = HuMoments(m)
    return hu.flatten()
