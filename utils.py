from numpy import vstack, hstack, ones
import numpy as np
from cv2 import findContours, boundingRect, RETR_TREE, CHAIN_APPROX_SIMPLE, resize, INTER_CUBIC
from cv2 import adaptiveThreshold, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV
import os

import uuid
from secrets import choice

interp = INTER_CUBIC


def adaptive_binary_inv(img_arr, bootstrap_scale, single_line=False):
    '''Precise float->uint8 conversion (the old plain cast merged tshegs into main
    characters) + a scale-adaptive odd block size / C, then an inverted Gaussian
    adaptive threshold. Shared by ContourProcessor._prep_binary and
    ScaleCalculator's preliminary-scale bootstrap. single_line: a pre-segmented line
    (force_single_line), whose C stays fixed.'''
    img_copy = img_arr.copy()
    if img_copy.dtype == np.float64 and img_copy.max() <= 1.0:
        img_copy = (img_copy * 255.0).round().astype(np.uint8)
    elif img_copy.dtype != np.uint8:
        img_copy = img_copy.astype(np.uint8)
    block_size = max(3, int(11 * bootstrap_scale))
    # C is a gray-level offset: for a single line it does not grow with the text. Grown with the line's
    # internal enlargement (1.2x -> C 3, 1.5x -> C 4) it erased the anti-aliased edges of tshegs and whole
    # lines lost them. Pages keep the scaled C (the experimental segmenter's golden reads worse without it).
    c_param = 3 if single_line else max(1, int(3.5 * bootstrap_scale))
    if block_size % 2 == 0:
        block_size += 1  # OpenCV requires an odd block size
    return adaptiveThreshold(img_copy, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                             THRESH_BINARY_INV, block_size, c_param)

urlsafechars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
def random_seq(length=15):
    return ''.join([choice(urlsafechars) for i in range(length)])

def normalize(a):
    '''Normalize raw character image array data into 32x32 matrix with an
    aspect ratio equal to the sqrt of the original aspect ratio.
    
    Parameters:
    ----------
    
    a: numpy 2d array
    
    Returns:
    --------
    Normalized 2d numpy array
    '''
    a = a.astype(np.uint8)
    h, w = a.shape
    h = float(h)
    w = float(w)
    L = 32
    sm = np.argmin([h,w])
    bg = np.argmax([h,w])
    R1 = [h,w][sm]/[h,w][bg]

    R2 = np.sqrt(R1) 

        
    if sm == 0:
        H2 = L*R2
        W2 = L
    else:
        H2 = L
        W2 = L*R2
    
    alpha = W2 / w
    beta = H2 / h

    a = resize(a, (0,0), fy=beta, fx=alpha, interpolation=interp)

    smn = a.shape[sm]
    offset = np.floor((L - smn) / 2.)
    c = np.ones((L,L), dtype=np.uint8)

    if (L - smn) % 2 == 1:
        start = offset+1
        end = offset
    else:
        start = end = offset
        
    if sm == 0:
#            print c[start:L-end, :].shape, a.shape
        c[start:L-end, :] = a
    else:
#            print c[:,start:L-end].shape, a.shape
        c[:,start:L-end] = a

    return c


def check_for_overlap(box1, box2, thresh = .77):
    if box1[0] == -1 or box2[0] == -1:
        return False
    
    x,y,w,h = box1[:4]
    xx,yy,ww,hh = box2[:4]
    r = x + w
    rr = xx + ww
    
    # Calculate gap between boxes for compound syllable tolerance
    gap_size = max(0, max(x - rr, xx - r))
    
    # Import config for gap tolerance (avoid circular import)
    try:
        from .config_manager import default_config
        gap_tolerance_ratio = default_config.get('compound_syllable_gap_tolerance', 0.15)
        # Use average character width as reference (approximate)
        char_mean = (w + ww) / 2.0
        gap_tolerance = char_mean * gap_tolerance_ratio
    except ImportError:
        # Fallback if config not available
        gap_tolerance = min(w, ww) * 0.15
    
    # If gap is within tolerance, treat as overlapping for compound syllables
    if gap_size <= gap_tolerance:
        overlap = 1.0  # Force overlap for components with small gaps
    else:
        # Use original overlap calculation for larger gaps
        overlap = float(max(rr,r) - min(x, xx) - abs(rr-r) - abs(xx-x))/float(min(w, ww))
    
    if overlap >= thresh:
        return True
    return False

def add_padding(arr, padding=3):
    '''Add padding to an array to avoid problems with contour extraction
    including the image edges as a contour.
    
    Arguments: arr - the array to be padded, padding - padding amount in pixels
    '''
    
    arr = vstack((ones((padding, arr.shape[1]), dtype=arr.dtype), arr))
    arr = vstack((arr, ones((padding, arr.shape[1]), dtype=arr.dtype)))
    arr = hstack((ones((arr.shape[0],padding), dtype=arr.dtype), arr))
    arr = hstack((arr, ones((arr.shape[0],padding), dtype=arr.dtype)))
    return arr

def _first_nonfull(rows_iter):
    """First index whose row is not all-white (i.e. has ink), or None."""
    for i, row in enumerate(rows_iter):
        if not row.all():
            return i
    return None


def _last_nonfull(rows, start):
    """Highest index i in [start..1] whose rows[i] is not all-white, or None."""
    for i in range(start, 0, -1):
        if not rows[i].all():
            return i
    return None


def _trim_vertical(arr, sides, bottom):
    top, oft, ofb = 0, 0, 0
    if 't' in sides:
        r = _first_nonfull(arr)
        if r is not None:
            top = oft = r
    if 'b' in sides:
        r = _last_nonfull(arr, bottom)
        if r is not None:
            ofb = -(bottom - r)
            bottom = r
    return top, bottom, oft, ofb


def _trim_horizontal(arr, sides, right):
    t = arr.transpose()
    left, ofl, ofr = 0, 0, 0
    if 'l' in sides:
        c = _first_nonfull(t)
        if c is not None:
            left = ofl = c
    if 'r' in sides:
        c = _last_nonfull(t, right - 1)
        if c is not None:
            ofr = -(right - c)
            right = c
    return left, right, ofl, ofr


def trim(arr, sides='trbl', new_offset=False):
    '''Remove empty white space from the edges of a matrix
    '''
    top, bottom, oft, ofb = _trim_vertical(arr, sides, len(arr) - 1)
    left, right, ofl, ofr = _trim_horizontal(arr, sides, arr.shape[1])
    if not new_offset:
        return arr[top:bottom, left:right]
    return arr[top:bottom, left:right], {'top': oft, 'bottom': ofb, 'right': ofr, 'left': ofl}


def local_file(local_file_name):
    return os.path.join(os.path.dirname(__file__), local_file_name)

def invert_bw(arr):
    '''
    Invert black and white

    '''
    arr = arr.copy()

    # `1 - arr` instead of `arr*-1 + 1`: identical for a 0/1 or 0..1 image, but
    # `arr*-1` raises OverflowError on numpy >= 2.0 when arr is a uint8 array
    # (the scalar -1 is out of uint8 range).
    return (1 - arr).astype(np.uint8)

def create_unique_id():
    return str(uuid.uuid4())

def clear_area_in_boxes(arr, boxes):
    for b in boxes:
        x,y,w,h = b
        arr[y:y+h,x:x+w] = 1
    return arr

def remove_small_contours(arr, wthresh=5, hthresh=5):
    area = arr.size
    arr = add_padding(arr)
    
    # Handle different OpenCV versions that return different numbers of values
    contour_result = findContours(arr.copy(), mode=RETR_TREE, method=CHAIN_APPROX_SIMPLE)
    if len(contour_result) == 2:
        contours, hier = contour_result
    else:
        _, contours, hier = contour_result
    rects = [boundingRect(c) for c in contours]
    
    for rect in rects:
        if rect[2] <= wthresh and rect[3] <= hthresh:
            arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1
    return arr


