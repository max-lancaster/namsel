import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange

DTYPE = np.uint8
ctypedef np.intp_t DTYPE_t

#@cython.wraparound(False) # this breaks things sometimes
# cpdef transition_features(np.ndarray[np.uint8_t, ndim=2] a not None):
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef transition_features(np.uint8_t[:,:] a, np.ndarray[np.uint8_t] allv):
    cdef int imgh = a.shape[0]
    cdef int imgw = a.shape[1]
    cdef int row2, col1, trs_v, trs_h, prev_h, prev_v, j, i, totale
    cdef np.uint8_t [:,:] b = a.T

    for i in range(imgh):
        trs_h = 0
        trs_v = 0
        prev_h = 1
        prev_v = 1
        for j in range(imgw):
            col1 = a[i,j]
            row2 = b[i,j]
            if col1 == 1 and prev_h == 0:
                trs_h = trs_h + 1
            prev_h = col1
            if row2 == 1 and prev_v == 0:
                trs_v = trs_v + 1
            prev_v = row2
                    
        if col1 == 0: # If the last pixel in the row is black, count it as a run
            trs_h = trs_h + 1
        if row2 == 0:
            trs_v = trs_v + 1
        allv[i] = trs_h
        allv[i+imgw] = trs_v
    


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
#@cython.wraparound(False) # this breaks things sometimes
def horizontal_transitions(np.ndarray[np.uint8_t, ndim=2] a not None):
    cdef int imgh = a.shape[0]
    cdef int imgw = a.shape[1]
    cdef unsigned int i,  trs, k, j, prev
    cdef np.ndarray[np.uint8_t, ndim=1] row
    cdef np.ndarray[np.uint8_t, ndim=1] all = np.zeros(imgh, dtype=DTYPE)
    
    # for efficiency, assume imgw = imgh, i.e. 32x32
    for i in range(imgh):
        row = a[i]
        prev = 1
        trs = 0

        for k in range(imgw):
            j = row[k]
            if j == 1 and prev == 0:
                trs = trs + 1
            prev = j
                    
        if j == 0: # If the last pixel in the row is black, count it as a run
            trs += 1

        all[i] = trs
    
    return all


