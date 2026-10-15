
from __future__ import division
from cv2 import Sobel
cimport numpy as np
cimport cython
from libc.math cimport sqrt, atan2, fmod, M_PI
ctypedef np.intp_t DTYPE_t

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sobel_features(np.ndarray[double, ndim=2] a,
                     np.ndarray[double, ndim=2] magnitude,
                     np.ndarray[double, ndim=2] direction,
                     np.ndarray[double, ndim=2] sx,
                     np.ndarray[double, ndim=2] sy,
                     np.ndarray[DTYPE_t] vector
                     ):

    cdef int imgh = 32
    cdef int imgw = 32
    cdef double o_tsize = 1.0/(imgh*imgw)
    cdef int ix0, ix1, i, ftmat_len, j, sec, cls, cls1, cls2, l
    cdef double rbar, msum, curdeg, dx, dy, mg
    cdef double d_30 = 1.0 / 30.0
    cdef double degc = 180.0 / M_PI
    

    Sobel(a, dst=sx, ddepth=-1, dx=1, dy=0, ksize=3)
    Sobel(a, dst=sy, ddepth=-1, dx=0, dy=1, ksize=3)

    
    for i in range(imgh):
        for j in range(imgw):
            dx = sx[i,j]
            dy = sy[i,j]
            mg = sqrt(dx*dx + dy*dy)
            magnitude[i,j] = mg
            msum += mg
            direction[i,j] = atan2(dy,dx)
            
    
    rbar = msum * o_tsize
    for k in range(imgh):
        for l in range(imgh):
            if magnitude[k,l] >= rbar:
                curdeg =  direction[k,l] * degc
                ix0 = k
                ix1 = l
                if ix1 < 8:
                    j = 0
                elif 8 <= ix1 < 16:
                    j = 1
                elif 16 <= ix1 < 24:
                    j = 2
                else:
                    j = 3
                
                if ix0 < 8:
                    sec = 0 + j
                elif 8 <= ix0 < 16:
                    sec = 4 + j
                elif 16 <= ix0 < 24:
                    sec = 8 + j
                else:
                    sec = 12 +j

#                 if curdeg % 30.0 != 0.0:
                if fmod(curdeg, 30.0) != 0.0:
                    if 0.0 < curdeg < 30.0:
                        cls1 = 0
                        cls2 = 1
                    elif 30.0 < curdeg < 60.0:
                        cls1 = 1
                        cls2 = 2
                    elif 60.0 < curdeg < 90.0:
                        cls1 = 2
                        cls2 = 3
                    elif 90.0 < curdeg < 120.0:
                        cls1 = 3
                        cls2 = 4
                    elif 120.0 < curdeg < 150.0:
                        cls1 = 4
                        cls2 = 5
                    elif 150.0 < curdeg < 180.0:
                        cls1 = 5
                        cls2 = 6
        
                    elif -30.0 < curdeg < 0.0:
                        cls1 = 0
                        cls2 = 11
                    elif -60.0 < curdeg < -30.0:
                        cls1 = 11
                        cls2 = 10
                    elif -90.0 < curdeg < -60.0:
                        cls1 = 10
                        cls2 = 9
                    elif -120.0 < curdeg < -90.0:
                        cls1 = 9
                        cls2 = 8
                    elif -150.0 < curdeg < -120.0:
                        cls1 = 8
                        cls2 = 7
                    elif -180.0 < curdeg < -150.0:
                        cls1 = 7
                        cls2 = 6
                    
                    sec = sec * 12
                    vector[sec+cls1] += 1
                    vector[sec+cls2] += 1
                else:
                    cls = int(curdeg * d_30)
                    sec = sec*12+cls
                    vector[sec] += 1 

