#encoding: utf-8

import cPickle as pickle
import cProfile
from classify import load_cls
from classify import label_chars
from classify import write_libsvm_file
from cv2 import GaussianBlur
from feature_extraction import get_zernike_moments, get_hu_moments, \
    extract_features, normalize_and_extract_features
from functools import partial
import glob
from multiprocessing.pool import Pool
import numpy as np
import os
from sklearn.externals import joblib
from sobel_features import sobel_features
from transitions import transition_features
from fast_utils import fnormalize, ftrim

cls = load_cls('logistic-cls')

# Load testing sets
print 'Loading test data'

tsets = pickle.load(open('datasets/testing/training_sets.pkl', 'rb'))

scaler = joblib.load('zernike_scaler-latest')

print 'importing classifier'
#from classify import fast_cls as cls
print cls.get_params()

print 'scoring ...'
keys = tsets.keys()
keys.sort()
all_samples = []

baseline = [0.608, 0.5785123966942148, 0.4782608695652174, 0.7522123893805309, 
            0.6884057971014492, 0.5447154471544715, 0.9752066115702479, 
            0.9830508474576272]


def test_accuracy(t, clsf=None, verbose=True):
    
    if clsf:
        cls = clsf
    else:
        global cls
    
    y = tsets[t][:,0]
    x = tsets[t][:,1:]
    
    x3 = []
    for j in x:
        j = ftrim(j.reshape((32,16)).astype(np.uint8))
        x3.append(normalize_and_extract_features(j))
    
    pred = cls.predict(x3)
#    
    s  = 0
    for i, p in enumerate(pred):
##        print p, i
#        pr = np.argmax(p)
        if float(p) == y[i]:
            s += 1.0            
        else:
            pass
            print 'correct', label_chars[y[i]], '||', label_chars[p], t #, max(cls.predict_proba(x3[i])[0])

    score = s / len(y)
    return score

def test_all(clsf=None, verbose=False):
    print 'starting tests. this will take a moment'
    
    test_accuracy(keys[0], clsf)
    
    test_all = partial(test_accuracy, clsf=clsf)
    p = Pool()
    all_samples = p.map(test_all, keys)
        
    for t, s in zip(keys, all_samples):
        print t, s
    return np.mean(all_samples)

if __name__ == '__main__':
    print test_all(verbose=True)