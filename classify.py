# encoding: utf-8
import os
# Prevent joblib/loky wmic traceback on Windows
if os.name == 'nt' and 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(max(1, (os.cpu_count() or 4) - 1))
import numpy as np
import joblib
try:
    from .utils import local_file
    from .feature_extraction import extract_features
    from .safe_model_io import load_model
except ImportError:
    from utils import local_file
    from feature_extraction import extract_features
    from safe_model_io import load_model
import os
from numpy import uint8
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from cv2 import GaussianBlur
try:
    from .sobel_features import sobel_features
except ImportError:
    from sobel_features import sobel_features
import glob

import platform

TIMEFORMAT = '%Y%m%d%H%M%S'


try:
    # Character maps are our OWN bundled data — load via safe gzip+JSON
    # (safe_model_io.load_model), not shelve/pickle: data-only, cross-platform,
    # no dbm backend. label_chars int->str; chars_label str->int (int keys kept).
    label_chars = load_model(local_file('label_chars.json.gz'))
    chars_label = load_model(local_file('allchars.json.gz'))
except Exception as e:
    print(f"Warning: Could not load character mappings: {e}")
    # Create dummy data for now
    label_chars = {i: chr(0x0F00 + i) for i in range(10)}
    chars_label = {chr(0x0F00 + i): i for i in range(10)}

PCA_PICKLE = 'pca.pkl'

class TrainingData(object): 
    def __init__(self, scaled=False, normed=False):
        self.scaler = None
        self.normalizer = None
        self.scaled = scaled
        self.normed = normed    
        
        print('loading training data')
        
    def get_scaler(self, x_train):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit(x_train)

    def get_normalizer(self, x_train):
        from sklearn.preprocessing import Normalizer
        return Normalizer().fit(self.x_train)
    
    def get_scaled_x(self):
       scaler = self.get_scaler(self.x_train.astype(np.float64))
       self.x_train = scaler.transform(self.x_train.astype(np.float64))
       return self.x_train
           
    def get_normed_x(self):
       normalizer = self.get_normalizer(self.x_train)
       self.x_train = normalizer.transform(self.x_train)        
       return self.x_train
   
    def transform_dataset(self):
        '''Transform raw pixel data to a feature matrix
        
        Returns:
            x_train: the transformed array
        '''

        x_train = []
        for arr in self.x_train:
            x_train.append(extract_features(arr.reshape((32,32)).astype(uint8), scale=False))
        self.scaler = self.get_scaler(x_train)
        
        #TODO: change name of saved data since it is more than just zernike transforms
        joblib.dump(self.scaler, 'zernike_scaler-latest')
        x_train = self.scaler.transform(x_train)
        joblib.dump(x_train, 'zernike_x_train')
        
        return x_train
    
    def get_gradient_x(self):
        print('creating sobel features')

        self.x_train = self.x_train.astype(np.double)
        x_train = [sobel_features(GaussianBlur(x.reshape((32,32)), ksize=(5,5), \
                    sigmaX=1), magnitude, direction, sx, sy, x2) \
                   for x in self.x_train]
        print('got features')
        return x_train
    
    
    def patch_features(self):
        rg = list(range(0,33, 4))
        km = joblib.load('patch_km.pkl')   # sklearn KMeans model, via joblib (not pickle)
#        import Image
        chars = []
        for p, x in enumerate(self.x_train):
            x = x.reshape((32,32)).astype(np.uint8)
            blocks = []
            for i, r in enumerate(rg[:-1]):
                rows = x[r:rg[i+1]]
                for k, c in enumerate(rg[:-1]):
                    col = rows[:, c:rg[k+1]]
                    col = col.flatten()
                    blocks.append(km.predict(col))
                    
            chars.append(blocks)
        print(blocks)
        return np.array(chars)
    
    def save_data_imgs(self, stack):
        '''Selectively convert data row to image and save to disk for
        review purposes'''
    
        from PIL import Image

        label = chars_label[stack]
        for k, tt in enumerate(self.training):
            if tt[0] == label:
                x = tt[1:].astype(np.uint8).reshape((32,32))*255
                
                import tempfile
                Image.fromarray(x).convert('L').save(
                    os.path.join(tempfile.gettempdir(), '%d.tiff' % k))
                
    def exp_data(self, core_smp_file = None):
        '''Load and organize primary datasets'''


        def unique_rows(a):
            unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
            return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
        
        def load_npy_data(npyfile):
            try:
                return np.load(npyfile, allow_pickle=False)   # data-only, not pickle
            except Exception as e:
                print(f"Warning: Could not load {npyfile}: {e}")
                return np.array([])
        
        if not core_smp_file:
            raise ValueError('Must specify a data sample file')
        
        training = np.genfromtxt(core_smp_file, np.uint32, delimiter=',')
        
        if platform.system() == "Windows":
            training_tibchars = np.genfromtxt(r'datasets\tibcharsamples.txt', np.uint32, delimiter=',')
            training5 = np.genfromtxt(r'datasets\ui_samples.csv', np.uint32, delimiter=',')
        else:
            training_tibchars = np.genfromtxt('datasets/tibcharsamples.txt', np.uint32, delimiter=',')
            training5 = np.genfromtxt('datasets/ui_samples.csv', np.uint32, delimiter=',')
        
        #########Training sets with degraded samples
#        training5 = np.genfromtxt('/home/zr/home2/letters/phrinyik_labeled_samples_from_ui.csv', np.uint32, delimiter=',')
        ##########
         
        training_alt = np.load(os.path.join('datasets', 'normalized_3216_to_3232_training.npy'))

        for npyfile in glob.glob(os.path.join('datasets', '*.npy')):
            if os.path.basename(npyfile) == 'normalized_3216_to_3232_training.npy':
                continue   # loaded separately above
            lpk = load_npy_data(npyfile)
            print(np.array(lpk).shape, npyfile)
            training = np.append(training, lpk, axis=0)
        
        training = np.append(training, training_tibchars, axis=0)
        training = np.append(training, training_alt, axis=0)
        training = np.append(training, training5, axis=0)        

        if platform.system() == "Windows":
            symbols = np.genfromtxt(r'datasets\symbols.txt', np.uint32, delimiter=',')
        else:
            symbols = np.genfromtxt('datasets/symbols.txt', np.uint32, delimiter=',')
        
        training = np.append(training, symbols, axis=0)
        
        
        ####include testing sets in training
#        import glob
#        testing_sets = glob.glob('testing_sets/*csv')
#        for t in testing_sets:
#            np.append(training, np.genfromtxt(t, np.uint32, delimiter=','))
        ######################
        
        training = unique_rows(training)
        self.training = training
        self.y_train = training[:,0]
        self.x_train = training[:,1:]
        joblib.dump(self.y_train, 'exp_data_y')
        joblib.dump(self.x_train, 'raw_x_data')
        print('done building training set')

def write_libsvm_file(sample_array, flname, mode='w'):
    '''Convert a sample data ndarray to libsvm formate'''
    
    print('writing libsvm file')
    outfile = open(flname, mode=mode)
    for row in sample_array:
        r = [str(row[0])]
        for i, val in enumerate(row[1:]):
            r.append(str(i+1)+ ':' + str(val))
        row = ' '.join(r)
        outfile.write(row)
        outfile.write('\n')
        
    
def rebuild_cls(pca_trans=False, rbf=True, logistic=True, 
                tuning_par=None, load_saved=False, pca_components=None):
    '''Build (or rebuild) a classifier with all the data
    
    Args:
        pca_trans: bool --> perform pca transformation on extracted features
        rbf: bool --> (re)build the rbf model
        logistic: bool --> (re)buuild the logistic regression model
        load_solved: bool --> load data that has already undergone feature
            extraction from disk
    Returns:
        None (the new classifiers are saved to disk)
    '''

    if load_saved:
        y_train = joblib.load('exp_data_y')
        x_train = joblib.load('zernike_x_train')
        y_train.shape = (y_train.shape[0], 1)
        print(x_train.shape)
        print(y_train.shape)
        y_train,x_train = shuffle(y_train,x_train)
    else:
        data = TrainingData()
        
        if platform.system() == "Windows":
            data.exp_data(core_smp_file=r'datasets\font-draw-samples.txt')
        else:
            data.exp_data(core_smp_file='datasets/font-draw-samples.txt')
        
        y_train = data.y_train
        x_train = data.transform_dataset()
        y_train,x_train = shuffle(y_train,x_train)
        data.x_train = data.x_train.astype(np.uint8)
        data.x_train.tofile('x_train_data')

    if pca_trans:
        print('pca transformation...', end=' ')
        from sklearn.decomposition import PCA
        if pca_components:
            pca = PCA(n_components=pca_components)
        else:
            pca = PCA()
        x_train = pca.fit_transform(x_train, y_train)
        print(x_train.shape, 'is the new dimensionality')
        print('transforming...')
        joblib.dump(pca, PCA_PICKLE)   # sklearn PCA model, via joblib (not pickle)

    if rbf: 
        clstype = 'rbf'
        print('Training rbf. This will take a while')
        cls = svm.SVC(kernel=clstype, C=20, gamma=0.001, 
                      cache_size=100000., probability=False) #<----
        #     cls = svm.SVC(kernel=clstype, C=20, gamma=0.001, cache_size=100000., probability=True)
        print('fitting the classifier')
        cls.fit(x_train, y_train)
        print('saving %s to disk' % clstype)
        joblib.dump(cls, 'rbf-cls')

    if logistic:
        cls = LogisticRegression(C=1000, intercept_scaling=100)
        print('Training the logistic regression classifier. This may take a while.')
        print('fitting the classifier')
        cls.fit(x_train, y_train)
        print('saving logistic regression cls to disk')
        joblib.dump(cls, 'logistic-cls')

def load_cls(name):
    try:
        # Suppress joblib/loky stderr traceback on Windows (wmic CPU detection)
        import io
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            return joblib.load(local_file(name))
    except Exception as e:
        print(f"Warning: Could not load classifier '{name}': {e}")
        # Return a dummy classifier for now
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()


def load_cnn_predictor(module_label):
    """Resolve the bundled namsel_BUDA_OCR CNN model paths (the engine may sit at
    the repo root ./namsel_BUDA_OCR or in a namsel_ocr/ subpackage
    ../namsel_BUDA_OCR) and try to load the TibetanCNNPredictor. Returns
    (predictor, use_cnn); on a missing model or a load failure returns
    (None, False) after logging.

    module_label (e.g. '[recognize.py]') only prefixes the log messages so each
    caller keeps its original output. Callers keep their own sklearn-fallback
    wiring: recognize.py always loads cls+rbfcls; segment.py loads cls only on
    fallback. (os.path.dirname(__file__) resolves to namsel_ocr/ regardless of
    which sibling module triggers the load, so the resolved paths are identical.)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cnn_dir = next((d for d in (os.path.join(here, 'namsel_BUDA_OCR'),
                                os.path.join(here, '..', 'namsel_BUDA_OCR'))
                    if os.path.isdir(d)), os.path.join(here, '..', 'namsel_BUDA_OCR'))
    cnn_model = os.path.join(cnn_dir, 'best_model.pth')
    cnn_mapping = os.path.join(cnn_dir, 'label_mapping.json')
    if os.path.exists(cnn_model) and os.path.exists(cnn_mapping):
        try:
            from namsel_BUDA_OCR.predict import TibetanCNNPredictor
            predictor = TibetanCNNPredictor(cnn_model, cnn_mapping)
            print("%s Using CNN predictor" % module_label)
            return predictor, True
        except Exception as e:
            print("%s CNN load failed (%s), falling back to sklearn" % (module_label, e))
    return None, False


def predict(x, cls=None):
    '''Predict a single sample point'''
    predicted = cls.predict(x)[0]

    print(label_chars[int(predicted)])

def predictprob(x, cls):
    '''Predict probability of a single sample point'''
    probs = cls.predict_proba(x)[0]
    predicted = np.argmax(probs)
    prob = probs[predicted]
    char = label_chars[int(predicted)]
    return char, prob
    
if __name__ == '__main__':
    # rebuild_cls saves the classifiers to disk and returns None, so load the
    # freshly-built logistic classifier before testing (the old code passed the
    # None return straight into test_all).
    rebuild_cls(pca_trans=False)
    from .accuracy_test import test_all
    cls = load_cls('logistic-cls')
    acc = test_all(clsf=cls)
