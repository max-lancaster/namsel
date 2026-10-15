'''A set of utilities for defining and working with namsel-ocr config files
'''

import json
import codecs
import os
import glob
try:
    from .utils import create_unique_id
except ImportError:
    from utils import create_unique_id


# Resolve confs dir relative to this file's location (works in frozen PyInstaller apps)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.join(_MODULE_DIR, 'confs')

if not os.path.exists(CONF_DIR):
    try:
        os.mkdir(CONF_DIR)
    except OSError:
        pass  # Read-only filesystem in frozen app — confs may be pre-bundled

def _open(fl, mode='r'):
    return codecs.open(fl, mode, encoding='utf-8')

default_config = {
    'page_type': 'book',
    'line_break_method': 'line_cut',
    'recognizer': 'probout', # or hmm (changed to probout as default due to better reliability)
    'break_width': 2.0,  # Increased to reduce false spacing in larger images
    'segmenter': 'stochastic', # or experimental
    'combine_hangoff': .6,
    'low_ink': False,
    'line_cluster_pos': 'top', # or center
    'viterbi_postprocessing': False, # determine if main is running using viterbi post processing
    'postprocess': False, # Run viterbi (or possibly some other) post processing
    'stop_line_cut': False,
    'detect_o': False,
    'clear_hr': False,
    'line_cut_inflation': 4, # The number of iterations when dilating text in line cut. Increase this value when need to blob things together
    'debug_output': False, # Enable debug output (warnings, preprocessing messages, etc.)
    'debug_output_dir': '.', # Directory to save debug files and interactive segmentation results
    'enable_interactive_segmentation': False, # Enable interactive manual segmentation for wide characters
    'force_single_line': False, # Skip multiline auto-detection for pre-segmented line images
    'test_simple_normalize': False,  # Test simple 32x32 normalization for scale invariance
    'test_normalized_features': False,  # Test normalized features for scale invariance
    'test_improved_normalize': False,  # Test improved fnormalize algorithm without sqrt
    'compound_syllable_gap_tolerance': 0.15,  # Gap tolerance for compound syllables (ratio of char_mean). Allows small gaps between diacritics and base characters, like ã in Brazilian Portuguese
}

def update_default():
    json.dump(default_config, _open(os.path.join(CONF_DIR, 'default.conf'), 'w'), indent=1)

def create_misc_confs():
    from sklearn.grid_search import ParameterGrid
    params = {'break_width': [1.5, 2.0, 3.6, 5.0], 
              'recognizer': ['probout', 'hmm'], 'combine_hangoff': [.4, .6, .8], 
              'postprocess': [True, False], 'segmenter': ['experimental', 'stochastic'],
              'line_cluster_pos': ['top', 'center'],
              }
    grid = ParameterGrid(params)
    for pr in grid:
        Config(save_conf=True, **pr)


class Config(object):
    def __init__(self, path=None, save_conf=False, **kwargs):


        self.conf = dict(default_config)  # Copy to avoid mutating the global default
        
            
        self.path = path
        if path:
            # Over-write defaults
            self._load_json_set_conf(path)
        
        
        # Set any manually specified config settings
        for k in kwargs:
            self.conf[k] = kwargs[k]
            
        if kwargs and save_conf:
            self._save_conf()
        
        
        # Set conf params as attributes to conf obj
        for k in self.conf:
            if k not in self.__dict__:
                setattr(self, k, self.conf[k])
        
        
    def _load_json_set_conf(self, path):
        try:
            conf = json.load(_open(path))
            for k in conf:
                self.conf[k] = conf[k]
        except IOError:
            print('Error in loading json file at %s. Using default config' % path)
            self.conf = dict(default_config)
    
     
    def _save_conf(self):
        '''Save a conf if it doesn't already exist'''
        
        confs = glob.glob(os.path.join(CONF_DIR, '*.conf'))

        for conf in confs:
            conf = json.load(_open(conf))
            if conf == self.conf:
                return
        else:
            json.dump(self.conf, _open(os.path.join(CONF_DIR, create_unique_id()+'.conf'), 'w'), indent=1)


