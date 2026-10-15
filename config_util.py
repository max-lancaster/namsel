import uuid
try:
    from .utils import local_file
except ImportError:
    from utils import local_file
import codecs
import json

CONF_FILE = local_file('configs.json')

defaults = {
            'too_small': 7,
            'small_contour_thresh': 2, # times small std dev
            'viterbi_post_process': True,
            'break_threshold': 2.0,
            'hang_off_amount': .4,  
            'segmenter': 'experimental',
            'line_breaker': 'line_cut',
            'detect_special_notation': False,
            'feature_transform': None, # pca pickle or whatever
            }



def save_config(config, path):
    '''Save a single configuration'''
    return json.dump(config, codecs.open(path, 'w', 'utf-8'))
    
def load_config(path):
    '''Load a single configuration'''
    return json.load(codecs.open(path, 'r', 'utf-8'))

def rand_id():
    return str(uuid.uuid4())

def add_new_config(conf):
    all_confs = load_configs()
    if not _is_duplicate_conf(all_confs, conf):
        all_confs[rand_id()] = conf
        save_configs(all_confs)
    return

def _is_duplicate_conf(all_confs, conf):
    keys = list(defaults.keys())
    is_duplicate = False

    for oconf in all_confs:
        if is_duplicate: break
        for key in keys:
            if conf[key] != oconf[key]:
                break
        else:
            is_duplicate = True
    return is_duplicate

def save_configs(confs):
    '''Save collection of configurations (JSON — data-only, not pickle)'''
    return json.dump(confs, codecs.open(CONF_FILE, 'w', 'utf-8'))

def load_configs():
    '''Load collection of configurations (JSON — data-only, not pickle)'''
    return json.load(codecs.open(CONF_FILE, 'r', 'utf-8'))

