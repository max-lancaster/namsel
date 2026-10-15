#encoding: utf-8

import os
import sys
import glob
import re
import codecs
from difflib import HtmlDiff
from recognize import run_main
import Levenshtein as L
import requests
import datetime
import multiprocessing
from config_manager import Config, run_all_confs_for_page


LOGIN_URL = 'https://dhattupages.appspot.com/accounts/login/?next=/'
PW = 'dartsedolhagangdege7'
credentials = {'username':'zach', 'password':PW}

HD = HtmlDiff()

test_vols = ['sample_book6', 'sample_book5', 'sample_book4', 'sample_book3', 
             'sample_book2', 'sample_book1', 'ldong-yon-tan-rgya-mtsho', 
             'don-grub-rgyal', 'taranatha']


test_vols.sort()

style_old = '''    <style type="text/css">
        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    </style>'''

style_new = '''    <style type="text/css">
        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    tr {line-height: 40px;}
    td {font-family: "Qomolangma-Uchen Sarchung" !important}
    </style>'''

multiple_spaces = re.compile(r'[ \t]{1,}')

pwd = os.getcwd()

def open(fl, mode):
    return codecs.open(fl, mode, 'utf-8')

def _normalize_input(txt):
    # Strip lines of extra whitespace
    lines = txt.split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    
    # remove top title line
    lines = lines[1:]
    txt = '\n'.join(lines)
    # collapse multiple spaces to 1 space
    txt = multiple_spaces.sub(' ', txt)
    txt = txt.replace('༎', '།།')
    txt = txt.replace('<', '〈')
    txt = txt.replace('>', '〉')
    txt = txt.replace('༑', '་།་')
    txt = txt.replace('-', '—')
    return txt

def _make_html_diff(txt, ocr):    
    html = HD.make_file(txt.split('\n'), ocr.split('\n'))
    html = html.replace(style_old, style_new)
    html = html.replace('ISO-8859-1', 'utf-8')
    html = html.replace('<tbody>\n', '<tbody>\n<tr><td></td><td></td><td>Manual input</td><td></td><td></td><td>OCR</td></tr>\n')
#    print html
    return html

def _get_compare_data(tif_txt_pair):
    tif = tif_txt_pair[0]
    txt = tif_txt_pair[1]
    if tif[:-4] == txt[:-4]: # This should always be true
#         ocr = run_main(tif, conf=Config(path='/home/zr/letters/conf/443cf9ec-76c7-44bc-95ad-593138d2d5fc.conf'), text=True)
#         ocr = run_main(tif, conf=Config(segmenter='stochastic', recognizer='hmm', break_width=3.6), text=True)
        ocr = run_main(tif, text=True)
#         ocr = run_all_confs_for_page(tif, text = True)
        ocr = ocr.strip()
        txt = open(txt,'r').read()
        txt = _normalize_input(txt)
        edit_dist = L.distance(txt, ocr)
        edit_ratio = L.ratio(txt, ocr)
        html = _make_html_diff(txt, ocr)
#        sys.exit()
        data = {'edit_distance': edit_dist,
                'edit_ratio': edit_ratio,
                'filename': os.path.basename(tif), 
                'html': html
            }
    return data

def do_pairwise_comparison(origflpath, ocrflpath):
    o = open(origflpath, 'r').read()
    s = open(ocrflpath, 'r').read()
    s = _normalize_input(s)
    
    return L.ratio(o,s)
    
    
#data = {'csrfmiddlewaretoken':s.cookies['csrftoken'], 
#        'edit_distance': edit_dist, 
#        'filename': os.path.basename(tif), 
#        'sample_set': t, 'html': html, 'timestamp': timestamp,
#        'comment': comment
#    }

if __name__ == '__main__':
        
    from sklearn.externals.joblib import Parallel, delayed
    timestamp = datetime.datetime.now()
    comment = input('Comment: ')
    for t in test_vols:
        os.chdir(os.path.abspath(t))
        
        tifs = glob.glob('*tif')
        txts = glob.glob('*txt')
        
        tifs.sort()
        txts.sort()
        
        pool = multiprocessing.Pool()
    #     all_data = Parallel(n_jobs=12)(delayed(_get_compare_data)(i) for i in zip(tifs, txts))
        all_data = pool.map(_get_compare_data, list(zip(tifs, txts)))
        
#         all_data = []
#         for i in zip(tifs, txts):
#             all_data.append(_get_compare_data(i))
        
    
        with requests.session() as s:
            s.get(LOGIN_URL)
            credentials['csrfmiddlewaretoken'] = s.cookies['csrftoken']
            s.post(LOGIN_URL, data=credentials, 
                   headers={'Referer': 'https://dhattupages.appspot.com/'}, 
                           allow_redirects=True)
            
            print('posting data for ', t)
            for data in all_data:
                data['csrfmiddlewaretoken'] = s.cookies['csrftoken']
                data['sample_set'] = t
                data['timestamp'] = timestamp
                data['comment'] = comment
                
                
                r = s.post('https://dhattupages.appspot.com/test-data-update', 
                       headers={'Referer': 'https://dhattupages.appspot.com/'}, 
                       data=data)
                r.raise_for_status()
                
        
        
        os.chdir(pwd)
