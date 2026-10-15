#! /usr/bin/env python

from subprocess import Popen
import sys
import glob
import os
from functools import partial

import platform

if platform.system() != "Windows":
    from multiprocessing import Pool, cpu_count

def process_tifs(fl, threshold, layout=1):
    """
    For the external call to scantailor-cli, not that --layout specifies 
    whether an image contains 1 or 2 pages. 0 is auto-detect, 1 is page, etc
    orientation=right means rotate right 90 deg
    """
    
    print(fl)
    out_folder=os.path.join(os.path.dirname(fl), 'out')
    command = ['scantailor-cli',  '--layout={}'.format(layout), \
               '--margins=1','--output-dpi=600', \
               '--threshold={}'.format(threshold), \
               '--despeckle=aggressive' , fl, out_folder]
    p = Popen(command)
    p.wait()
    

def run_scantailor(folder, threshold, layout='single', processes=None):
    '''Run scantailor image processing in batch
    
    Parameters:
    -----------
        folder: a valid folder path (str)
        threshold: the amount of thinning or thickening of the output.
            Good values are -40 to 40 (for thinning and thickening respectively)
    
    Returns:
    --------
        None -- the transformed images are saved in a subfolder of *folder* called "out"
    '''
    tifs = glob.glob(os.path.join(folder, '*tif'))
    TIFS = glob.glob(os.path.join(folder, '*TIF'))
    jpg = glob.glob(os.path.join(folder, '*jpg'))
    JPG = glob.glob(os.path.join(folder, '*JPG'))
    tifs.extend(TIFS)
    tifs.extend(jpg)
    tifs.extend(JPG)
    tifs.sort()
    out_folder = os.path.join(folder, 'out')
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    print('starting to process images at threshold = {}'.format(threshold))
    
    if layout == 'single' or layout is None:
        layout = 1
    else:
        layout = 2
    
    process_tifs2 = partial(process_tifs, threshold=threshold, layout=layout)

    if platform.system() == "Windows":
        list(map(process_tifs2, tifs))
    else:
        if not processes:
            processes = cpu_count()
        p = Pool(processes=processes, maxtasksperchild=30)
        p.map(process_tifs2, tifs)
        p.terminate()
    

if __name__ == '__main__':
    threshold = 0
    try:
        folder = sys.argv[1]
        
        print(folder, 'FOLDER')
        if len(sys.argv) == 3:
            threshold = sys.argv[2]
    except:
        folder = ''


    run_scantailor(folder, threshold)
