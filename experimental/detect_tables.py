'''Functions for detecting tables/boxes on a scanned page image'''

import cv2 as cv
import numpy as np
from PIL import Image
import sys
from collections import OrderedDict

## Most of this logic for identifying rectangles comes from the 
## squares.py sample in opencv source code.
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_boxes(tiff_fl, blur=False):
    im = Image.open(tiff_fl).convert('L')
    a = np.asarray(im)
    if blur:
        a = cv.GaussianBlur(a, (5, 5), 0)
    contours, hierarchy = cv.findContours(a.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    border_boxes = []
#     n = np.ones_like(a)
    for j,cnt in enumerate(contours):
        cnt_len = cv.arcLength(cnt, True)
        orig_cnt = cnt.copy()
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and ((a.shape[0]-3) * (a.shape[1] -3)) > cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if max_cos < 0.1:
                b = cv.boundingRect(orig_cnt)
                x,y,w,h = b
                border_boxes.append(b)
#                 cv.rectangle(n, (x,y), (x+w, y+h), 0)
#                 cv.drawContours(n, [cnt], -1,0, thickness = 5)
#     Image.fromarray(n*255).show()
    return border_boxes

def get_edges(b):
    l = b[0]
    r = b[0] + b[2]
    t = b[1]
    b = b[1] + b[3]
    return (l,r,t,b)

def b_contains_nb(b,nb):
    l1,r1,t1,b1 = get_edges(b)
    l2,r2,t2,b2 = get_edges(nb)
    return l1 <= l2 and r2 <= r1 and t1 <= t2 and b1 >= b2

def organize_boxes(boxes):

    boxes.sort(key=lambda x: x[0], reverse=True)
    tree = OrderedDict((i, {'parent':None,'children':[]}) for i in range(len(boxes)-1,  -1, -1))
    b = boxes[0]
    for i, nb in enumerate(boxes[:-1]):
        for j, b in enumerate(boxes[i+1:]):
            
            jo = i+1 + j
            if b_contains_nb(b, nb):
                tree[jo]['children'].append(i)
                tree[i]['parent'] = jo
                break
    return tree

def DFS(tree, v, func=None):
    tree[v]['discovered'] = True
    print('visited node', v)
#     raw_input()
#     if func:
#         func(tree[v])

    for e in tree[v]['children']:
        if not tree[e].get('discovered'):
            DFS(tree, e)
    return v

def visualize_tree_page(flpath, tree, boxes):
    size = Image.open(flpath).size
    arr = np.ones((size[1], size[0]))
    boxes.sort(key=lambda x: x[0], reverse=True)
#     import ImageDraw
#     import Image
#     im = Image.fromarray(a*255)
#     draw = ImageDraw.Draw(im)
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        print('node', i, 'contains', tree[i]['children'])
        cv.rectangle(arr, (x,y), (x+w, y+h), 0)
        cv.putText(arr, str(i), (x-5, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, 0)

    Image.fromarray(arr*255).show()

def visualize_tree_graphviz(flpath, tree, boxes):
    
    import pygraphviz as pgv
    
    G=pgv.AGraph()
    
    for i in range(len(boxes)-1, -1, -1):
        print('node', i, 'contains', tree[i]['children'])
        for node in tree[i]['children']:
            G.add_edge(str(i), str(node))
        else:
            G.add_node(str(i))
    G.layout('dot')
    G.draw('/tmp/graph.png')

def find_tables_in_volume(vol_dir):
    TABLE_DIR = '/tmp/tables/'
    if os.path.exists(TABLE_DIR):
        shutil.rmtree(TABLE_DIR)
        os.mkdir(TABLE_DIR)
        os.mkdir(os.path.join(TABLE_DIR, 'sections'))
    
    tiffs = glob.glob(os.path.join(vol_dir, '*tif'))
     
    p = multiprocessing.Pool()
    print('detecting boxes')
    now = datetime.datetime.now()
    boxes = p.map(find_boxes, tiffs)
    print('checking for tables i.e. pages w/ > 3 boxes(!?!)')
    for tiff, pboxes in zip(tiffs, boxes):
        if len(pboxes) > 3:
            shutil.copy(tiff, TABLE_DIR)
            pboxes.sort(key=lambda x: x[0], reverse=True)
            tree = organize_boxes(pboxes)
            for k in tree:
                if not tree[k]['parent']:
                    b = pboxes[k]
               
    #         largestarea = 0
    #         largest_box = None
    #         for b in boxes:
    #             area = b[2]*b[3]
    #             if area > largestarea:
    #                 largest_box = b
    #                 largestarea = area
    #         if largest_box:
                    im = Image.open(tiff).convert('L')
                    a = np.asarray(im)
                    x,y,w,h = b
                    section = a[y:y+h,x:x+w]
                    section = np.array(section)
                    Image.fromarray(section).save(os.path.join(TABLE_DIR, 'sections/x%d_y%d_w%d_h%d.tif' % b ))

    
    print(datetime.datetime.now() - now)
    

def traverse_page_nodes(tree, func=None):
    
    for k in tree:
        if not tree[k]['parent']:
#             no_parent.append(k)
            print('New tree traversal')
            DFS(tree, k, func=func)

def find_tables_in_dirtree(root_dir):
    '''there are bugs in this code.. still not working properly...use find_tables_in_volume for more reliable results'''
    os.chdir(root_dir)
    TABLE_DIR = '/tmp/tables/'
    if os.path.exists(TABLE_DIR):
        shutil.rmtree(TABLE_DIR)
        os.mkdir(TABLE_DIR)
    objs = []
    p = multiprocessing.Pool()
    print('walking directory', root_dir)
    for a,b,c in os.walk(root_dir):
        if a.endswith('cache'): # ignore scantailor cache folders
            continue
        for f in c:
            if f.endswith('tif'):
                objs.append([a, os.path.relpath(a), f, p.apply_async(find_boxes, [os.path.join(a, f)])])

    resl = [False]
    loop = 0
    while not all(resl):
        resl = [i[-1].ready() for i in objs]
        if loop % 25 == 0:
            sys.stdout.write('\r%f %% finished' %  (100*resl.count(True)/float(len(resl))))
            sys.stdout.flush()
        loop+=1
        continue
    
    print('copying found pages to temporary dir for review')
    for fulparpath, relpath, fl, res in objs:
        try:
            res = res.get()
        except:
            print(fl)
            continue
        if len(res) > 2:
            path = os.path.join(TABLE_DIR, relpath)
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
                try:
                    shutil.copy(os.path.join(fulparpath, fl), path)
                except:
                    print(fulparpath, fl)

if __name__ == '__main__':
    import multiprocessing
    import glob
    import shutil
    import os
    import datetime
    from subprocess import call
    flpath = '/ngbtiffs/ngb_vol07/out_0007.tif'
    boxes = find_boxes(flpath, blur=False)
    tree = organize_boxes(boxes)
    visualize_tree_page(flpath, tree, boxes)
    visualize_tree_graphviz(flpath, tree, boxes)
    call(['display', '/tmp/graph.png'])
    traverse_page_nodes(tree)
