# encoding: utf-8

import numpy as np
from sklearn.mixture import GMM
from collections import Counter
from scipy.stats import mode as statsmode
from yik import alphabet
from sklearn.mixture.dpgmm import DPGMM
import json
import zlib
import re
import sys
import webbrowser
import multiprocessing

alphabet = set(alphabet)

def num_stacks(chars):
    return len(alphabet.intersection(chars))

def get_gmm_for_stack(vals, thresh=10, num_sizes=2):
    '''  Assume no more than num_sizes font sizes present... '''
    if len(vals) > 30:
#         vals = [v for v in vals if vals.count(v) > 2]
#         counts = Counter(vals)
#         vals = [v for v in counts if counts[v] > 2]
        n_components = num_sizes
    elif len(vals) == 1:
        gmm = GMM()
        gmm.means_ = np.array(vals).reshape((1,1))
#         gmm.labels_ = np.array([0])
        return gmm
    else:
        n_components = 1
        
    gmm = GMM(n_components=2)
    gmm.fit(np.array(vals).reshape(len(vals), 1))    
    return gmm
    
#     while True:
#         gmm = GMM(n_components=n_components)
#         try:
#             gmm.fit(vals)
#         except:
#             print vals, n_components
#             raise
# #         gmm.labels_ = np.argsort(gmm.means_)
#         means = list(gmm.means_.copy().flatten())
#         means.sort()
#         if n_components > 1:
#             for i, m in enumerate(means[:-1]):
#                 if means[i+1] - m < thresh or not gmm.converged_:
#  
#                     n_components -= 1
#                     break
#             else:
#                 return gmm
#         elif n_components == 1:
#             return gmm
                
def big_or_small(gmm, width, stack_mean):
    order = np.argsort(gmm.means_.copy().flatten())
#     print order
#     if len(order) == 1:
#         if width - stack_mean > 10:
#             return 'b'
#         else:
#             return 's' #... and what about medium?
        #FIXME: add setting for medium?
#     label = gmm.predict([width])
    label = np.argmax(gmm.predict_proba([width]))
    
#     print
#     print width, gmm.means_.flatten(), gmm.predict_proba([width]), label, order[label]
    pos = order[label]
#     print label, 'LABEL --------<<<'
#     for k, ix in enumerate(order):
# #         ixlabel = gmm.labels_[ix]
#         if ix == label:
#             if k == 0:
#                 return 's'
#             elif k == 1:
#                 if len(order) == 2:
#                     return 'b'
#                 else:
#                     return 'm'
#             else:
#                 return 'b'
    if pos == 0:
        return 's'
    elif pos == 1:
        if len(order) == 2:
            return 'b'
        else:
            return 'm'
    else:
        return 'b'

def granular_smooth(all_lines):
    
    '''this is broken'''
    
    second_pass = []
    punc = set('་། ')
    def reset_chars(s):
        r = set(' )\t')
        if s in r:
            reset = True
        else:
            reset = False
        return reset
    
    
    all_lines = list(zip(list(range(len(all_lines))), all_lines))
    all_lines = list(map(list, all_lines))
    spass_items = []
    just_punc = []
    for i in all_lines:
        if i[1] not in '་།':
            spass_items.append(i)
        else:
            just_punc.append(i)
            
    prev_size = all_lines[0][1]
    ## Iterate through all conditions that would be reason to smooth over
    ## the size of a stack using the previous (and next) size
    for i, s in enumerate(spass_items[:-1]):
        prev = [k[1] for k in spass_items[max(0,i-2):i]]
        nxt = [k[1] for k in spass_items[i+1:min(i+3, len(all_lines))]]
        
        if prev:
            if prev[-1] in ' )':
                reset_before_change_b = True
            elif prev[-1] != s[1]:
                reset_before_change_b = False
        else:
            reset_before_change_b = True
        if nxt:
            if nxt[0] == ' ':
                reset_before_change_f = True
            elif nxt[0] != s[1] :
                reset_before_change_f = False
        else:
            reset_before_change_f = True
        
        if s[1] in punc:
            s[1] = prev_size
        elif not reset_before_change_b and not reset_before_change_f:
            s[1] = prev[-1]
        
        second_pass.append(s)
        prev_size = s[1]
    s = all_lines[-1]
    if s[1] in punc:
        s[1] = prev_size
    elif s[1] != spass_items[-2][1] and spass_items[-2][1] not in ' )':
        s[1] = spass_items[-2][1]
    second_pass.append(s)
    
    second_pass.extend(just_punc)
    second_pass.sort()
    final_pass = []
    for i in second_pass:
        if i[1] in punc:
            final_pass.append(prev_size)
        else:
            final_pass.append(i[1])
            prev_size = i[1]
        
#     if all_lines[-2] not in u' )':
        
        
#         for j in range(3): # look ahead
#             fut_char = all_lines[i+j+1]
#             prev_char = all_lines[i-j-1]
# #             if fut_char in u'་།' or fut_char != ' ':
# #                 reset = False
# #             else:
# #                 reset = True
#             if prev_char in u'་།' or prev_char != ' ':
#                 reset = False
#             else:
#                 reset = True
#         
        
#         if i + 2 < len(all_lines):
#             if s in punc:
#                 s = prev_size
#                 reset = reset_chars(s)
#             # There was a change, then punc, then change back --> smooth over
#             elif not reset and prev_size != s and (all_lines[i+1] in punc and all_lines[i+1] != ' ' and all_lines[i+2] != s) :
#                 s = prev_size
#                 reset = reset_chars(s)
#             # There was a change, then change back --> smooth over
#             elif not reset and prev_size != s and all_lines[i+1] != s :
#                 s = prev_size
#                 reset = reset_chars(s)
#         
#         elif i + 1 < len(all_lines):
#             if s in punc:
#                 s = prev_size
#                 reset = reset_chars(s)
#            # There was a change and then abrupt change back --> smooth over
#             elif not reset and prev_size != s and all_lines[i+1] != s :
#                 s = prev_size
#                 reset = reset_chars(s)
#         elif i == len(all_lines) - 1:
#             if s != prev_size and not reset:
#                 s = prev_size
#                 reset = reset_chars(s)

#         if s not in punc:
#             prev_size = s
#         second_pass.append(s)
    return final_pass
    
def line_smooth(all_lines):
    smoothed = []
    cnts = Counter()
    list(map(cnts.update, all_lines))
    if cnts['b'] > cnts['s']:
        sizemode = 'b'
    else:
        sizemode = 's'
    for line in all_lines:
        for s in _majority_smooth(line, sizemode):
            smoothed.append(s)
    return smoothed

def _majority_smooth(cur_seg, sizemode):
    scount = cur_seg.count('s')
    bcount = cur_seg.count('b')
    if scount > bcount:
        dom_size = 's'
    elif bcount > scount:
        dom_size = 'b'
    else:
        dom_size = sizemode
        
    for c in cur_seg:
        yield dom_size
    

def segment_smooth(all_lines):
    
    
    cur_seg = []
    smoothed_sizes = []
    tbcount = all_lines.count('b')
    tscount = all_lines.count('s')
    if tbcount > tscount:
        sizemode = 'b'
    else:
        sizemode = 's'
            
    for j, s in enumerate(all_lines):
        if s in '། ()〈〉༽༼༔༑':
#             print 'val is', s,
            if s in '། )〉༽༔༑':
                cur_seg.append(s)
#                 print 'segment added', s
#                 print
            
            for sz in _majority_smooth(cur_seg, sizemode):
                smoothed_sizes.append(sz)
            
            if s in '(〈༼':
                cur_seg = [s]
#                 print 'segment added', s
#                 print
            else:
                cur_seg = []
        else:
            cur_seg.append(s)
#         print j, len(smoothed_sizes) + len(cur_seg) - 2, 'CUR S', s
    
    

    for sz in _majority_smooth(cur_seg, sizemode):
        smoothed_sizes.append(sz)

    assert len(smoothed_sizes) == len(all_lines), 'Length of input does not equal length of output. %d != %d' % (len(smoothed_sizes), len(all_lines))
    return smoothed_sizes
            

def assign_sizes_for_page(content, stack_inx_pointer, stacks_gmms, stack_mean, prev_size='', smooth_method=''):
    all_lines = []
    
    if not content:
        return None, all_lines 
    # First pass
    for line in content:
#         prev_size = 'b'
        cur_line = []
        for s in line:
            if s[-1] in '་། ()〈〉༽༼༔༑' or num_stacks(s[-1]) > 1 :   
#                 size = prev_size
                if smooth_method == 'line_smooth':
                    cur_line.append(s[-1])
                else:
                    all_lines.append(s[-1])
                continue
#             print s
            try:
#                 size = big_or_small(stacks_gmms[s[-1]], s[2], stack_mean)
                size = big_or_small(stacks_gmms[stack_inx_pointer[s[-1]]], s[2], stack_mean)
            except KeyError:
                if smooth_method == 'line_smooth':
                    cur_line.append(s[-1])
                else:
                    all_lines.append(s[-1])
#                 print 'KEY ERROR', s[-1]
                continue

#             prev_size = size
#             assert size in 'bs', 'fatal error in first assignment pass'
            if smooth_method == 'line_smooth':
                cur_line.append(size)
            else:
                all_lines.append(size)
        if smooth_method == 'line_smooth':
            all_lines.append(cur_line)
            cur_line = []
            
#             all_lines.append(size)
#         all_lines.append(cur_line)
#         print '\n'
    
    # second pass. smooth over abrupt size changes
    
    if smooth_method == 'line_smooth':
        second_pass = line_smooth(all_lines)
    elif smooth_method == 'segment_smooth':
        second_pass = segment_smooth(all_lines)
    elif smooth_method == 'granular':
        second_pass = granular_smooth(all_lines)
    else:
        second_pass = []
        for i, s in enumerate(all_lines):
            if s in '་། ()〈〉༽༼༔༑':
                s = prev_size
                second_pass.append(prev_size)
            else:
                second_pass.append(s)
                prev_size = s
    if not second_pass:
        mode = 's'
        print('WARNING: using default value for mode')
    else:
        mode = statsmode(second_pass)[0][0]
    final_lines = []
    prev_inx = 0
    for line in content:
        cur_inx = len(line) + prev_inx
        final_lines.append(second_pass[prev_inx:cur_inx])
        prev_inx = cur_inx
    
    return mode, final_lines

def fit_dpgmm(arrs):
#     dpgmm = DPGMM(n_components = 45, alpha=10, n_iter=50)
#     dpgmm = DPGMM(n_components = 15, alpha=2.25, n_iter=50) ### good for kandze histories
    dpgmm = DPGMM(n_components = 20, alpha=.0, n_iter=50)
#     dpgmm = DPGMM(n_components = 55, alpha=2.75, n_iter=50)
    dpgmm.fit(arrs)
#     print 'dpgmm means'
#     print dpgmm.means_
#     print len(dpgmm.means_)
    return dpgmm


# def generate_formatting(page_objects, smooth_method='line_smooth'):
# def generate_formatting(page_objects, smooth_method='segment_smooth'):
def generate_formatting(page_objects, smooth_method='', ignore_first_line=False):
    from matplotlib import pyplot as plt
    allsizes = []
    single_chars = {}
    # print page_objects
    
    ##### First, gather widths into lists, one global (all widths) and one for 
    #####  each stack
    for p in page_objects:
        # content = p['other_page_info']
        content = p['page_info']
#         content = pickle.loads(content.decode('base64'))
#         content = json.loads(zlib.decompress(str(content).decode('string-escape')))
        p['content'] = content['content']
        
        for i, line in enumerate(p['content']):
            if ignore_first_line and i == 0:
                continue
            for char in line:
                try:
                    if char[2] not in (-1, 0) and num_stacks(char[-1]) == 1 and char[-1] not in '་།':
    #                     allchars.append(char[-1])
    #                     allsizes.append(char[3]) # using height rather than width
                        allsizes.append(char[2])
                        cl = single_chars.get(char[-1], [])
                        cl.append(char[2])
    #                     cl.append(char[3])
                        single_chars[char[-1]] = cl
                except:
                    print(char, line, content)
                    raise
    
    
    stack_mean = np.mean(allsizes)
    
    ### Modeling distributions of sizes
    print('modeling distribution of widths')
    maxsize = np.min([200, np.max(allsizes)])
    arrs = []
    keys = list(single_chars.keys())
    for c in keys:
        vals = single_chars[c]
#         plt.hist(vals, max(10, len(vals)/10))
#         plt.savefig(u'/home/zr/width-plots/%s-%d.png' % (c, len(vals)))
#         plt.clf()
        
        counts = Counter(vals)
        arr = np.zeros((maxsize+1), int)
        for cn in counts:
            if cn < 200:
#             if cn < 200 and counts[cn] > 5:
                arr[cn] = counts[cn]
        arrs.append(arr.astype(float))
    
#     arrs = np.array(arrs).astype(float)
#     print arrs.shape
#     dpgmm = DPGMM(n_components = 10, alpha=1.0, n_iter=50)
    
    dpgmm = fit_dpgmm(arrs)
    ndists = len(dpgmm.means_)
    letter_groups = [[] for i in range(ndists)]
    
    for i, c in enumerate(keys):
        label = dpgmm.predict(arrs[i].reshape((1,maxsize+1)))[0]
        letter_groups[label].append(c)

    from matplotlib import pyplot as plt
    pooled_gmms = {}
    stack_group_inx = {}
    for i, gr in enumerate(letter_groups):
        print('group ------')
        group_ws = []
        for l in gr:
            print(l, end=' ')
            group_ws.extend(single_chars[l])
            stack_group_inx[l] = i
        print()
        if group_ws:
            gmm = get_gmm_for_stack(group_ws)
            pooled_gmms[i] = gmm
            print('\tgroup', i, 'has', len(gmm.means_), 'dists. Converged? ', gmm.converged_)
#             if u'ཚ' in gr:
#                 from matplotlib.mlab import normpdf
            n,bins,p = plt.hist(group_ws, maxsize+1, normed=True)
#                 for i in range(2):
#                     plt.plot(bins, normpdf(bins, gmm.means_[i], gmm.covars_[i]),  label='fit', linewidth=1)
#             plt.show()
            plt.savefig('/media/zr/zr-mechanical/kandze3_hists/%s' % gr[0])
            plt.clf()
#             import sys; sys.exit()
        print()
        print('-------')
    
    print("converged?", dpgmm.converged_)
    print('number of chars in each group:')
    for i in letter_groups:
        print(len(i), end=' ')
    stacks_gmms = pooled_gmms
#     import sys; sys.exit()
    ##### Create a GMM for widths corresponding to each stack
#     stacks_gmms = {}
#     for c in single_chars:
# #         vals = gaussian_filter1d(single_chars[c], 2)
#         vals = single_chars[c]
#         stacks_gmms[c] = get_gmm_for_stack(vals)
#         print c, 'number of sizes' ,len(stacks_gmms[c].means_)
    
    from functools import partial
    
    assign_sizes = partial(assign_sizes_for_page, stack_inx_pointer=stack_group_inx, stacks_gmms=stacks_gmms, stack_mean=stack_mean, prev_size='', smooth_method=smooth_method)
    pool = multiprocessing.Pool()
    prev_size = ''
    modes = Counter()
    mapped = pool.map(assign_sizes, [p['content'] for p in page_objects])
    for i, p in enumerate(page_objects):
        mode, size_info = mapped[i]
#         mode, size_info = assign_sizes_for_page(p['content'], stack_group_inx, stacks_gmms, stack_mean, prev_size, smooth_method=smooth_method)
        modes.update(mode)
        p['size_info'] = size_info

    mode = modes.most_common(1)[0][0]
#     truncated_page_objects = []
    for p in page_objects:
#         pn = {}
        if not p['size_info']:
            continue
        size_info = p['size_info']
        buffer = 'volume-mode\t' + mode + '\n'
#         print size_info
        for l in size_info:
            if l:
                buffer += ''.join(l)
                buffer += '\n'
                prev_size = l[-1]
#         print buffer
#         print
        p['page_info']['size_info'] = buffer
#         pn['id'] = p['id']
#         truncated_page_objects.append(pn)
    
    
    # reduce info being sent back to database by throwing out all info that
    # hasn't been changed for each page
#     print 'trying to visualize'
#     for i, p in page_objects:
#     p = page_objects[55]
#     show_sample_page(p)
#     for i, p in enumerate(page_objects):
#         if p['tiffname'] == 'tbocrtifs/ngb_vol05/out_0017.tif':
#             print 'showing sample_page', i
#             show_sample_page(p)
#             break
#     else:
#         print 'tiffnotfound'
#     sys.exit()
    return page_objects
        
def show_sample_page(page_object):
    print('generating a sample image to view')
    pginfo = page_object['other_page_info']
    print(page_object['tiffname'])
    contents = json.loads(zlib.decompress(pginfo.decode('string-escape')))
    size_info = page_object['line_boundaries']
    slines = size_info.split('\n')
    mode = slines[0].split('\t')[1]
    
#     if mode == 'b':
#         special = 's'
#     else:
#         special = 'b'
    
    prev_size = slines[1][0]
    all_lines = []
    for i, s in enumerate(contents):
        line = contents[i]
        sizes = slines[i+1]
        cur_line = []
        for j, char in enumerate(line):
            size = sizes[j]
            if j == 0 and size != mode:
                cur_line.append('<span class="format_size">')
                cur_line.append(char[-1])
            elif size != mode and prev_size == mode:
                cur_line.append('<span class="format_size">')
                cur_line.append(char[-1])                
            elif size == mode and prev_size != mode:                
                cur_line.append('</span>')
                cur_line.append(char[-1])
            elif j == len(line) -1 and size != mode and prev_size != mode:
                cur_line.append(char[-1])
                cur_line.append('</span>')
            elif j == len(line) -1 and size != mode and prev_size == mode:
                cur_line.append('<span class="format_size">')
                cur_line.append(char[-1])
                cur_line.append('</span>')
            else:
                cur_line.append(char[-1])
            prev_size = size
        
        all_lines.append(''.join(cur_line))
    
    template = '''
    <html>
    <head>
    <style>
    body {
    font-family: "Qomolangma-Uchen Sarchung";
    }
    .format_size {
        color: blue;
    }
    </style>
    </head>
    <body>
    %s
    <hr>
    %s
    </body>
    </html>
    
    ''' % ('<br>'.join(all_lines), '\n'.join(size_info))
    import codecs
    codecs.open('/tmp/format_test.html', 'w', 'utf-8').write(template)
    import webbrowser
    webbrowser.open_new_tab('file:///tmp/format_test.html')
#     return mode, all_lines
