import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from page_elements2 import PageElements

from PIL import Image
import numpy as np
from page_elements2 import PageElements



line_nums = {
    '/home/zr/letters/page_line_test/out_0001.tif': 1,
    '/home/zr/letters/page_line_test/out_0003_kama.tif': 3,
    '/home/zr/letters/page_line_test/out_0004.tif': 1,
    '/home/zr/letters/page_line_test/out_0595.tif': 5,
    '/home/zr/letters/page_line_test/out__0659.tif': 5,
    '/home/zr/letters/page_line_test/out__0655.tif':6,
    '/home/zr/letters/page_line_test/out_0003_ngb.tif': 4,
    '/home/zr/letters/page_line_test/out_0002.tif': 1,
    '/home/zr/letters/page_line_test/out_0594.tif': 4,
    '/home/zr/letters/page_line_test/out_0596.tif': 6,
    '/home/zr/letters/page_line_test/out_0598.tif': 6,
    '/home/zr/letters/page_line_test/2_line_ex.tif':2,
    '/home/zr/letters/page_line_test/1_line_ex.tif':1,
    '/home/zr/letters/page_line_test/3_line_ex.tif': 3,
    '/home/zr/letters/page_line_test/1_line_ex2.tif': 1,
    '/home/zr/letters/page_line_test/1_line_ex3.tif': 1,
    '/home/zr/letters/page_line_test/1_line_ex4.tif': 1,
    '/home/zr/letters/page_line_test/1_line_ex5.tif': 1,
    '/home/zr/letters/page_line_test/4_line_ex2.tif': 4,
    '/home/zr/letters/page_line_test/5_line_ex.tif': 5,
    '/home/zr/letters/page_line_test/3_line_ex2.tif': 3,
    '/home/zr/letters/page_line_test/1_line_ex6.tif': 1,
             }

res = []
fails = 0
for key, val in sorted(line_nums.items()):
    im = Image.open(key).convert('L')
    a = np.asarray(im)/255
    p = PageElements(a, page_type='pecha')
    is_match = val == p.num_lines
    print(key, val, 'actual', p.num_lines, 'predicted', "match?", is_match)
    if not is_match:
        fails += 1
    res.append(is_match)

print("All matched?", all(res), '(fails: %d of %d)' % (fails, len(res)))
