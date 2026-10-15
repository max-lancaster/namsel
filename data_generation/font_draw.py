# encoding: utf-8

from PIL import ImageFont, ImageDraw, Image
import glob
from itertools import chain
import numpy as np
try:
    from scipy.misc import imresize
except ImportError:
    # scipy.misc.imresize was removed in scipy 1.3.0
    from skimage.transform import resize
    def imresize(arr, size):
        return resize(arr, size, preserve_range=True).astype(arr.dtype)
import sys
from scipy.ndimage.filters import gaussian_filter
import cv2
from scipy.ndimage.morphology import binary_erosion
from random import randint
import os
import io

sys.path.append('..')
from yik import *
from utils import add_padding

import platform

if platform.system() != "Windows":
    import multiprocessing

#num, punctuation, vowels1, 

letters = chain(alphabet, twelve_ra_mgo, ten_la_mgo,\
    eleven_sa_mgo, seven_ya_tags, twelve_ra_tags, six_la_tags, ya_tags_stack,\
    ra_tags_stack)

wa_zur = ['\u0f40\u0fad', '\u0f41\u0fad', '\u0f42\u0fad', '\u0f45\u0fad', '\u0f49\u0fad', '\u0f4f\u0fad', '\u0f51\u0fad', '\u0f59\u0fad', '\u0f5a\u0fad', '\u0f5e\u0fad', '\u0f5f\u0fad', '\u0f62\u0fad', '\u0f63\u0fad', '\u0f64\u0fad', '\u0f66\u0fad', '\u0f67\u0fad']
roman_num = [str(i) for i in range(10)]
misc_glyphs = ['དྨ', 'གྲྭ',  '༄',  '༅',  '྅',]
skt = ['ཧཱུཾ','བཱ', 'ནཱ', 'ནྟྲ', 'ཏྣ', 'ཛྲ', 'བྷ', 'ཀྟ', 'དྷ', 'ཨཱོཾ', 'ཎྜ', 'དྷི']
retroflex = ['ཊ','ཋ','ཌ','ཎ','ཥ',]
other = ['(', ')', '༼',  '༽', '༔',  '༑', ]
other2 = [ '〈', '〉' , 'ཏཱ', 'ཤཱ', '༷', 'ཿ', 'རཱ', 'ཤཱ', 'ནྟི', 'ཧཱ', 'དྡྷི',
           'ཀྐི', 'ཏྟ','ཛྙཱ', 'སྭཱ', 'ཧྲཱི', 'ཀྐོ', 'དྷེ', 'ཝཾ', 'ཀྵ', 'ལླ', 'ཧཾ',
          'གྷ', 'ཛྫ', 'མཱ', 'ངྒཱ', 'ཨོཾ', 'ཨཱ','ཁཾ', 'ཀྵི', 'བྷྱ', 'ཀཱ', 'ཥྛ', 
          'དཱི', 'རྦྷེ', 'ཛྲེ', 'སཱ', 'ཎྜི', 'ལཱ', 'པཱ', 'ཉྫ', 'བྷྱོ', 'ཉྩ','རྒྷཾ', 
          'ནྡ', 'ཁཱ', 'ཛྷ', 'ཤྲཱི', 'ཌི', 'ཛཱ', 'བེེ', 'ཏྤ', 'ཌཱུ', 'ནྟ', 'གཱ', 
          'ཤཱི', 'རྱ', 'རྶཱ', 'པཾ', 'ཙཱ', 'རྨཱ', 'ཎི', 'ཪྴ', 'བཱི', 'ངྒེ', 'ཊི', 
          'རཱི', 'ངྐ', 'མྦྷ', 'སྠཱ' , '①','②', '③', '④', '⑤', '⑥','⑦', 
          '⑧', '⑨', '⑩', '⑪', '⑫', '⑬', '⑭', '⑮', '⑯','⑰', 
          '⑱', '⑲', '⑳', '༈', '—', '‘', '’', 'ཀྲྀ', 'ནྡྲ','ནྡྲ','ཛེེ',
          'ཉྫུ', 'སྠཱ', 'བྷཱུ', 'བྷུ','དྷཱུ', 'རྦྷ', 'ཌོ', 'མྦི', 'བྷེ', 'ཡྂ', 'དྷྲི',
          'ཧཱུྃ', 'ཤཾ', 'ཏྲྂ', 'བྂ', 'རྂ', 'ལྂ', 'སུྂ', 'ཕྱྭ', 'ཪྵ', 'ཪྷ', 'ཧཻ', 
          'ཅྀ', 'ཀྱྀ', 'ཛཱི', 'བྲྀ', 'ཏཱུ', 'རྻ', 'ཊོ', 'རྀ', 'ཊུ', 'ཕྱྀ', 'ཤྲི', 
          'ཊཱ', 'ངྒྷ', 'ནྜི', 'གཽ', 'རྞ', 'ཀྱཱ', 'རྩྭ', 'ཡཱི', 'ཛྷཻ', 'སྭོ', 'ཁྲྀ', 
          'ཀྐ', 'ཙྪ', 'ཏཻ', 'སྭེ', 'ཧྲཱུ', 'ལྦཱ', 'གྷཱ', 'གྷི', 'ངྐུ', 'གྷུ', 'ཤྱི', 
          'གྷོ', 'ཥྚ', 'སྐྲྀ', 'ཧཱུ', 'ཥྐ', 'དྔྷ', 'ཐཱ', 'ཏྠ', 'པཱུ', 'བྷྲ', 'ཇྀ', 
          'ཥཱ', 'ཏྱ', 'ཤྱ', 'གྷྲ', 'པྱྀ', 'ཧྲཱྀ', 'ནཱི', 'ཤྀ', 'དཱུ', 'ཏྲཱི', 'ཀཻ', 
          'ཤྭེ', 'ཤྐ', 'ཀཽ', 'གྒུ', 'དྷཱི', 'ཧླ', 'ཧྥ', 'ཙཱུ', 'པླུ', 'ཟྀ', 'ཉྫི', 
          'ཤླཽ', 'ངྒི', 'མྱྀ', 'སྟྲི', 'ཀྱཻ', 'དྲྭ', 'རྒྷ', 'དྲྀ', 'ཏྭོ', 'ཧྥི', 'ཀྲཱ', 
          'ནྟུ', 'ཧྥུ', 'ཧྥེ', 'ཧྥོ', 'སྠ', 'གཱི', 'ཞྀ', 'ཉྀ', 'ཀྵྀ', 'ཀཱུ', 'གྱྀ', 
          'བྱཱ', 'ཀྴི', 'ཁཱེ', 'སྷོ', 'རཱུ', 'ཉྪ', 'དཱ', 'དྡྷཱ', 'ངྷྲུ', 'ཧྨ', 'ཊཱི', 
          'དྷྭ', 'ནཻ', 'མྲྀ', 'ནྡྷེ', 'ནྡྷོ', 'ཨཽ', 'ལླཱི', 'ནྡྷུ', 'གྷྲི', 'ལཱི', 'ངྒ', 
          'དྐུ', 'པྟ', 'དྨཱ', 'ཨཱོ', 'ཏཱི', 'ཉྩི', 'དྨེ', 'དྨོ', 'མཻ', 'དྷོ', 'སྟྱ', 
          'ལླེ', 'སཱི', 'དྷུ', 'ནྡྷ', 'ལླི', 'མྦྷི', 'ཊྭཱ', 'ྈ', 'ནྱ', 'ཥེ', 'ཡཱ', 
          'ནྨ', 'ཁྱྀ', 'ཌཱ', 'བྷོ', 'འྀ', 'ཨྱོ', 'ཨྱཽ', 'ཏྱཱ', 'བྷྲི', 'ཤྲ', 'བྷཱ', 
          'བྷི', 'ནྡི', 'ནྀ', 'ཥི', 'དྷྲ', 'དྷྱ', 'ནྡྷི', 'ཛྙ', 'སཽ', 'ཝཱ', 'ལྱ', 
          'མཱུ', 'དྭོ', 'ཀྵུ', 'ཀྞི', 'ཥྚི', 'རྤཱ', 'བཻ', 'མྦུ', 'ཛྭ', '༵', 'དྡྷ',
          'ནྡེ', 'སྨྲྀ', 'མེེ', 'ཀྵེ', 'ཀྵིཾ', 'སཾ', 'ཪྻཱ', 'དྷྱཱ', 'ཧྱེ', 'བཾ', 'སྫ',
          'ཝཱི', '྾', 'ཥྤེ', 'ནེེ', 'ཊྭཱཾ', 'དྷཱ', 'ལཱུ', 'ཕྲོཾ', 'མྨུ', 'ཏྨ', 'ཎྜཾ',
          'མཱཾ', 'ནྣི', 'སྟྲ', 'སྟྭཾ','ཏྤཱ', 'ཥྚྲྀ', 'ཥྚྲྀ', 'ཉྩཱ', 'ཧྱ', 'ཏྟྭཾ', 'ཛྙོ',
          'ཤྩ','ཏྭེ','ཌྷོ','ཥྱོ', 'ཀྟོ', 'ཏྲེེ', 'ཛྲཱི', 'ཊཾ', 'མྨེ', 'ོ', 'གྣེ',
          'གྣ', 'གྣི', '༴', 'ཪྻ', 'བྷྲཱུཾ', 'རཾ', 'ཡཾ', 'ཋཱ', 'པེཾ', 'ལིཾ', 'སྥ',
          'ཀེཾ', 'ཀྩི', 'ཏྲཾ', 'མྺ', 'རྭི', 'ཏྟི', 'ཉྩུ', 'ལྐཾ', 'ལཾ', 'ཉྫཾ', 'རྔྷི',
          'སཱཾ', 'ཊེ', 'ཋི', 'ནཾ', 'ཎྛ', 'ཎྛཱི', 'ཎྛོ', 'ཉྪ', 'ཀཾ', 'ཋོ', 'ཋཾ',
          'གྷྣཱ', 'ཙྩ', 'ཛྫུ', 'ཌྜི', 'ཌྷ', 'ཀཱི', 'བྫ', 'བྷྣྱ', 'ཪྻཾ', 'ཥྐུ', 'ཧྣཱ', 
          'ཀྵྞཱ', 'ཨཱི', 'ཙྪཱ', 'ཊྚ', 'མྤ', 'རྦྦ', 'པྟཾ', 'རྞེ', 'སྨཾ', 'ཥྷ', 'རྡྷ', 
          'ཧྨེ', 'ནྡྲི', 'ཪྵི', 'ཎཱ', 'ཎུ', 'ཎོ', 'ཎཾ', 'ཏྤུ', 'ཏྤཱུ', 'ཥུ', 'ནཱཾ', 
          'ཏྲྀ', 'ཏྱུ', 'ཏྭཱ', 'ཏྭི', 'ནྣཱི', 'ཐྱཱ', 'ཏྻི', 'ནྟཱ', 'ནྟྭ', 'ནྠ', 'ནྟྭ', 
          'ནྠི', 'ཪྤྤ', 'རྦྱ', 'པྱཱ', 'ཎྜུ', 'ཉྥ', 'བྱཾ', 'མྠ', 'མྤ', 'མྤི', 'མྤྱ', 
          'མྤྲཾ', 'མྦ', 'མྦོ', 'མྦྱ', 'མླི', 'ཏཾ', 'བྠཱ', 'ཪྪེ', 'རྞི', 'རྞོ',  'རྞྞ',
          'ཊྚི', 'ཥྚཾ', 'མུཾ','ཀྟི', 'གྷེེ', 'དྲུཾ', 'ལྱེ', 'ཀྷྃ','ཏྲཱ' , '▶▶', '[',
          ']', 'སྫོ', 'ཟྱ', 'ཨྠིྀ', 'བྷེེ']
### Extended
#other = [u'(', u')', u'༼',  u'༽', u'༔',  u'༑' ] # comment out above "other" if use this
english_alphbet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
english_punc = list('!"#$%&*+,—/:;<=>?@[]{}')
english_misc = list('〈〉①②③④⑤⑥⑦⑧⑨⑩¶')
english_attached = 'oo ar ri ri ki ak ry ar ar ri ri fl th ary ry ar th art ry try tt ar od th tt ki ri rm tt ar th a/ as ar ar fi art ri'.split(' ')
other_extended = []

k=0
everything = []
for i in letters:
    everything.append(i)
#    k+=1
#    print k, i
    for j in vowels1:
        everything.append(i+j)
#        k+=1
#        print '\t',k, i+jརྔྷི

allchars = list(chain(everything, ['་','།'], num, wa_zur, roman_num, misc_glyphs, skt, retroflex, other, other2))
#allchars = list(chain(everything, [u'་',u'།'], num, wa_zur, roman_num, misc_glyphs, skt, retroflex, other, english_alphbet, english_punc, english_misc, other2, other_extended))

# import codecs
# tstacks = codecs.open('tibetan_stacks.txt', 'w', 'utf-8')
# for a in allchars:
#     tstacks.write(a)
#     tstacks.write('\n')
#     
# tstacks.close()
# sys.exit()
#     

# g = [u'\u0f4e\u0f9c\u0f7e', u'\u0f42\u0fa3', u'\u0f42\u0fb7\u0fa3\u0f71', u'\u0f59\u0fa9', u'\u0f5b\u0fab\u0f74', u'\u0f4c\u0f9c\u0f72', u'\u0f4c\u0f72', u'\u0f4c\u0fb7', u'\u0f40\u0f71\u0f72', u'\u0f56\u0fab', u'\u0f56\u0fb7\u0fa3\u0fb1', u'\u0f51\u0fb7\u0f71\u0f74', u'\u0f64\u0fa9', u'\u0f6a\u0fbb\u0f7e', u'\u0f65\u0f90\u0f74', u'\u0f67\u0fa3\u0f71', u'\u0f40\u0fb5\u0f9e\u0f71', u'\u0f42\u0fa3\u0f7a', u'\u0f68\u0f71\u0f72', u'\u0f42\u0fb7', u'\u0f59\u0faa\u0f71', u'\u0f5b\u0f99', u'\u0f49\u0fa9', u'\u0f4a\u0f9a', u'\u0f58\u0fa4', u'\u0f62\u0fa6\u0fa6', u'\u0f54\u0f9f\u0f7e', u'\u0f62\u0f9e\u0f7a', u'\u0f65\u0f9a', u'\u0f66\u0fa8\u0f7e', u'\u0f65\u0fb7', u'\u0f62\u0fa1\u0fb7', u'\u0f67\u0fa8\u0f7a', u'\u0f53\u0fa1\u0fb2\u0f72', u'\u0f64\u0fb1', u'\u0f59\u0faa', u'\u0f6a\u0fb5\u0f72', u'\u0f4e\u0f71', u'\u0f4e\u0f74', u'\u0f4e\u0f7c', u'\u0f4e\u0f7e', u'\u0f4f\u0fa4', u'\u0f4f\u0fa4\u0f74', u'\u0f4f\u0fa4\u0f71\u0f74', u'\u0f63\u0f71\u0f72', u'\u0f56\u0fb7\u0f72', u'\u0f65\u0f74', u'\u0f53\u0f71\u0f7e', u'\u0f4f\u0f71\u0f72', u'\u0f4f\u0fb2\u0f80', u'\u0f4f\u0fb1\u0f74', u'\u0f4f\u0fad\u0f71', u'\u0f4f\u0fad\u0f72', u'\u0f53\u0fa3\u0f71\u0f72', u'\u0f50\u0f71', u'\u0f50\u0fb1\u0f71', u'\u0f4f\u0fbb\u0f72', u'\u0f51\u0fb7\u0f7c', u'\u0f51\u0fb7\u0fb1', u'\u0f53\u0f9f\u0f71', u'\u0f53\u0f9f\u0f72', u'\u0f53\u0f9f\u0f74', u'\u0f53\u0f9f\u0fad', u'\u0f53\u0fa0', u'\u0f53\u0f9f\u0fad', u'\u0f53\u0fa0\u0f72', u'\u0f6a\u0fa4\u0fa4', u'\u0f62\u0fa6\u0fb1', u'\u0f53\u0fb1', u'\u0f54\u0fb1\u0f71', u'\u0f6a\u0fb4', u'\u0f64\u0fa9', u'\u0f6a\u0fbb', u'\u0f4e\u0f9c\u0f74', u'\u0f49\u0fa5', u'\u0f56\u0fb1\u0f71', u'\u0f56\u0fb1\u0f7e', u'\u0f56\u0fb7\u0f7a', u'\u0f58\u0fa0', u'\u0f58\u0fa4', u'\u0f58\u0fa4\u0f72', u'\u0f58\u0fa4\u0fb1', u'\u0f58\u0fa4\u0fb2\u0f7e', u'\u0f58\u0fa6', u'\u0f58\u0fa6\u0f72', u'\u0f58\u0fa6\u0f74', u'\u0f58\u0fa6\u0f74', u'\u0f58\u0fa6\u0f7c', u'\u0f58\u0fa6\u0fb1', u'\u0f58\u0fb3\u0f72', u'\u0f42\u0f71\u0f72', u'\u0f4f\u0f7e', u'\u0f53\u0fa1\u0fb7', u'\u0f56\u0fa0\u0f71', u'\u0f66\u0fa0', u'\u0f6a\u0faa\u0f7a']
# for iii in g:
#     if iii not in allchars:
#         continue
#     else:
#         print u"u'{}',".format(iii),

############# check if a char is already in the complete char set, then exit
# tst_char = u'བྷེེ'
# print tst_char, tst_char in allchars
# print allchars.count(tst_char)
# sys.exit()
###########


allchars_label = list(zip(list(range(len(allchars))),allchars))
#print len(allchars)
## NORMAL DICT — write the modern gzip+JSON char maps (data-only, not shelve/pickle)
import sys as _sys, os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _root)
from safe_model_io import dump_model
dump_model(dict((j, i) for i, j in allchars_label), _os.path.join(_root, 'allchars.json.gz'))
dump_model(dict((i, j) for i, j in allchars_label), _os.path.join(_root, 'label_chars.json.gz'))
####

### EXTENDED
#s = shelve.open('/home/zr/letters/allchars_dict_extended')
#s['allchars'] = dict((j,i) for i,j in allchars_label)
#s['label_chars'] = dict((i, j) for i,j in allchars_label)
#s.close()

#sys.exit()
#import os
fonts = glob.glob('*ttf')
sample_word = 'བསྒྲུབས'
sample_word = 'ཆ'
fontnames = ['Tibetan BZDMT Uni',
 'Tibetan Machine Uni',
 'Qomolangma-Uchen Sarchen',
 'Qomolangma-Uchen Sarchung',
 'Qomolangma-Uchen Suring',
 'Qomolangma-Uchen Sutung',
# 'Monlam Uni Ochan2',
 'Jomolhari',
 'Microsoft Himalaya',
 'TCRC Youtso Unicode',
 'Uchen_05',
# 'XTashi',
# 'Tib-US Unicode',
 "Monlam Uni OuChan4",
"Monlam Uni OuChan5",
"Monlam Uni OuChan2",
"Monlam Uni OuChan3",
"Monlam Uni OuChan1",
"Monlam Uni Sans Serif",
#"Amne"
# 'Wangdi29'
 ]

def trim(arr):
    top=0
    bottom = len(arr)-1
    left = 0
    right = arr.shape[1]

    for i, row in enumerate(arr):
        if not row.all():
            top = i
            break
    
    for i in range(bottom, 0, -1):
        if not arr[i].all():
            bottom = i
            break
    for i, row in enumerate(arr.transpose()):
        if not row.all():
            left = i
            break
    
    for i in range(right-1, 0, -1):
        if not arr.transpose()[i].all():
            right = i
            break
    
#    print bottom, top, left, right
    return arr[top:bottom, left:right]
#    Image.fromarray(arr.transpose()).show()
    

import cairo
import pango
import pangocairo
import sys
import pprint
from cv2 import resize, INTER_AREA

# DON'T FORGET TO COMMENT THIS OUT IF SAVING IMAGES
#out = open('training_set_new.csv','w')

#for label, char in allchars_label:
def draw_fonts(args):
    label = args[0]
    char = args[1]
    output = []
    for cycle in range(1):
        for fontname in fontnames:
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 600, 600)
            context = cairo.Context(surf)
            
            #draw a background rectangle:
            context.rectangle(0,0,600,600)
            context.set_source_rgb(1, 1, 1)
            context.fill()
            
            #get font families:
            
            font_map = pangocairo.cairo_font_map_get_default()
        #    context.translate(0,0)
            
            pangocairo_context = pangocairo.CairoContext(context)
            pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
            
            layout = pangocairo_context.create_layout()
            #fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"
    #         font = pango.FontDescription(fontname + " 200")
            if cycle == 0:
                font = pango.FontDescription(fontname + " 200")
            else:
                font = pango.FontDescription(fontname + " bold 200")
                
            layout.set_font_description(font)
            
            layout.set_text(char)
            context.set_source_rgb(0, 0, 0)
            pangocairo_context.update_layout(layout)
            pangocairo_context.show_layout(layout)
            
            im = Image.frombuffer("RGBA", (surf.get_width(), surf.get_height()), surf.get_data(), "raw", "RGBA",0 ,1)
            im = im.convert('L')
            
            a = np.asarray(im)
            a = trim(a)
            a = add_padding(a, padding=2)
            #####experimental normalization
            
            h, w = a.shape
            h = float(h)
            w = float(w)
            L = 32
            sm = np.argmin([h,w])
            bg = np.argmax([h,w])
            R1 = [h,w][sm]/[h,w][bg]
    #        R2 = np.sqrt(np.sin((np.pi/2.0)*R1))
    #        R2 = pow(R1, (1/3)) 
            R2 = np.sqrt(R1) 
    #        R2 = R1 
    #        if R1 < .5:
    #            R2 = 1.5*R1 + .25
    #        else:
    #            R2 = 1
                
            if sm == 0:
                H2 = L*R2
                W2 = L
            else:
                H2 = L
                W2 = L*R2
            
            alpha = W2 / w
            beta = H2 / h
            
            a = resize(a, (0,0), fy=beta, fx=alpha, interpolation=INTER_AREA)
            
            smn = a.shape[sm]
            offset = int(np.floor((L - smn) / 2.))
            c = np.ones((L,L), dtype=np.uint8)*255
    #        print a
    #        print a.shape
            if (L - smn) % 2 == 1:
                start = offset+1
                end = offset
            else:
                start = end = offset
                
            if sm == 0:
    #            print c[start:L-end, :].shape, a.shape
                c[start:L-end, :] = a
            else:
    #            print c[:,start:L-end].shape, a.shape
                c[:,start:L-end] = a
            
            
            #########classic approach
    #        im = Image.fromarray(a)
    #        im.thumbnail((16,32), Image.ANTIALIAS)
    #        im = np.asarray(im)
    #        a = np.ones((32,16), dtype=np.uint8)*255
    #        a[0:im.shape[0],0:im.shape[1]] = im
            ###########
        #### USE FOLLOWING IF WANT TO SAVE NEW TRAINING DATA ##################
            a = c
            a[np.where(a<120)] = 0
            a[np.where(a>=120)] = 1
             
    #        a = degrade(a)
     
            output.append(str(label)+','+','.join(str(i) for i in a.flatten()))
#
#   
    return output
    ####################################
    
    
    ## USE FOLLOWING IF YOU WANT TO SAVE SAMPLE IMAGES###################
#      MAKE SURE TO COMMENT OUT ABOVE FILE WRITING SECTION
#         Image.fromarray(a).save('training_letters_latest_bold/'+str(label)+'_'+fontname+'.tif', )
#         Image.fromarray(c).save('training_letters_extended/'+str(label)+'_'+fontname+'.tif', )
    ##########################

#import numpy as np
#im = Image.frombuffer('RGB', (100,100), content)
#im.show()
#a = np.asarray(im)
#print a
#from scipy.misc import imread, imshow


#for f in fonts:
#    im = Image.new('L', (200,200), 255)
#    draw = ImageDraw.Draw(im)
#    font = ImageFont.truetype(f, 30)
#    draw.text((0,0), sample_word, font=font)
#    draw.text((0,50), f)
#    im.save(f+'_sample.png')

#from matplotlib.font_manager import FontProperties
#from matplotlib import use
#
#use('gtk')

#from matplotlib import pyplot as plt
#font_prop = FontProperties(fname=f)
#fig = plt.Figure(figsize=(1,1),facecolor='white')
#ax = plt.axes([0,0,.25,.25],axisbg='white',frameon=False)
#ax.set_xticks([])
#ax.set_yticks([])
#ax.text(0,0,sample_word, fontproperties=font_prop, size=40)
##plt.figtext(0,0,sample_word, fontproperties=font_prop, size=40)
#plt.savefig('out.png',bbox_inches=0)
#plt.matshow(a)
#plt.show()

def gen_img_rows(outfile, parallel=True):
    if parallel == True:
        p = multiprocessing.Pool()
        data = p.map(draw_fonts, allchars_label)
    else:
        data = list(map(draw_fonts, allchars_label))
    out = open(outfile, 'w')
    outf = []
    for let in data:
        for fnt in let:
            outf.append(fnt)
    out.write('\n'.join(outf))
    

if __name__ == '__main__':
    if platform.system() == "Windows":
        gen_img_rows(r'..\datasets\font-draw-samples.txt', False)
    else:
        gen_img_rows('../datasets/font-draw-samples.txt')
