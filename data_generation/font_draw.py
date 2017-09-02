# encoding: utf-8

from PIL import ImageFont, ImageDraw, Image
import glob
from itertools import chain
import numpy as np
from scipy.misc import imresize
import sys
from scipy.ndimage.filters import gaussian_filter
import cv
from scipy.ndimage.morphology import binary_erosion
import multiprocessing
from random import randint
import os

sys.path.append('..')
from yik import *
from utils import add_padding

#num, punctuation, vowels1, 

letters = chain(alphabet, twelve_ra_mgo, ten_la_mgo,\
    eleven_sa_mgo, seven_ya_tags, twelve_ra_tags, six_la_tags, ya_tags_stack,\
    ra_tags_stack)

wa_zur = [u'\u0f40\u0fad', u'\u0f41\u0fad', u'\u0f42\u0fad', u'\u0f45\u0fad', u'\u0f49\u0fad', u'\u0f4f\u0fad', u'\u0f51\u0fad', u'\u0f59\u0fad', u'\u0f5a\u0fad', u'\u0f5e\u0fad', u'\u0f5f\u0fad', u'\u0f62\u0fad', u'\u0f63\u0fad', u'\u0f64\u0fad', u'\u0f66\u0fad', u'\u0f67\u0fad']
roman_num = [unicode(i) for i in range(10)]
misc_glyphs = [u'དྨ', u'གྲྭ',  u'༄',  u'༅',  u'྅',]
skt = [u'ཧཱུཾ',u'བཱ', u'ནཱ', u'ནྟྲ', u'ཏྣ', u'ཛྲ', u'བྷ', u'ཀྟ', u'དྷ', u'ཨཱོཾ', u'ཎྜ', u'དྷི']
retroflex = [u'ཊ',u'ཋ',u'ཌ',u'ཎ',u'ཥ',]
other = [u'(', u')', u'༼',  u'༽', u'༔',  u'༑', ]
other2 = [ u'〈', u'〉' , u'ཏཱ', u'ཤཱ', u'༷', u'ཿ', u'རཱ', u'ཤཱ', u'ནྟི', u'ཧཱ', u'དྡྷི',
           u'ཀྐི', u'ཏྟ',u'ཛྙཱ', u'སྭཱ', u'ཧྲཱི', u'ཀྐོ', u'དྷེ', u'ཝཾ', u'ཀྵ', u'ལླ', u'ཧཾ',
          u'གྷ', u'ཛྫ', u'མཱ', u'ངྒཱ', u'ཨོཾ', u'ཨཱ',u'ཁཾ', u'ཀྵི', u'བྷྱ', u'ཀཱ', u'ཥྛ', 
          u'དཱི', u'རྦྷེ', u'ཛྲེ', u'སཱ', u'ཎྜི', u'ལཱ', u'པཱ', u'ཉྫ', u'བྷྱོ', u'ཉྩ',u'རྒྷཾ', 
          u'ནྡ', u'ཁཱ', u'ཛྷ', u'ཤྲཱི', u'ཌི', u'ཛཱ', u'བེེ', u'ཏྤ', u'ཌཱུ', u'ནྟ', u'གཱ', 
          u'ཤཱི', u'རྱ', u'རྶཱ', u'པཾ', u'ཙཱ', u'རྨཱ', u'ཎི', u'ཪྴ', u'བཱི', u'ངྒེ', u'ཊི', 
          u'རཱི', u'ངྐ', u'མྦྷ', u'སྠཱ' , u'①',u'②', u'③', u'④', u'⑤', u'⑥',u'⑦', 
          u'⑧', u'⑨', u'⑩', u'⑪', u'⑫', u'⑬', u'⑭', u'⑮', u'⑯',u'⑰', 
          u'⑱', u'⑲', u'⑳', u'༈', u'—', u'‘', u'’', u'ཀྲྀ', u'ནྡྲ',u'ནྡྲ',u'ཛེེ',
          u'ཉྫུ', u'སྠཱ', u'བྷཱུ', u'བྷུ',u'དྷཱུ', u'རྦྷ', u'ཌོ', u'མྦི', u'བྷེ', u'ཡྂ', u'དྷྲི',
          u'ཧཱུྃ', u'ཤཾ', u'ཏྲྂ', u'བྂ', u'རྂ', u'ལྂ', u'སུྂ', u'ཕྱྭ', u'ཪྵ', u'ཪྷ', u'ཧཻ', 
          u'ཅྀ', u'ཀྱྀ', u'ཛཱི', u'བྲྀ', u'ཏཱུ', u'རྻ', u'ཊོ', u'རྀ', u'ཊུ', u'ཕྱྀ', u'ཤྲི', 
          u'ཊཱ', u'ངྒྷ', u'ནྜི', u'གཽ', u'རྞ', u'ཀྱཱ', u'རྩྭ', u'ཡཱི', u'ཛྷཻ', u'སྭོ', u'ཁྲྀ', 
          u'ཀྐ', u'ཙྪ', u'ཏཻ', u'སྭེ', u'ཧྲཱུ', u'ལྦཱ', u'གྷཱ', u'གྷི', u'ངྐུ', u'གྷུ', u'ཤྱི', 
          u'གྷོ', u'ཥྚ', u'སྐྲྀ', u'ཧཱུ', u'ཥྐ', u'དྔྷ', u'ཐཱ', u'ཏྠ', u'པཱུ', u'བྷྲ', u'ཇྀ', 
          u'ཥཱ', u'ཏྱ', u'ཤྱ', u'གྷྲ', u'པྱྀ', u'ཧྲཱྀ', u'ནཱི', u'ཤྀ', u'དཱུ', u'ཏྲཱི', u'ཀཻ', 
          u'ཤྭེ', u'ཤྐ', u'ཀཽ', u'གྒུ', u'དྷཱི', u'ཧླ', u'ཧྥ', u'ཙཱུ', u'པླུ', u'ཟྀ', u'ཉྫི', 
          u'ཤླཽ', u'ངྒི', u'མྱྀ', u'སྟྲི', u'ཀྱཻ', u'དྲྭ', u'རྒྷ', u'དྲྀ', u'ཏྭོ', u'ཧྥི', u'ཀྲཱ', 
          u'ནྟུ', u'ཧྥུ', u'ཧྥེ', u'ཧྥོ', u'སྠ', u'གཱི', u'ཞྀ', u'ཉྀ', u'ཀྵྀ', u'ཀཱུ', u'གྱྀ', 
          u'བྱཱ', u'ཀྴི', u'ཁཱེ', u'སྷོ', u'རཱུ', u'ཉྪ', u'དཱ', u'དྡྷཱ', u'ངྷྲུ', u'ཧྨ', u'ཊཱི', 
          u'དྷྭ', u'ནཻ', u'མྲྀ', u'ནྡྷེ', u'ནྡྷོ', u'ཨཽ', u'ལླཱི', u'ནྡྷུ', u'གྷྲི', u'ལཱི', u'ངྒ', 
          u'དྐུ', u'པྟ', u'དྨཱ', u'ཨཱོ', u'ཏཱི', u'ཉྩི', u'དྨེ', u'དྨོ', u'མཻ', u'དྷོ', u'སྟྱ', 
          u'ལླེ', u'སཱི', u'དྷུ', u'ནྡྷ', u'ལླི', u'མྦྷི', u'ཊྭཱ', u'ྈ', u'ནྱ', u'ཥེ', u'ཡཱ', 
          u'ནྨ', u'ཁྱྀ', u'ཌཱ', u'བྷོ', u'འྀ', u'ཨྱོ', u'ཨྱཽ', u'ཏྱཱ', u'བྷྲི', u'ཤྲ', u'བྷཱ', 
          u'བྷི', u'ནྡི', u'ནྀ', u'ཥི', u'དྷྲ', u'དྷྱ', u'ནྡྷི', u'ཛྙ', u'སཽ', u'ཝཱ', u'ལྱ', 
          u'མཱུ', u'དྭོ', u'ཀྵུ', u'ཀྞི', u'ཥྚི', u'རྤཱ', u'བཻ', u'མྦུ', u'ཛྭ', u'༵', u'དྡྷ',
          u'ནྡེ', u'སྨྲྀ', u'མེེ', u'ཀྵེ', u'ཀྵིཾ', u'སཾ', u'ཪྻཱ', u'དྷྱཱ', u'ཧྱེ', u'བཾ', u'སྫ',
          u'ཝཱི', u'྾', u'ཥྤེ', u'ནེེ', u'ཊྭཱཾ', u'དྷཱ', u'ལཱུ', u'ཕྲོཾ', u'མྨུ', u'ཏྨ', u'ཎྜཾ',
          u'མཱཾ', u'ནྣི', u'སྟྲ', u'སྟྭཾ',u'ཏྤཱ', u'ཥྚྲྀ', u'ཥྚྲྀ', u'ཉྩཱ', u'ཧྱ', u'ཏྟྭཾ', u'ཛྙོ',
          u'ཤྩ',u'ཏྭེ',u'ཌྷོ',u'ཥྱོ', u'ཀྟོ', u'ཏྲེེ', u'ཛྲཱི', u'ཊཾ', u'མྨེ', u'ོ', u'གྣེ',
          u'གྣ', u'གྣི', u'༴', u'ཪྻ', u'བྷྲཱུཾ', u'རཾ', u'ཡཾ', u'ཋཱ', u'པེཾ', u'ལིཾ', u'སྥ',
          u'ཀེཾ', u'ཀྩི', u'ཏྲཾ', u'མྺ', u'རྭི', u'ཏྟི', u'ཉྩུ', u'ལྐཾ', u'ལཾ', u'ཉྫཾ', u'རྔྷི',
          u'སཱཾ', u'ཊེ', u'ཋི', u'ནཾ', u'ཎྛ', u'ཎྛཱི', u'ཎྛོ', u'ཉྪ', u'ཀཾ', u'ཋོ', u'ཋཾ',
          u'གྷྣཱ', u'ཙྩ', u'ཛྫུ', u'ཌྜི', u'ཌྷ', u'ཀཱི', u'བྫ', u'བྷྣྱ', u'ཪྻཾ', u'ཥྐུ', u'ཧྣཱ', 
          u'ཀྵྞཱ', u'ཨཱི', u'ཙྪཱ', u'ཊྚ', u'མྤ', u'རྦྦ', u'པྟཾ', u'རྞེ', u'སྨཾ', u'ཥྷ', u'རྡྷ', 
          u'ཧྨེ', u'ནྡྲི', u'ཪྵི', u'ཎཱ', u'ཎུ', u'ཎོ', u'ཎཾ', u'ཏྤུ', u'ཏྤཱུ', u'ཥུ', u'ནཱཾ', 
          u'ཏྲྀ', u'ཏྱུ', u'ཏྭཱ', u'ཏྭི', u'ནྣཱི', u'ཐྱཱ', u'ཏྻི', u'ནྟཱ', u'ནྟྭ', u'ནྠ', u'ནྟྭ', 
          u'ནྠི', u'ཪྤྤ', u'རྦྱ', u'པྱཱ', u'ཎྜུ', u'ཉྥ', u'བྱཾ', u'མྠ', u'མྤ', u'མྤི', u'མྤྱ', 
          u'མྤྲཾ', u'མྦ', u'མྦོ', u'མྦྱ', u'མླི', u'ཏཾ', u'བྠཱ', u'ཪྪེ', u'རྞི', u'རྞོ',  u'རྞྞ',
          u'ཊྚི', u'ཥྚཾ', u'མུཾ',u'ཀྟི', u'གྷེེ', u'དྲུཾ', u'ལྱེ', u'ཀྷྃ',u'ཏྲཱ' , u'▶▶', u'[',
          u']', u'སྫོ', u'ཟྱ', u'ཨྠིྀ', u'བྷེེ']
### Extended
#other = [u'(', u')', u'༼',  u'༽', u'༔',  u'༑' ] # comment out above "other" if use this
english_alphbet = list(u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
english_punc = list(u'!"#$%&*+,—/:;<=>?@[]{}')
english_misc = list(u'〈〉①②③④⑤⑥⑦⑧⑨⑩¶')
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

allchars = list(chain(everything, [u'་',u'།'], num, wa_zur, roman_num, misc_glyphs, skt, retroflex, other, other2))
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


allchars_label = zip(range(len(allchars)),allchars)
#print len(allchars)
import shelve

## NORMAL DICT
s = shelve.open('../allchars_dict2')
s['allchars'] = dict((j,i) for i,j in allchars_label)
s['label_chars'] = dict((i, j) for i,j in allchars_label)
s.close()
####

### EXTENDED
#s = shelve.open('/home/zr/letters/allchars_dict_extended')
#s['allchars'] = dict((j,i) for i,j in allchars_label)
#s['label_chars'] = dict((i, j) for i,j in allchars_label)
#s.close()

#sys.exit()
#import os
fonts = glob.glob('*ttf')
sample_word = u'བསྒྲུབས'
sample_word = u'ཆ'
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
            fname = "/tmp/%s%d.png" % (fontname, randint(0,20000000))
            with open(fname, "wb") as image_file:
                    surf.write_to_png(image_file)
        
            im = Image.open(fname)
            os.remove(fname)
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
        data = map(draw_fonts, allchars_label)
    out = open(outfile, 'w')
    outf = []
    for let in data:
        for fnt in let:
            outf.append(fnt)
    out.write('\n'.join(outf))
    

if __name__ == '__main__':
    gen_img_rows('../datasets/font-draw-samples.txt')
