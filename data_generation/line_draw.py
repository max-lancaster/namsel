# encoding: utf-8
import cairo
import pango
import pangocairo
from .font_draw import fontnames, trim
from random import randint
from PIL import Image
from utils import add_padding
import numpy as np
import codecs
import os
import re
allmatch = 0

# fontnames = ['serif']

def draw_line(line, outpath, spacing='normal', gtpath = None):
    global allmatch
    if spacing == 'normal':
        font_space = 0
    elif spacing == "condensed":
        font_space = -5000
    size = "40"
    w = 3000
    h = 200
    nof = 0
#     spacing = 'condensed' # this doesn't seem to work anyway
#     for spacing in ['normal', 'condensed']:
    for fontname in fontnames:
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        context = cairo.Context(surf)
        #draw a background rectangle:
        context.rectangle(0,0,w,h)
        context.set_source_rgb(1, 1, 1)
        context.fill()
#         font_map = pangocairo.cairo_font_map_get_default()
        pangocairo_context = pangocairo.CairoContext(context)
        pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
        layout = pangocairo_context.create_layout()
#         font_params = [fontname,'normal', 'normal', spacing, size]
        font_params = [fontname, size]
        font = pango.FontDescription(" ".join(font_params))
        attr = pango.AttrLetterSpacing(font_space, 0, -1)
#         attr = pango.AttrLetterSpacing(0, 0, -1)
        attrlist = pango.AttrList()
        attrlist.change(attr)
#         font.set_stretch(pango.STRETCH_CONDENSED)
#         font.set_stretch(pango.STRETCH_ULTRA_CONDENSED)
#         print font.get_stretch()
        layout.set_font_description(font)
        layout.set_attributes(attrlist)
        layout.set_text(line)
        context.set_source_rgb(0, 0, 0)
        pangocairo_context.update_layout(layout)
        pangocairo_context.show_layout(layout)
#             fname = "/tmp/%s%d.png" % (fontname, randint(0,20000000))
#         fname = "/tmp/%s.png" % ('-'.join(font_params))
        fname = outpath + fontname + '-'+ spacing + '.png'

#         if os.path.exists(fname) and os.path.exists(outpath + fontname + '.gt.txt'):
#             allmatch += 1

#         if os.path.exists(fname) and not os.path.exists(gtpath + fontname + '.gt.txt'):
#             print fname, gtpath + fontname + '.gt.txt'
#             codecs.open(gtpath + fontname + '.gt.txt', 'w', 'utf-8').write(line)
#         else:
#             nof += 1
#             continue

        
        with open(fname, "wb") as image_file:
                surf.write_to_png(image_file)
        im = Image.open(fname)
        im = im.convert('L')
        a = np.asarray(im)
        a = trim(a)/255
        a = add_padding(a, padding=2)
#         os.remove(fname)
#         Image.fromarray(a*255).show()
        Image.fromarray(a*255).save(fname)
        
        codecs.open(gtpath + fontname + '.gt.txt', 'w', 'utf-8').write(line)
    print(allmatch)
    
def draw_line2(line, font, size):
#     size = "40"
    if line.strip().endswith('་'):
        line = line.strip('་')
        line = line + '་'
    line = line.strip(' ')
    w = 8000
    h = 250
    spacing = 'normal' # this doesn't seem to work anyway
#     for spacing in ['normal', 'condensed']:

    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    context = cairo.Context(surf)
    
    #draw a background rectangle:
    context.rectangle(0,0,w,h)
    context.set_source_rgb(1, 1, 1)
    context.fill()
    
    #get font families:
    
#         font_map = pangocairo.cairo_font_map_get_default()
#    context.translate(0,0)
    
    pangocairo_context = pangocairo.CairoContext(context)
    pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
    
    layout = pangocairo_context.create_layout()
    #fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"
#         font = pango.FontDescription(fontname + " 200")
    font_params = [font,'normal', 'normal', spacing, str(size)]
    font = pango.FontDescription(" ".join(font_params))
#     font.set_stretch(pango.STRETCH_CONDENSED)
#             else:
#                 font = pango.FontDescription(fontname + " bold 200")
        
    layout.set_font_description(font)
    
    layout.set_text(line)
    context.set_source_rgb(0, 0, 0)
    pangocairo_context.update_layout(layout)
    pangocairo_context.show_layout(layout)
#             fname = "/tmp/%s%d.png" % (fontname, randint(0,20000000))
    fname = "/tmp/%s.png" % ('-'.join(font_params))
#         fname = outpath + fontname + '.png'
#         codecs.open(outpath + fontname + '.gt.txt', 'w', 'utf-8').write(line)
    with open(fname, "wb") as image_file:
            surf.write_to_png(image_file)
            
    im = Image.open(fname)
#         im = im.convert('L')
    im = im.convert('L')
    a = np.asarray(im, 'f')
    os.remove(fname)
    return a
#         a = trim(a)/255
#         a = add_padding(a, padding=2)
#         Image.fromarray(a*255).save(fname)

def draw_lines_from_file(fl, outdir='generated-line-imgs', \
         outdir_gt='generated-line-imgs-gt', spacing="normal", limit=None):
    fl = codecs.open(fl, 'r', 'utf-8')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(outdir_gt):
        os.mkdir(outdir_gt)

    for i, line in enumerate(fl):
        if limit and i > limit:
            break
        line = line.strip(' ་')
        if not line:
            continue
        print(line)
#         line = next(fl).strip()
        outfilepath = os.path.join(outdir, '%06d-' % i) 
        gtoutfilepath = os.path.join(outdir_gt, '%06d-' % i) 
        draw_line(' ' + line, outfilepath, gtpath=gtoutfilepath, spacing=spacing)

if __name__ == '__main__':
    import os
#     fl = codecs.open('/media/zr/zr-mechanical/Dropbox/sera-khandro-sample.txt', 'r', 'utf-8')
#     fl = codecs.open('/media/zr/zr-mechanical/eKangyur-FINAL-sources-20141023/eKangyur/W4CZ5369/sources/allkangyurlines-nolinenum.txt', 'r', 'utf-8')
#     fl = codecs.open('/media/zr/mech2/DownloadsOverflow/chintamanishatpadi_D.txt', 'r', 'utf-8')
#     outdir = '/media/zr/mech2/sera-khandro-lines'
#     fl = codecs.open('/tmp/testlines', 'r', 'utf-8')
    fl = '/media/zr/zr-mechanical/Dropbox/sera-khandro-sample.txt'
    draw_lines_from_file(fl, '/media/zr/mech2/seratest', '/media/zr/mech2/seratest-gt', spacing='condensed', limit=20)
    import sys; sys.exit()
    outdir = '/tmp/lines'
#     outdir = '/media/zr/mech2/kangyur-lines'
#     gtoutdir = '/media/zr/mech2/kangyur-lines-gt'
#     outdir = '/media/zr/mech2/skt-lines'
#     gtoutdir = '/media/zr/mech2/skt-lines-gt'
    gtoutdir = '/tmp/lines-gt'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(gtoutdir):
        os.mkdir(gtoutdir)
#     for i in range(10):
    for i, line in enumerate(fl):
        if i > 30000:
            break
        line = line.strip()
        if not line:
            continue
        print(line)
#         line = next(fl).strip()
        outfilepath = os.path.join(outdir, '%06d-' % i) 
        gtoutfilepath = os.path.join(gtoutdir, '%06d-' % i) 
        draw_line(' ' + line, outfilepath, gtpath=gtoutfilepath)
