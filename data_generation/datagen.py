#encoding: utf-8

from fonts.font_draw import gen_img_rows
from fonts.line_draw import draw_lines_from_file
import sys


# TODO:
# -Add support for listing fonts to use in text generation
# -Add support for specifying font emphasis and weight
# -Allow for specifying output directories for lines
# -Make language agnostic
# -Add option to use ocropy-linegen as a backend
# -Add support for add random distortion to images


def generate_lines(fl, spacing='normal'):
    '''Generate images from text-lines in a file

Args:
    fl: the name of a file containing text to be drawn
    spacing: the spacing of characters relative to on another in the horizontal
    direction. Options are: normal, condensed, or all
    
Returns:
    Generated images are saved on disk in the directory where this script is
    run
    
    '''
    if spacing == 'all':
        for s in ['normal', 'condensed']:
            draw_lines_from_file(fl, spacing=s)
    else:
        draw_lines_from_file(fl, spacing=s)

def generate_stacks(outfilename):
    '''Generates a simple text file containing pixel values for all sample
stack data, machine-generated and hand-labeled. Each row of the file is a
length 1025 list of 1s and 0s representing a 32x32 pixel image, together with
an integer label. 

Args:
    outfilename: Name of the file where all sample images and labels are written
    
Returns:
    None
    '''
    gen_img_rows(outfilename)

def generate_syllables():
    '''Not implemented. Use generate_lines instead with a file of syllables
rather than entire lines of texts'''
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    ### Positional arguments
    parser.add_argument("type", help="The type of sample to be generated.\
 Options: stacks, syllables, or lines", type=str)
    
    ### Optional arguments
    parser.add_argument('--outfile', help='Name of the file the flattened\
 32x32 stack images will be written to. For use with the "stacks" option only')
    parser.add_argument('--filename', help='A filename for a file containing\
 the lines to be generated as images', default=None)
    parser.add_argument('--spacing', help='Amount of spacing to use between stacks.',
                choices=['all', 'normal', 'condensed'], default=None)
    
    args = parser.parse_args()
    
    if args.type == 'lines' or args.type == 'syllables':
        if not args.filename:
            print('Error: --filename not specified. You must specify a file \
containing lines of text to be generated')
            sys.exit()
        generate_lines(args.linefl, spacing=(args.spacing or "all"))
    elif args.type == 'stacks':
        if not args.outfile:
            outfile = '/home/zr/letters/experimental3232.txt'
        else: 
            outfile = args.outfile
        generate_stacks(outfile)
#     elif args.type == 'syllables':
#         generate_syllables()
    
