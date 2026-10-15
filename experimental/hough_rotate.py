from PIL import Image
import cv2 as cv
import numpy as np
import os

def rotation(a, i):
    center = (a.shape[1]/2.0, a.shape[0]/2.0)
    trs_mat = cv.getRotationMatrix2D(center, i, 1)
    b = cv.warpAffine(a, trs_mat, (a.shape[1],a.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=255)
    return b

degree_scaler = (180/np.pi)

def get_avg_angle(a, draw=False):
    lines = cv.HoughLinesP(a,  1, np.pi/180, 1, minLineLength=a.shape[1]*.30, maxLineGap=50)
    angles = []
    if lines is not None:
        for line in lines[0]:
            if draw:
                
                cv.line(a, tuple(line[0:2]), tuple(line[2:]), 1, thickness=2)
            angle = np.arctan2(line[3]-line[1], line[2]-line[0])
            angles.append(angle)
        return np.mean(angles)*degree_scaler
    else:
        return 0
    

def rotate_img(src):
    '''Rotate an image given a file path'''
    print(src)
    im1 = Image.open(src).convert('L')
    a = np.asarray(im1)/255
    a = ((a*-1) + 1).astype(np.uint8)
#    avg_angle = get_avg_angle(a, draw=True)
    avg_angle = get_avg_angle(a)
    print(avg_angle)
    a = rotation(a, avg_angle)
    a = (a*-1) + 1
    Image.fromarray(a*255).convert('L').show()
    
def rotate_img_arr(a):
    '''Rotate an image given an array'''
    a = a/255
    a = ((a*-1) + 1).astype(np.uint8)
#    avg_angle = get_avg_angle(a, draw=True)
    avg_angle = get_avg_angle(a)
    a = rotation(a, avg_angle)
    return 255*((a*-1) + 1)
    
def rotate_img_cli(src):
    print(src)
    im1 = Image.open(src).convert('L')
    a = np.asarray(im1)/255
    a = ((a*-1) + 1).astype(np.uint8)
    avg_angle = get_avg_angle(a)
    a = rotation(a, avg_angle)
    a = (a*-1) + 1
    Image.fromarray(a*255).convert('L').save('%d_%s' % (np.ceil(avg_angle), os.path.basename(src).split('.')[0]+'.tif'), compression='LZW')

if __name__ == '__main__':
    
    import sys
    infile = sys.argv[1]
    rotate_img_cli(infile)
    
