import numpy as np
from PIL import Image  #import scipy.misc
from ISR.models import RDN, RRDN
import time
import cv2 as cv

#Color
#High
#1-3x
#1-4x
#1-4y
#1-3y

img5 = cv.imread('E:/flashhhhhhhhhhhhhhhhh/15.01.00/7High.tiff')
#img3 = cv.imread('E:/flashhhhhhhhhhhhhhhhh/15.01.00/1Color.tiff')

#height, width = img2.shape[:2]
#height2, width2 = img3.shape[:2]

img1 = cv.resize(img5, None, fx = 1/3, fy = 1, interpolation = cv.INTER_CUBIC)
img2 = cv.resize(img5, None, fx = 1/3, fy = 1, interpolation = cv.INTER_CUBIC)
img3 = cv.resize(img5, None, fx = 1/4, fy = 1, interpolation = cv.INTER_CUBIC)
img4 = cv.resize(img5, None, fx = 1/4, fy = 1, interpolation = cv.INTER_CUBIC)

rdn = RRDN(c_dim=3,weights='gans')



sr_img2 = rdn.predict(img1)
sr_img2 = cv.resize(sr_img2, None, fx = 1, fy = 1/4, interpolation = cv.INTER_CUBIC)
cv.imwrite('7High-3x.tiff',sr_img2)
#cv.imwrite('HighImageMainViewGan2.tiff',sr_img2)

sr_img2 = rdn.predict(img2)
sr_img2 = cv.resize(sr_img2, None, fx = 1, fy = 1/4, interpolation = cv.INTER_CUBIC)
cv.imwrite('7High-4x.tiff',sr_img2)


sr_img2 = rdn.predict(img3)
sr_img2 = cv.resize(sr_img2, None, fx = 1, fy = 1/4, interpolation = cv.INTER_CUBIC)
cv.imwrite('7High-4y.tiff',sr_img2)


#start = time.process_time()
sr_img2 = rdn.predict(img4)
sr_img2 = cv.resize(sr_img2, None, fx = 1, fy = 1/3, interpolation = cv.INTER_CUBIC)
cv.imwrite('7High-3y.tiff',sr_img2)
#width = int(sr_img3.shape[1])
#height = int(890)
#dim = (width, height)
#End = (time.process_time() - start)/10
