import numpy as np
from PIL import Image  #import scipy.misc
from ISR.models import RDN, RRDN
import cv2 as cv
import os.path
import sys

pathimg=sys.argv[1]

if not os.path.isfile(pathimg):
    #ignore if no such file is present.
    print("***File Not Founded***")
    pass
img = cv.imread(pathimg, -1)

imgTH = cv.threshold(img,1000,1, cv.THRESH_BINARY_INV)
imgMult = cv.multiply(img, imgTH[1])
BinayMask = (imgMult + 1000 * (1 - imgTH[1]))

imgNorm = cv.normalize(BinayMask, None, 0.0, 1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
iRGB = cv.cvtColor(imgNorm, cv.COLOR_GRAY2BGR)
dimensionz = img.shape[:]

rdn = RRDN(c_dim=3,weights='gans')
sr_img = rdn.predict(iRGB, by_patch_of_size=150)

grayGAN = cv.cvtColor(sr_img, cv.COLOR_BGR2GRAY)
dimensionzGAN = grayGAN.shape[:]
cv.imwrite(os.path.join("E:/safavi/Super Resolution" , 'GANs.tiff'), grayGAN)

print("Inputed link:", pathimg)
print("Shape Of Input Image:", dimensionz)
print("Shape Of Output Image:", dimensionzGAN)