import numpy as np
from PIL import Image  #import scipy.misc
from ISR.models import RDN, RRDN
import time
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import png
import pydicom

img = cv.imread('D:/A.mazrouei/proje/C++/convert/convert8/convert8/8bit.tiff', -1)
#img = cv.normalize(img, None, 0.0, 1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

#img.dtype = np.float16
iRGB = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgplot = plt.imshow(img)
dimensionz = img.shape[:]
height=dimensionz[0]
width=dimensionz[1]

rdn = RRDN(c_dim=3,weights='noise-cancel')
sr_img = rdn.predict(img, by_patch_of_size=150)
output= sr_img * np.max(img) ###for out put of gan, because the input is divided by max
cv.imwrite('GAN8TAD.tiff', output)