import numpy as np
from PIL import Image  #import scipy.misc
from ISR.models import RDN, RRDN
import time
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = cv.imread('D:/A.mazrouei/proje/C++/convert/convert8/convert8/8bit.tiff', -1)
#img = cv.normalize(img, None, 0.0, 1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


###
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
iRGB = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#img2 = np.zeros_like(img)
#img2 = np.stack((img,)*3, axis=-1)
#img2 = cv2.merge((imgray,imgray,imgray))
###
img.max()
img.min()
#img.dtype = np.float16
#img = mpimg.imread('your_image.png')
#imgplot = plt.imshow(img)
#plt.show()
###
#img = Image.open(image_file)
#lr_img = np.array(img)
###
dimensionz = img.shape[:]
height=dimensionz[0]
width=dimensionz[1]

rdn = RRDN(c_dim=3,weights='gans')
sr_img = rdn.predict(img, by_patch_of_size=150)
output= sr_img * np.max(img) ###for out put of gan, because the input is divided by max
cv.imwrite('GAN8TAD.tiff', output)


#if (width >=2200):
    
    #img = cv.resize(img1, None, fx = 1/4, fy = 1/4, interpolation = cv.INTER_LINEAR)
   # rdn = RRDN(c_dim=3,weights='gans')
  #  sr_img1 = rdn.predict(img1)
 #   output = cv.resize(sr_img1, None, fx = 1/2, fy = 1/4, interpolation = cv.INTER_CUBIC)
#    cv.imwrite('H4j.tiff', output)
    
#else:
    
    #img1 = cv.resize(img1, None, fx = 1/4, fy = 1/4, interpolation = cv.INTER_AREA)
   # rdn = RRDN(c_dim=3,weights='gans')
  #  sr_img1 = rdn.predict(img1)
 #   output = cv.resize(sr_img1, None, fx = 1/2, fy = 1/4, interpolation = cv.INTER_CUBIC)
#    cv.imwrite('H4jj.tiff', output)




##rdn = RRDN(c_dim=3,weights='gans')
##img3 = cv.resize(img3, None, fx = 1, fy = 1, interpolation = cv.INTER_CUBIC)
#cv.imwrite('HighImageMainViewGan2.tiff',sr_img2)

#sr_img2 = rdn.predict(img2)
#cv.imwrite('img2.tiff',sr_img2)
#cv.imwrite('HighImageMainViewGan2.tiff',sr_img2)

##start = time.process_time()

##sr_img3 = rdn.predict(img3)

#width = int(sr_img3.shape[1])
#height = int(890)
#dim = (width, height)

##sr_img3 = cv.resize(sr_img3, None, fx = 1/2, fy = 1/4, interpolation = cv.INTER_CUBIC)

# resize image
#resized = cv.resize(sr_img3, dim, interpolation = cv.INTER_CUBIC)
#cv.imwrite('HighImageMainViewGan3.tiff',resized)
##cv.imwrite('H1.tiff', sr_img3)

##End = (time.process_time() - start)/10