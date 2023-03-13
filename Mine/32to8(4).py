import cv2 as cv
i = cv.imread('D:/A.mazrouei/A.mazrouei-Cargo/Data/new bandar/698599.tiff', -1)
i = i % 4294967296
n4 = i % 256
i = i / 256
n3 = i % 256
i = i / 256
n2 = i % 256
n1 = i / 256
