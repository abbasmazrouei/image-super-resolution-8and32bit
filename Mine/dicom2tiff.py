import numpy as np
import png
import pydicom

ds = pydicom.dcmread('C:/Users/A.mazrouei/Desktop/noise/1/4.dcm')

shape = ds.pixel_array.shape

# Convert to float to avoid overflow or underflow losses.
image_2d = ds.pixel_array.astype(float)

# Rescaling grey scale between 0-255
image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

# Convert to uint
image_2d_scaled = np.float32(image_2d_scaled)

# Write the PNG file
with open('C:/Users/A.mazrouei/Desktop/noise', 'ws') as tiff_file:
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(tiff_file, image_2d_scaled)