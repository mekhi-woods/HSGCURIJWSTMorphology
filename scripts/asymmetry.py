import numpy as np
from PIL import Image as im
import math
import utilities as utils
import scipy
import sys
import matplotlib.pyplot as plt
from photutils.isophote import EllipseGeometry

galaxyArr = np.load('files/SMACSGalPics/imsAsArrCrop/00898galaxy22.npy')
origImage = im.fromarray(galaxyArr)
# origImage.show()

# Finding center of galaxy
(yMax, xMax) = galaxyArr.shape
print(galaxyArr.shape)
originXY = np.where(galaxyArr == galaxyArr.max())
print(originXY)
originX = originXY[1][0]
originY = originXY[0][0]
g = EllipseGeometry(x0=originX, y0=originY, sma=10, eps=0.01, pa=(120 / 180.0) * np.pi)    # Make the outline of the initial guess using                                                                                             # above parameters
g.find_center(galaxyArr)
originXY = [[int(g.y0)], [int(g.x0)]] # Grab updated centers
print(originXY)
# print('adsfasdfsdf:', center)
# print(originXY)

origToEdgeX = xMax - originX
origToEdgeY = yMax - originY

# Finding margins around center point
if origToEdgeX < originX:
    xMargin = round(origToEdgeX * 0.9)
else:
    xMargin = round(originX * 0.9)

if origToEdgeY < originY:
    yMargin = round(origToEdgeY * 0.9)
else:
    yMargin = round(originY * 0.9)

# Cropping image around center of galaxy scaled
galaxyArrCropped = galaxyArr[originY-yMargin:originY+(yMargin+1), originX-xMargin:originX+(xMargin+1)]
# plt.imshow(galaxyArrCropped)
# plt.show()
origImageCropped = im.fromarray(galaxyArrCropped)

# Rotating image
galaxyArr180 = scipy.ndimage.rotate(galaxyArrCropped, 180)
image180 = im.fromarray(galaxyArr180)
# imageAdd = im.fromarray(galaxyArrCropped + galaxyArr180)
# imageAdd.show()

# Displaying images
# origImage.show()
# origImageCropped.show()
# image180.show()

# Asymmetry calculations without square
# subtracting rotated from original
numerator = sum(sum(abs(galaxyArrCropped - galaxyArr180)))
# Summing original image
denominator = 2 * sum(sum(abs(galaxyArrCropped)))

A = numerator / denominator
print(f'Absolute value A: {A}')

# Asymmetry calculations with square
# subtracting rotated from original and squaring
numerator = sum(sum((galaxyArrCropped - galaxyArr180)**2))
# Summing original image
denominator = 2 * sum(sum(galaxyArrCropped**2))

A = numerator / denominator
# A = np.sqrt(A)
A /= np.sqrt(galaxyArrCropped.shape[0] * galaxyArrCropped.shape[1])
print(f'Squared A: {np.sqrt(A)}')
# print(f'Squared A: {A}')

# Asymmetry using mock galaxy
mock = utils.mockGalaxy(2, 2, 315, galaxyArrCropped.shape[1], galaxyArrCropped.shape[0])
numerator = sum(sum(abs(mock - galaxyArrCropped)))
denominator = 2 * sum(sum(abs(galaxyArrCropped)))
A = numerator / denominator
print(f'Mock A: {A}')


# Displaying images
# subImageABS = im.fromarray(abs(galaxyArrCropped - galaxyArr180))
# subImageSQR = im.fromarray((galaxyArrCropped - galaxyArr180)**2)
# subImageABS.show()
# subImageSQR.show()
plt.imshow(galaxyArrCropped, cmap='gray', vmin=0.2, vmax=0.6)
plt.show()

plt.imshow(abs(galaxyArrCropped - galaxyArr180), cmap='gray', vmin=0.0, vmax=0.1)
plt.show()

