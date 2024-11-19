import numpy as np
from PIL import Image as im
import math
import utilities as utils
import scipy
import sys
import matplotlib.pyplot as plt
import os
from photutils.isophote import EllipseGeometry

def getCenter(x, y):
    g = EllipseGeometry(x0=x, y0=y, sma=1, eps=0.01, pa=(120 / 180.0) * np.pi)  # Make the outline of the initial guess using                                                                                             # above parameters
    g.find_center(galaxyArr)
    center = [[int(g.y0)], [int(g.x0)]]  # Grab updated centers

    return center

path = 'files/SMACSGalPics/arraysWError'
fileList = os.listdir(path)
# print(fileList)

sqrdAList = []
absAList = []
err1List = []
err2List = []
idList = []
for file in fileList:
    if file in ['.DS_Store']:
        continue

    if file[:2] == '00':
        idList.append(file[2:5])
    elif file[:1] == '0':
        idList.append(file[1:5])
    else:
        idList.append(file[0:5])

    # galaxyArr = np.load(f'{path}/{file}')
    galWithErr = np.load(f'{path}/{file}')
    galaxyArr = galWithErr[:, :, 0]
    err = galWithErr[:, :, 1]
    origImage = im.fromarray(galaxyArr)
    # origImage.show()

    # Finding center of galaxy
    # print(galaxyArr.shape)
    (yMax, xMax) = galaxyArr.shape
    # centerBoxX = round(0.9 * (0.5*xMax))  # Narrowing the scope of the search for brightest point
    # centerBoxY = round(0.9 * (0.5*yMax))  # to avoid centering around wrong object
    # originVal = galaxyArr[centerBoxX:(xMax - centerBoxX), centerBoxY:(yMax - centerBoxY)].max()
    # originXY = np.where(galaxyArr == originVal)
    originXY = np.where(galaxyArr == galaxyArr.max())
    originX = originXY[1][0]
    originY = originXY[0][0]

    # originXY = getCenter(originX, originY)
    # originX = originXY[1][0]
    # originY = originXY[0][0]
    # print(galaxyArr)
    # g = EllipseGeometry(x0=originX, y0=originY, sma=1, eps=0.01, pa=(120 / 180.0) * np.pi)  # Make the outline of the initial guess using                                                                                             # above parameters
    # g.find_center(galaxyArr)
    # originXY = getCenter(originX, originY)  # Grab updated centers
    # print(originXY)
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
    errCropped = err[originY-yMargin:originY+(yMargin+1), originX-xMargin:originX+(xMargin+1)]

    # print(errCropped.shape)
    # print(galaxyArrCropped.shape)
    # print()
    # plt.imshow(galaxyArrCropped)
    # plt.show()
    origImageCropped = im.fromarray(galaxyArrCropped)

    # Rotating image
    galaxyArr180 = scipy.ndimage.rotate(galaxyArrCropped, 180)
    err180 = scipy.ndimage.rotate(errCropped, 180)
    image180 = im.fromarray(galaxyArr180)
    # imageAdd = im.fromarray(galaxyArrCropped + galaxyArr180)
    # imageAdd.show()

    # Displaying images
    # origImage.show()
    # origImageCropped.show()
    # image180.show()
    # Asymmetry calculations without square
    # subtracting rotated from original
    numerator1 = np.sum(abs(galaxyArrCropped - galaxyArr180))
    errNum1 = utils.errorSum(np.sqrt(errCropped ** 2 + err180 ** 2))
    # Summing original image
    denominator1 = 2 * np.sum(abs(galaxyArrCropped))
    # errDenom1 = 2 * utils.errorSum(np.sqrt((errCropped/galaxyArrCropped)**2 * 2) * denominator1)
    errDenom1 = 2 * utils.errorSum(errCropped)

    # print(denominator1, file)
    A = numerator1 / denominator1
    # A /= np.sqrt(yMax * xMax)
    totalErr1 = np.sqrt((errNum1 / numerator1) ** 2 + (errDenom1 / denominator1) ** 2) * A
    # A /= np.sqrt(len(galaxyArrCropped))
    absAList.append(A)
    err1List.append(totalErr1)
    # print(f'Absolute value A: {A}')

    # Asymmetry calculations with square
    # subtracting rotated from original and squaring
    numerator2 = np.sum((galaxyArrCropped - galaxyArr180)**2)
    # errNum2 = utils.errorSum(np.sqrt(2 * (np.sqrt(errCropped**2 + err180**2)/(galaxyArrCropped - galaxyArr180)))) * numerator2
    subtraction = np.sqrt(errCropped**2 + err180**2)
    square = np.sqrt((subtraction/(galaxyArrCropped - galaxyArr180))**2 * 2) * (galaxyArrCropped - galaxyArr180)**2
    errNum2 = utils.errorSum(square)

    # Summing original image
    denominator2 = 2 * np.sum(galaxyArrCropped**2)
    errDenom2 = 2 * utils.errorSum(np.sqrt((errCropped/galaxyArrCropped)**2 * 2) * galaxyArrCropped**2)

    sqrdA = numerator2 / denominator2
    totalErr2 = np.sqrt((errNum2 / numerator2) ** 2 + (errDenom2 / denominator2) ** 2) * sqrdA
    # sqrdA /= np.sqrt(yMax * xMax)
    sqrdAList.append(np.sqrt(sqrdA))
    err2List.append(totalErr2)
    # print(f'Squared A: {np.sqrt(A)}')

    # Displaying images
    subImageABS = im.fromarray(abs(galaxyArrCropped - galaxyArr180))
    subImageSQR = im.fromarray((galaxyArrCropped - galaxyArr180)**2)
    # subImageABS.show()
    # subImageSQR.show()

    # Organizing images by low, medium, and high asymmetry
    if A <= 0.03:
        plt.imsave(f'files/SMACSGalPics/lowA/{file[:-4]}CROP.png', galaxyArrCropped, cmap='gray', vmin=0.2, vmax=0.6)
        plt.imsave(f'files/SMACSGalPics/lowA/{file[:-4]}SUB.png', abs(galaxyArrCropped - galaxyArr180), cmap='gray')
    elif 0.03 < A <= 0.1:
        plt.imsave(f'files/SMACSGalPics/medA/{file[:-4]}CROP.png', galaxyArrCropped, cmap='gray', vmin=0.2, vmax=0.6)
        plt.imsave(f'files/SMACSGalPics/medA/{file[:-4]}SUB.png', abs(galaxyArrCropped - galaxyArr180), cmap='gray')
    else:
        # histCount, edge, tmp = plt.hist(galaxyArrCropped.flatten(), bins=100)
        plt.imsave(f'files/SMACSGalPics/highA/{file[:-4]}CROP.png', galaxyArrCropped, cmap='gray', vmin=0.2, vmax=0.6)
        plt.imsave(f'files/SMACSGalPics/highA/{file[:-4]}SUB.png', abs(galaxyArrCropped - galaxyArr180), cmap='gray')

plt.scatter(absAList, sqrdAList)
# print(np.array(err1List))
# plt.errorbar(absAList, sqrdAList, xerr=err1List, fmt='o')
plt.xlabel('Abs. Val. Method')
plt.ylabel('Squared Method')
plt.show()
plt.clf()

plt.hist(absAList, bins='auto')
plt.title('Absolute Value A')
plt.show()
plt.clf()

plt.hist(sqrdAList, bins='auto')
plt.title('Squared A')
plt.show()
plt.clf()

# getting z values for each target based on ID
zList = []
f = open('files/SMACSTargets.txt')
count = 0
for line in f:
    if count <= 2:
        count += 1
        continue

    (ID, RA, dec, zGrism, zErr, zSpec) = line.split(' ')
    if ID in idList:
        zList.append(zGrism)
        idList.remove(ID)

zArr = np.array(zList)
zList = list(map(float, zArr))

print(err2List)
# plt.errorbar(zList, sqrdAList, yerr=err2List, fmt='o')
plt.errorbar(zList, absAList, yerr=err1List, fmt='o')
# plt.legend(['absA', 'sqrdA'])
plt.xlabel('z')
plt.ylabel('A')
plt.show()
plt.clf()

plt.scatter(zList, absAList)
# plt.legend(['absA', 'sqrdA'])
plt.xlabel('z')
plt.ylabel('Abs. Val. A')
plt.show()
