import csv

import numpy as np
from PIL import Image as im
import math
import utilities as utils
import scipy
import sys
import matplotlib.pyplot as plt
import os

def absoA(cropped, rotated, croppedErr, rotatedErr):
    numerator = np.sum(abs(cropped - rotated))
    errNum = utils.errorSum(np.sqrt(croppedErr**2 + rotatedErr**2))

    denominator = 2 * np.sum(abs(cropped))
    errDenom = 2 * utils.errorSum(croppedErr)

    asymmetry = numerator / denominator
    totalErr = np.sqrt((errNum / numerator) ** 2 + (errDenom / denominator) ** 2) * asymmetry

    # if math.isnan(totalErr):
    #     print(errNum, errDenom)
    #     print(np.sqrt(croppedErr**2 + rotatedErr**2))
    #     print(croppedErr, rotatedErr)
    #     print(asymmetry)
    #     print()


    return asymmetry, totalErr

def squaredA(cropped, rotated, croppedErr, rotatedErr):
    numerator = np.sum((cropped - rotated)**2)
    sub = cropped - rotated
    subErr = np.sqrt(croppedErr**2 + rotatedErr**2)
    squareErr = sub**2 * np.sqrt(2 * (subErr / sub)**2)
    errNum = utils.errorSum(squareErr)

    denominator = 2 * np.sum(cropped**2)
    errDenom = 2 * utils.errorSum(np.sqrt(2 * (croppedErr / cropped)**2) * cropped**2)

    asymmetry = numerator / denominator
    # totalErr = errNum / errDenom
    # print(totalErr)

    return asymmetry



# This is slow and stupid, but oh well
def coordsFromID(file, id):
    with open(file, 'r') as f:
        count = 0
        for line in f:
            if count <= 2:
                count += 1
                continue

            (ID, RA, Dec, z, *_) = line.split(' ')
            if id == ID:
                return RA, Dec, z

def multiAsymm(arr, errArr, center):
    (yMax, xMax) = arr.shape
    r = 1
    centerList = [center, (center[0] + r, center[1]), (center[0] - r, center[1]),
                  (center[0], center[1] + r), (center[0], center[1] - r),
                  (center[0] + r, center[1] + r), (center[0] - r, center[1] + r),
                  (center[0] - r, center[1] - r), (center[0] + r, center[1] - r)]

    AStr = ''
    errAStr = ''
    sqrdAStr = ''
    for point in centerList:
        origToEdgeX = xMax - point[0]
        origToEdgeY = yMax - point[1]

        if origToEdgeX < center[0]:
            xMargin = round(point[0] * 0.9)
        else:
            xMargin = round(point[0] * 0.9)

        if origToEdgeY < point[1]:
            yMargin = round(origToEdgeY * 0.9)
        else:
            yMargin = round(point[1] * 0.9)

        croppedArr = arr[point[0] - xMargin:point[0] + (xMargin + 1),
                           point[1] - yMargin:point[1] + (yMargin + 1)]

        croppedErr = errArr[point[0] - xMargin:point[0] + (xMargin + 1),
                     point[1] - yMargin:point[1] + (yMargin + 1)]

        arr180 = scipy.ndimage.rotate(croppedArr, 180)
        err180 = scipy.ndimage.rotate(croppedArr, 180)

        A, errA = absoA(croppedArr, arr180, croppedErr, err180)
        # A /= np.sqrt(xMax * yMax)
        AStr += str(A) + ' '
        errAStr += str(errA) + ' '

        sqrdA = squaredA(croppedArr, arr180, croppedErr, err180)
        # sqrdA /= np.sqrt(xMax * yMax)
        sqrdAStr += str(sqrdA) + ' '

    return (AStr, errAStr, sqrdAStr)

path = 'files/SMACSGalPics/arraysWError'
fileList = os.listdir(path)

f = open('files/finalFile.csv', 'w')
header = ['ID', 'RA', 'dec', 'z', 'absAList',
          'sqrdAList', 'errAList', 'absAMin',
          'sqrdAMin', 'absAMinErr']
writer = csv.writer(f)
writer.writerow(header)
for file in fileList:
    infoList = []
    if file in ['.DS_Store']:
        continue

    if file[:2] == '00':
        infoList.append(file[2:5])
    elif file[:1] == '0':
        infoList.append(file[1:5])
    else:
        infoList.append(file[0:5])

    RA, dec, z = coordsFromID('files/SMACSTargets.txt', infoList[0])
    infoList.append(RA)
    infoList.append(dec)
    infoList.append(z)

    galWithErr = np.load(f'{path}/{file}')
    galaxyArr = galWithErr[:, :, 0]
    err = galWithErr[:, :, 1]

    originXY = np.where(galaxyArr == galaxyArr.max())
    originXY = (originXY[1][0], originXY[0][0])

    (AStr, errAStr, sqrdAStr) = multiAsymm(galaxyArr, err, originXY)
    infoList.append(AStr)
    infoList.append(sqrdAStr)
    infoList.append(errAStr)

    AList = AStr.split(' ')
    AList = list(map(float, AList[:-1]))
    A = min(AList)
    infoList.append(A)

    sqrdAList = sqrdAStr.split(' ')
    sqrdAList = list(map(float, sqrdAList[:-1]))
    sqrdA = min(sqrdAList)
    infoList.append(sqrdA)

    errAList = errAStr.split(' ')
    errAList = list(map(float, errAList[:-1]))
    errA = errAList[AList.index(A)]
    infoList.append(errA)

    writer.writerow(infoList)

f.close()

