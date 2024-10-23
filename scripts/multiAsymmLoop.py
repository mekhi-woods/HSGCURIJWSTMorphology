import numpy as np
from PIL import Image as im
import math
import utilities as utils
import scipy
import sys
import matplotlib.pyplot as plt
import os

def multiAsymm(arr, center):
    (yMax, xMax) = arr.shape
    r = 1
    centerList = [center, (center[0] + r, center[1]), (center[0] - r, center[1]),
                  (center[0], center[1] + r), (center[0], center[1] - r),
                  (center[0] + r, center[1] + r), (center[0] - r, center[1] + r),
                  (center[0] - r, center[1] - r), (center[0] + r, center[1] - r)]

    AList = []
    sqrdAList = []
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

        arr180 = scipy.ndimage.rotate(croppedArr, 180)

        numerator1 = np.sum(abs(croppedArr - arr180))
        denominator1 = 2 * np.sum(abs(croppedArr))
        A = numerator1 / denominator1
        # A /= np.sqrt(xMax * yMax)
        AList.append(A)

        numerator2 = np.sum((croppedArr - arr180)**2)
        denominator2 = 2 * np.sum(croppedArr**2)
        sqrdA = numerator2 / denominator2
        # sqrdA /= np.sqrt(xMax * yMax)
        sqrdAList.append(sqrdA)

    return (AList, sqrdAList)

path = 'files/SMACSGalPics/imsAsArrCrop'
fileList = os.listdir(path)

idList = []
sqrdList = []
absAList = []
absListList = []
sqrdListList = []
for file in fileList:
    if file in ['.DS_Store']:
        continue

    if file[:2] == '00':
        idList.append(file[2:5])
    elif file[:1] == '0':
        idList.append(file[1:5])
    else:
        idList.append(file[0:5])

    galaxyArr = np.load(f'{path}/{file}')
    originXY = np.where(galaxyArr == galaxyArr.max())
    originXY = (originXY[1][0], originXY[0][0])
    (AList, sqrdAList) = multiAsymm(galaxyArr, originXY)

    A = min(AList)
    sqrdA = min(sqrdAList)
    print('max:', max(AList))
    print(A)
    print(sqrdA)
    print(file)
    print()

    absListList.append(AList)
    sqrdListList.append(sqrdAList)
    absAList.append(A)
    sqrdList.append(sqrdA)

plt.scatter(absAList, sqrdList)
# print(np.array(err1List))
# plt.errorbar(absAList, sqrdAList, xerr=err1List)
plt.xlabel('Abs. Val. Method')
plt.ylabel('Squared Method')
plt.show()
plt.clf()

plt.hist(absAList, bins='auto')
plt.title('Absolute Value A')
plt.show()
plt.clf()

plt.hist(sqrdList, bins='auto')
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


plt.scatter(zList, absAList)
plt.title('Absolute Value A')
plt.legend(['absA', 'sqrdA'])
plt.ylim(-0.001, 0.04)
plt.xlabel('z')
plt.ylabel('A')
plt.show()
plt.clf()

plt.scatter(zList, sqrdList)
plt.title('Squared A')
plt.legend(['absA', 'sqrdA'])
plt.xlabel('z')
plt.ylabel('A')
plt.show()
plt.clf()

for i in range(len(absListList) - 1):
    for j in range(len(absListList[i]) - 1):
        plt.scatter(zList[i], absListList[i][j])

plt.xlabel('z')
plt.ylabel('Abs. Val. A')
# plt.ylim(-0.001, 0.15)
plt.show()
plt.clf()

for i in range(len(absListList) - 1):
    for j in range(len(absListList[i]) - 1):
        plt.scatter(zList[i], sqrdListList[i][j])

plt.xlabel('z')
plt.ylabel('sqaured A')
# plt.ylim(-0.001, 0.15)
plt.show()



