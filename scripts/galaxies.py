import utilities as utils
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import scipy
import data
import time

class Galaxy(object):

    # Initializes the galaxy using the ID
    # Pulls coordinates
    def __init__(self, ID, RA, DEC, z, obName):
        pass

    # Finds galaxy from fits file
    # Crops image and error around center
    # Option to display image
    def isolate(self, display=False):
        pass

    # Calculates asymmetry
    # Options for absolute method, squared method,
    # and changing the center to get different A values
    def asymmetry(self, abso=True, sqrd=False, multi=False):
        pass

    # Function for calculating absolute asymmetry method
    def absoluteA(self, croppedArr, arr180, croppedErr, err180):
        pass

    # Function for calculating squared asymmetry method
    def squaredA(self, croppedArr, arr180, croppedErr, err180):
        pass

    # Plots image
    def showImage(self):
        pass

    # Returns RA and DEC
    def getCoords(self):
        pass

    # Returns z(redshift)
    def getRedshift(self):
        pass

    # Returns the ID
    def getID(self):
        pass

    # Returns object name
    def getName(self):
        pass



class RegularGalaxy(Galaxy):
    def __init__(self, ID, RA, DEC, z, obName):
        self.ID = ID
        self.infitsList = data.infitsList
        self.RA = RA
        self.DEC = DEC
        self.obName = obName
        self.z = z

    def isolate(self, display=False):
        hdfra = float(self.RA) * u.deg
        hdfdec = float(self.DEC) * u.deg

        for infits in self.infitsList:
            # time.sleep(1)
            galaxy, err = utils.isolateGal(hdfra, hdfdec, infits)
            if len(galaxy) > 1:
                yMax, xMax = galaxy.shape
                if yMax == 0 or xMax == 0:
                    # print(f"Dimension is 0: (ID-{self.ID}, {yMax}, {xMax})")
                    continue
                elif 0.0 in galaxy:
                    continue
                else:
                    originXY = np.where(galaxy == galaxy.max())
                    center = (originXY[1][0], originXY[0][0])
                    (yMax, xMax) = galaxy.shape
                    dimensions = (yMax, xMax, center)

                    origToEdgeX = xMax - center[0]
                    origToEdgeY = yMax - center[1]

                    # Finding margins around center point
                    if origToEdgeX < center[0]:
                        xMargin = round(origToEdgeX * 0.9)
                    else:
                        xMargin = round(center[0] * 0.9)

                    if origToEdgeY < center[1]:
                        yMargin = round(origToEdgeY * 0.9)
                    else:
                        yMargin = round(center[1] * 0.9)

                    self.croppedArr = galaxy[center[1] - yMargin:center[1] + (yMargin + 1),
                                 center[0] - xMargin:center[0] + (xMargin + 1)]

                    croppedErr = err[center[1] - yMargin:center[1] + (yMargin + 1),
                                 center[0] - xMargin:center[0] + (xMargin + 1)]

                    originXY = np.where(self.croppedArr == self.croppedArr.max())
                    center = (originXY[1][0], originXY[0][0])
                    (yMax, xMax) = self.croppedArr.shape

                    if yMax/xMax > 2 or xMax/yMax > 2:
                        # print(f"Weird dimensions: (ID-{self.ID}, {yMax}, {xMax})")
                        continue
                    else:
                        self.dimensions = (yMax, xMax, center)
                        self.galWithErr = np.dstack((self.croppedArr, croppedErr))

                        return True
            else:
                # print(f'Out of bounds: ID-{self.ID}')
                continue

    def asymmetry(self, abso=True, sqrd=False, multi=False):
        galaxy = self.galWithErr[:, :, 0]
        err = self.galWithErr[:, :, 1]
        yMax, xMax, center = self.dimensions

        if not multi:
            gal180 = scipy.ndimage.rotate(galaxy, 180)
            err180 = scipy.ndimage.rotate(err, 180)
            if abso and sqrd:
                return self.squaredA(galaxy, gal180, err, err180), self.absoluteA(galaxy, gal180, err, err180)
            elif abso:
                return self.absoluteA(galaxy, gal180, err, err180)
            else:
                return self.squaredA(galaxy, gal180, err, err180)
        else:
            r = 1
            centerList = [center, (center[0] + r, center[1]), (center[0] - r, center[1]),
                          (center[0], center[1] + r), (center[0], center[1] - r),
                          (center[0] + r, center[1] + r), (center[0] - r, center[1] + r),
                          (center[0] - r, center[1] - r), (center[0] + r, center[1] - r)]

            AList = []
            errList = []
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

                croppedArr = galaxy[point[0] - xMargin:point[0] + (xMargin + 1),
                             point[1] - yMargin:point[1] + (yMargin + 1)]

                croppedErr = err[point[0] - xMargin:point[0] + (xMargin + 1),
                             point[1] - yMargin:point[1] + (yMargin + 1)]

                arr180 = scipy.ndimage.rotate(croppedArr, 180)
                err180 = scipy.ndimage.rotate(croppedArr, 180)

                # plt.imshow(abs(croppedArr-arr180))
                # plt.show()

                if abso:
                    A, errA = self.absoluteA(croppedArr, arr180, croppedErr, err180)
                    # A /= np.sqrt(xMax * yMax)
                    AList.append(A)
                    errList.append(errA)

                elif sqrd:
                    sqrdA, errSqrdA = self.squaredA(croppedArr, arr180, croppedErr, err180)
                    # sqrdA /= np.sqrt(xMax * yMax)
                    AList.append(sqrdA)
                    errList.append(errSqrdA)
            AList.append(self.ID)
            errList.append(self.ID)
            return AList, errList

    def absoluteA(self, croppedArr, arr180, croppedErr, err180):
        # print('err:', croppedErr)
        numerator = np.sum(abs(croppedArr - arr180))
        errNum = utils.errorSum(np.sqrt(croppedErr ** 2 + err180 ** 2))

        denominator = 2 * np.sum(abs(croppedArr))
        errDenom = 2 * utils.errorSum(croppedErr)

        asymmetry = numerator / denominator
        totalErr = np.sqrt((errNum / numerator) ** 2 + (errDenom / denominator) ** 2) * asymmetry

        return (asymmetry, totalErr)

    def squaredA(self, croppedArr, arr180, croppedErr, err180):
        # print(croppedErr)
        numerator = np.sum((croppedArr - arr180) ** 2)
        sub = croppedArr - arr180
        subErr = np.sqrt(croppedErr ** 2 + err180 ** 2)
        squareErr = sub ** 2 * np.sqrt(2 * (subErr / sub) ** 2)
        errNum = utils.errorSum(squareErr)

        denominator = 2 * np.sum(croppedArr ** 2)
        errDenom = 2 * utils.errorSum(np.sqrt(2 * (croppedErr / croppedArr) ** 2) * croppedArr ** 2)

        asymmetry = numerator / denominator
        totalErr = np.sqrt((errNum / numerator) ** 2 + (errDenom / denominator) ** 2) * asymmetry

        return (asymmetry, totalErr)

    def showImage(self):
        plt.imshow(self.croppedArr)

    def getCoords(self):
        return self.RA, self.DEC

    def getRedshift(self):
        return self.z

    def getID(self):
        return self.ID

    def getName(self):
        return self.obName
