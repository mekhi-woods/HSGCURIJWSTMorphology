import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import utilities as utils
from photutils.isophote import build_ellipse_model
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse
from photutils.aperture import EllipticalAperture

outputs = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ID_PARENT', 'X_WORLD', 'Y_WORLD',
           'THETA_IMAGE', 'ELLIPTICITY', 'A_IMAGE', 'B_IMAGE',
           'PETRO_RADIUS', ]

infits = 'jw02736-o001_t001_nircam_clear-f090w_i2d.fits'

f = open('default.param', 'wb')
for param in outputs:
    f.write(f'{param}\n'.encode('ascii'))
f.close()

# os.system(f'sex {fitsImage}')

data = np.loadtxt('catalog.txt')
thetaList = data[:, 18]
ellipList = data[:, 19]
majorList = data[:, 20]
minorList = data[:, 21]
fluxRadList = data[:, 9]
xList = data[:, 1]
yList = data[:, 2]
RAList = data[:, 16]
decList = data[:, 17]

indexList = []
f = open('/Users/gus/Desktop/PycharmProjects/HSGCURIJWSTMorphology-repo/files/SMACSTargets.txt', 'r')
for line in f:
    if 'S' not in line:
        ID, RA, dec, z, *_ = line.split(' ')
        for i in range(len(xList)-1):
            if abs(float(RA) - RAList[i]) <= 1e-4 and abs(float(dec) - decList[i]) <= 1e-4:
                if z != '""' and 0.6<=float(z)<=1.4:
                    print(RA, RAList[i], dec, decList[i])
                    indexList.append(i)

print(len(indexList))
infits = infits.removesuffix(".fits") + ".fits"
with fits.open(infits) as hdul:
    data = hdul[1].data

for i in indexList[5:6]:
    radius = 3*fluxRadList[i]
    galArr = data[round(yList[i] - radius):round(yList[i] + radius),
             round(xList[i] - radius):round(xList[i] + radius)]

    origin = np.where(galArr == galArr.max())
    originX = origin[1][0]
    originY = origin[0][0]

    geometry = EllipseGeometry(x0=radius, y0=radius, sma=majorList[i], eps=ellipList[i],
                               pa=thetaList[i] * np.pi / 180.0)

    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                              geometry.sma * (1 - geometry.eps),
                              geometry.pa)
    plt.imshow(galArr, origin='lower')
    aper.plot(color='white')
    plt.show()

    ellipse = Ellipse(galArr, geometry)

    isolist = ellipse.fit_image()
    # print(isolist.sma)

    model_image = build_ellipse_model(galArr.shape, isolist)

    # plt.imshow(galArr)
    # plt.show()

    plt.imshow(model_image)
    plt.show()
