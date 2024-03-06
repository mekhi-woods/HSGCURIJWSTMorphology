from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from PIL import Image as im, ImageDraw

problemChildren = [11, 13, 30, 31, 32, 35, 40]
coordsList = []


f = open('files/SMACSTargets2.txt', 'r')
count = 0
for line in f:
    if count <= 2:
        count += 1
        continue

    (ID, RA, dec, zGrism, zErr, zSpec) = line.split(' ')
    if zGrism == '' or zGrism == '""':
        continue
    else:
        if 0.6 <= float(zGrism) <= 1.4:
            coords = (RA, dec, ID)
            coordsList.append(coords)
    count += 1

index = 0
for coord in coordsList:
    (RA, dec, ID) = coord

    hdfra = float(RA) * u.deg
    hdfdec = float(dec) * u.deg

    infits = "mastDownload/JWST/jw02736-o001_t001_nircam_clear-f444w/jw02736-o001_t001_nircam_clear-f444w_i2d.fits"

    infile = infits.removesuffix(".fits")+".fits"

    # Open the image - the data and the header separately
    fitsimage = fits.open(infile)
    ncards = len(fitsimage)
    headerword = 'CRVAL1'

    phu = 0
    sciii = phu
    badc = 0
    for ii in range(0, ncards):
        # print(fitsimage[ii].header)
        headname = fitsimage[ii].name
        try:
            valhead = fitsimage[ii].header[headerword]
            sciii = ii
            break
        except:
            badc += 1
            valhead = "INDEF"

    if (badc == ncards):
        print("=" * 50)
        print(f"ERROR: Header keyword {headerword} not found in any extension!")
        print("      Available keywords could be targ_ra, etc.")
        print("      Refer to instrument/observatory handbook - Good luck!")

    image = fitsimage[sciii].data
    headersci = fitsimage[sciii].header
    # print(repr(headersci))
    headerphu = fitsimage[phu].header
    print(f"target RA:{headerphu['TARG_RA']}")
    print(f"target dec:{headerphu['TARG_DEC']}")

    # Now grab the necessary info from the header (usually [SCI] or [1] extension) that is relevant
    # to the world-coordinate-system wcs
    wcs = WCS(headersci)

    xHDF, yHDF = wcs.all_world2pix([[hdfra/u.deg, hdfdec/u.deg]], 0)[0]
    print(xHDF, yHDF)

    xRange = 30
    yRange = 30

    xLower = round(xHDF - xRange)
    xUpper = round(xHDF + xRange)
    yLower = round(yHDF - yRange)
    yUpper = round(yHDF + yRange)

    with fits.open(infits) as hdul:
        data = hdul[1].data #* 200
        err = hdul['ERR'].data

    image = im.fromarray(data)
    # image.show()
    # bbox = (round(xHDF-30), round(yHDF-30), round(xHDF+30), round(yHDF+30))
    # draw = ImageDraw.Draw(image)
    # draw.ellipse(bbox, outline='white')
    # del draw

    data = np.array(image)

    galaxy = data[yLower:yUpper, xLower:xUpper]# * 200
    err = data[yLower:yUpper, xLower:xUpper]
    galWithErr = np.dstack((galaxy, err))
    # print(galaxy.shape)
    # print(galWithErr.shape)
    # print(galaxy)

    croppedImage = im.fromarray(galaxy)
    # image = im.fromarray(galaxy)
    # x2, y2 = image.size
    # bbox = (x2 - (x2 / 2) - 10, y2 - (y2 / 2) - 10, x2 - (x2 / 2) + 10, y2 - (y2 / 2) + 10)
    # draw = ImageDraw.Draw(image)
    # draw.ellipse(bbox, fill=50)
    # del draw

    # image.show()
    # croppedImage.show()
    if len(ID) < 4:
        ID = '00' + ID
    elif len(ID) < 5:
        ID = '0' + ID

    if index in problemChildren:
        np.save(f'files/SMACSGalPics/problemChildren/{ID}galaxy{index}.npy', galaxy)
        plt.imsave(f'files/SMACSGalPics/problemChildren/{ID}galaxy{index}.png', galaxy, cmap='gray', vmin=0.2, vmax=0.6)
    else:
        np.save(f'files/SMACSGalPics/imsAsArrays/{ID}galaxy{index}.npy', galaxy)
        plt.imsave(f'files/SMACSGalPics/images/{ID}galaxy{index}.png', galaxy, cmap='gray', vmin=0.2, vmax=0.6)

    index += 1
