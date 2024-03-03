import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def getDataFITS(path=None):
    """
    Determine surface brights at points radius, r, outward using photoutils
    ---
    Input:  path, string; path to the fits file
    Output: data_cropped, numpy.array; intensity values for cropped area of fits file
            center_cropped, numpy.array; center of galaxy via max value
    """
    with open(path, 'r') as f:
        s_path = f.read()
        with fits.open(s_path) as hdul:
            hdr = hdul[0].header
            data = hdul['SCI'].data
            err = hdul['ERR'].data

    return data

def plotHistogram(dat=None):
    """
    Plot the values of the data in a histogram
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
    Output: None
    """
    histCount, edge, tmp = plt.hist(dat.flatten(), bins=100)
    plt.title("Intensity Histogram")
    plt.ylabel("Count")
    plt.xlabel("Intensity, I [MJy/sr]")
    plt.show()

    return None

if __name__ == "__main__":
    path = "downloads/downloads.txt"
    data = getDataFITS(path)

    # Plotting raw data
    print("Plotting raw data...")
    plt.imshow(data, origin="lower", cmap='magma', vmax=4)
    plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    plt.title("Raw Data")
    plt.show()

    # Remove Background
    print("Removing Background...")
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    # Set detection threshold
    sigma = 1.5
    print("Setting threshold (simga=" + str(sigma) + ")...")
    threshold = sigma * bkg.background_rms

    # Convolve the data with a 2D Gaussian kernel
    FWHM = 3.0
    print("Combing with kernal (FWHM=" + str(FWHM) + ")...")
    kernel = make_2dgaussian_kernel(FWHM, size=5)
    convolved_data = convolve(data, kernel)

    # Plotting convolved data
    print("Plotting convolved data...")
    plt.imshow(convolved_data, origin="lower", cmap='magma', vmax=4)
    plt.title("Convolved Data")
    plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    plt.show()

    # Detect the sources in the background-subtracted convolved image
        # segment_map is a SegmentationImage object
    print("Detecting sources ...")
    segment_map = detect_sources(convolved_data, threshold, npixels=10) # connectivity=8, aka. sides+corners
    print("Segment Map Info: \n", segment_map)

    # Plot background-subtracted image and the segmentation image (detected sources)
        # Seems to be detected the edges of the images
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', vmax=0.5)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')
    plt.show()




