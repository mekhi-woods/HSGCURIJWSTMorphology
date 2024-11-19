import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceCatalog
from astropy.visualization import simple_norm

# from photutils.background import Background2D, MedianBackground
# from astropy.convolution import convolve
# from photutils.segmentation import make_2dgaussian_kernel
# from photutils.segmentation import detect_sources
# from astropy.visualization import SqrtStretch
# from astropy.visualization.mpl_normalize import ImageNormalize

def get_data_fits(path=None, bin=0):
    """
    Determine surface brights at points radius, r, outward using photoutils
    ---
    Input:  path, string; path to the FITS file
    Output: data, numpy.ndarray; intensity values for cropped area of FITS file
            hdr, astropy.io.fits.header.Header; header of the FITS file
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[bin].data

    return data, hdr

def plot_data(data=None, title="Default", label="[pixels]"):
    z1, z2 = ZScaleInterval().get_limits(values=data)
    plt.imshow(data, origin="lower", cmap='magma', vmin=z1, vmax=z2)
    plt.xlabel(label); plt.ylabel(label)
    plt.title(title)
    plt.show()
    return

def full_process(data=None):
    # Plotting raw data
    print("Plotting raw data...")
    plot_data(data=data, title="Raw data", label="[pixels]")

    # Remove Background
    ## This step may be unnessasary if the FITS are already BG subtracted
    # print("Removing background...")
    # bkg_estimator = MedianBackground()
    # bkg = Background2D(data, (50, 50), filter_size=(3, 3),
    #                    bkg_estimator=bkg_estimator)
    # data -= bkg.background  # subtract the background
    # plot_data(data=data, title="Background Subtracted", label="[pixels]")
    #
    # # Setting threshold for ###
    # threshold = 20 * bkg.background_rms
    threshold = 0.5



    # Convolving data with a 2D kernal
    ## I'm thinking that this step is to remove flatfeild from the image, but I think that's
    ## already been done to these images
    print("Convolving data with a 2D kernal...")
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    #plot_data(data=kernel, title="Kernal", label="[?]")

    # Detecting sources
    print("Detecting sources in convolved data...")
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    print(segment_map)

    # Plot the Segment Map
    plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    plt.title("Segmentation Image")
    plt.show()

    # # Deblend overlapping sources
    # print("Deblend overlapping sources...")
    # segm_deblend = deblend_sources(convolved_data, segment_map,
    #                                npixels=10, nlevels=32, contrast=0.001,
    #                                progress_bar=True)
    # plt.imshow(segm_deblend, origin='lower', cmap=segment_map.cmap,
    #            interpolation='nearest')
    # plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    # plt.title("Deblended Segmentation Image")
    # plt.show()


    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    print(cat)

    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'
    print(tbl)

    sources_x = tbl['xcentroid']
    sources_y = tbl['ycentroid']
    sources_eps = tbl['eccentricity']

    print(sources_x, sources_y, sources_eps)

    norm = simple_norm(data, 'sqrt')
    z1, z2 = ZScaleInterval().get_limits(values=data)
    plt.imshow(data, origin='lower', cmap='magma', vmin=z1, vmax=z2)
    plt.title('kron apertures')
    cat.plot_kron_apertures(color='white', lw=1.5)
    plt.show()



    return None

def process_sans_bkg_reduction(data=None):
    # Plotting raw data
    print("Plotting raw data...")
    plot_data(data=data, title="Raw data", label="[pixels]")

    # Setting threshold for ###
    threshold = 0.4

    # Convolving data with a 2D kernal
    ## I'm thinking that this step is to remove flatfeild from the image, but I think that's
    ## already been done to these images
    print("Convolving data with a 2D kernal...")
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    # plot_data(data=kernel, title="Kernal", label="[?]")

    # Detecting sources
    print("Detecting sources in convolved data...")
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    print(segment_map)

    # Plot the Segment Map
    plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    plt.xlabel("[pixels]");
    plt.ylabel("[pixels]")
    plt.title("Segmentation Image")
    plt.show()

    # # Deblend overlapping sources
    # print("Deblend overlapping sources...")
    # segm_deblend = deblend_sources(convolved_data, segment_map,
    #                                npixels=10, nlevels=32, contrast=0.001,
    #                                progress_bar=True)
    # plt.imshow(segm_deblend, origin='lower', cmap=segment_map.cmap,
    #            interpolation='nearest')
    # plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    # plt.title("Deblended Segmentation Image")
    # plt.show()

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    print(cat)

    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'
    print(tbl)

    sources_x = tbl['xcentroid']
    sources_y = tbl['ycentroid']
    sources_eps = tbl['eccentricity']

    print(sources_x, sources_y, sources_eps)

    norm = simple_norm(data, 'sqrt')
    z1, z2 = ZScaleInterval().get_limits(values=data)
    plt.imshow(data, origin='lower', cmap='magma', vmin=z1, vmax=z2)
    plt.title('kron apertures')
    cat.plot_kron_apertures(color='white', lw=1.5)
    plt.show()

    return None

if __name__ == "__main__":
    # path = "downloads/jw01181-o098_t010_nircam_clear-f115w_i2d.fits"
    # data, header = get_data_fits(path, bin=1)
    path = "downloads/jw01181-o098_t010_nircam_clear-f115w_i2d.fits"
    data, header = get_data_fits(path, bin='SCI')

    plot_data(data=data)

    full_process(data=data)

    # process_sans_bkg_reduction(data=data)



