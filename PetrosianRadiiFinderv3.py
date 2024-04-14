"""
Created by Mekhi D. Woods
04/03/2024
Current Version: 3.0

"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simpson

from astroquery.mast import Observations

import astropy.units as u
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.convolution import convolve
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
# from gwcs import WCS

from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceCatalog
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.background import Background2D, MedianBackground

from astropy.utils.data import get_pkg_data_filename
from astropy.wcs.utils import skycoord_to_pixel


# from astropy.visualization import SqrtStretch
# from astropy.visualization.mpl_normalize import ImageNormalize

SYS_TIME = str(int(time.time())) # System Time, for purposes of naming files uniquely
PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
DISPLAY_MODE = False
fn = get_pkg_data_filename('data/j94f05bgq_flt.fits', package='astropy.wcs.tests')


class petrosianObject():
    def __init__(self, ID='None', pos=(0, 0), SB=[], SBerr=[], iso_radii=[], iso_eps=[], isolist=None, aper=None, petroR=0.00):
        self.ID = ID
        self.pos = pos
        self.SB = SB
        self.SBerr = SBerr
        self.iso_radii = iso_radii
        self.iso_eps = iso_eps
        self.isolist = isolist
        self.aper = aper
        self.petroR = petroR
        return

    def __str__(self):
        return("Petrosian object, " + str(self.ID) + " | Center Position: " + str(self.pos) + ", " +
                                                     "Petrosian Radius [pix]: " + str(self.petroR) + ", " +
                                                     "Petrosian Radius [arcsec]: " + str(self.petroR*PIX_SCALE))
        # return(str(self.ID) + str(self.pos) + str(self.iso_radii) + str(self.SB) +
        #        str(self.SBerr) + str(self.iso_eps) + str(self.aper) + str(self.petroR))

def queryMAST(file='targets.txt'):
    # Construct targets list
    print("Constructing targets list...")
    targets_txt = np.genfromtxt(fname="targets.txt", delimiter="\t", dtype="float")
    targets = SkyCoord(ra=(targets_txt[:, 0]) * u.degree, dec=(targets_txt[:, 1]) * u.degree)
    print("Targets list constructed.")

    targetList = []
    for i in range(len(targets)):
        targetList.append(queryMAST_download(id=(queryMAST_pull(ra=targets[i].ra.degree, dec=targets[i].dec.degree))))
    return targetList

def queryMAST_pull(filter='F115W', dec=0, ra=0, sensitivity=0.01):
    """
    Query MAST database for JWST targets and obtain ids
    ---
    filter: string, desired JWST filter
    dec: float64, target declination in degrees
    ra: float64, target right ascension in degrees
    sensitivity: float64, target sensitivity
    ---
    return: astropy.table, object ID of found target(s)
    """
    print("Querying MAST database with parameters: DEC=", dec, ", RA=", ra, ", Sensitivity=", sensitivity)
    decRange = [dec - sensitivity, dec + sensitivity]
    raRange = [ra - sensitivity, ra + sensitivity]
    data = Observations.query_criteria(filters='F115W',  # Query MAST database for JWST targets
                                       s_dec=decRange,
                                       s_ra=raRange,
                                       intentType="science",
                                       instrument_name='NIRCAM/IMAGE',
                                       dataproduct_type='IMAGE',
                                       obs_collection='JWST',
                                       dataRights="public",
                                       calib_level=3)
    print("Finished. Target ID: ", data['obsid'].data[0])
    return data['obsid']

def queryMAST_count(filter='F115W', dec=0, ra=0, sensitivity=0.01):
    """
    Query MAST database for JWST targets
    ---
    filter: string, desired JWST filter
    dec: float64, target declination in degrees
    ra: float64, target right ascension in degrees
    sensitivity: float64, target sensitivity
    ---
    hits: int, number of found JWST targets within criteria
    """
    print("Counting hits from MAST database with parameters: DEC=", dec, ", RA=", ra, ", Sensitivity=", sensitivity)
    decRange = [dec - sensitivity, dec + sensitivity]
    raRange = [ra - sensitivity, ra + sensitivity]
    hits = Observations.query_criteria_count(filters=filter,
                                             s_dec=decRange,
                                             s_ra=raRange,
                                             intentType="science",
                                             instrument_name='NIRCAM/IMAGE',
                                             dataproduct_type='IMAGE',
                                             obs_collection='JWST',
                                             dataRights="public",
                                             calib_level=3)
    print("Finished. Number of hits: ", len(hits))
    return hits

def queryMAST_download(id=None, download_path="downloads\\"):
    """
    Downloads data associated with some astroquery id
    ---
    id: astropy.table, id of target to download
    download_path: string, default download path
    ---
    return: None
    """
    print("Beginning download of data asscociated with ID:", str(id.data[0]))
    # data_products = Observations.get_product_list(id) # ALL data products associated with ID

    location = None
    if os.path.isdir(download_path + str(id.data[0])):
        print("Data already downloaded, skipping...\n")
    else:
        manifest = Observations.download_products(id,
                                                  download_dir=download_path + str(id.data[0]),
                                                  productType=["SCIENCE"],
                                                  extension="fits",
                                                  mrp_only=True)
        print("Finished! ", str(id.data[0]), "can be found at...", manifest['Local Path'][0], "\n")
        # Store the Save directory
        with open(download_path+"downloads.txt", "a") as f:
            f.write(manifest['Local Path'][0])
        location = manifest['Local Path'][0]
    return location

def get_data_fits(path=None, bin=0):
    """
    Read in data from fits file
    ---
    Input:  path, string; path to the fits file
            bin, int; the bin to pull data from
    Output: data_cropped, numpy.array; intensity values for cropped area of fits file
            hdr, astropy.io.fits.header.Header; header of the fits file
    """
    with fits.open(path) as hdul:
        hdr = hdul[bin].header
        data = hdul[bin].data

    return data, hdr

def quick_plot(data=None, title="Default" , cmap='magma', interpolation='antialiased', show=True):
    z1, z2 = ZScaleInterval().get_limits(values=data)
    if cmap=='magma':
        plt.imshow(data, origin="lower", cmap=cmap, interpolation=interpolation, vmin=z1, vmax=z2)
    else:
        plt.imshow(data, origin="lower", cmap=cmap, interpolation=interpolation)
    plt.title(title)
    if show:
        plt.show()
    return

def image_segmintation(data, threshold=0.5):
    print("Plotting raw data...")
    quick_plot(data=data, title="Raw data")

    print("Convolving data with a 2D kernal...")
    convolved_data, kernel = image_seg_convolve(data, FWHM=3.0, size=5, display=False)

    print("Detecting sources in convolved data...")
    segment_map = image_seg_detect_sources(convolved_data, threshold=threshold, npixels=10, display=False)

    print("Deblend overlapping sources...")
    segm_deblend = image_seg_deblend(convolved_data, segment_map, display=False)

    print("Catalog sources...")
    cat, sources_x, sources_y, sources_eps, apers = image_seg_cat(data, segm_deblend, convolved_data, display=False)

    print("Setting Kron apertures...")
    image_seg_kron_apertures(convolved_data, cat, display=False)



    return sources_x, sources_y, sources_eps, apers

def image_seg_convolve(data, FWHM=3.0, size=5, display=False):
    """
    Convolving data with a 2D kernal
    I'm thinking that this step is to remove flatfeild from the image, but I think that's
    already been done to these images
    """
    kernel = make_2dgaussian_kernel(FWHM, size=size)
    convolved_data = convolve(data, kernel)
    if display:
        quick_plot(data=kernel, title="Kernal")
    return convolved_data, kernel

def image_seg_detect_sources(convolved_data, threshold=20, npixels=10, display=False):
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    if display:
        print(segment_map)
    return segment_map

def image_seg_deblend(convolved_data, segment_map, display=False):
    # segm_deblend = deblend_sources(convolved_data, segment_map,
    #                                npixels=10, nlevels=32, contrast=0.001,
    #                                progress_bar=True)
    # if display:
    #     plt.imshow(segm_deblend, origin='lower', cmap=segment_map.cmap,
    #                interpolation='nearest')
    #     plt.xlabel("[pixels]"); plt.ylabel("[pixels]")
    #     plt.title("Deblended Segmentation Image")
    #     plt.show()
    segm_deblend = segment_map
    quick_plot(segment_map, title="Segmentation Image", cmap=segment_map.cmap, interpolation='nearest')
    return segm_deblend

def image_seg_cat(data, segm_deblend, convolved_data, display=False):
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)

    apers = cat.make_kron_apertures()
    # print(apers[0].positions)

    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'

    sources_x = tbl['xcentroid']
    sources_y = tbl['ycentroid']
    sources_eps = tbl['eccentricity']

    if display:
        print('\n')
        print(cat)
        print(tbl)
        print(sources_x, sources_y, sources_eps)

    return cat, sources_x, sources_y, sources_eps, apers

def image_seg_kron_apertures(data, cat, display=False):
    z1, z2 = ZScaleInterval().get_limits(values=data)
    norm = simple_norm(data, 'sqrt')
    plt.imshow(data, origin='lower', cmap='magma', vmin=z1, vmax=z2)
    plt.title('kron apertures')
    cat.plot_kron_apertures(color='green', lw=1.5)
    plt.show()
    return

def isophote_fit_image_aper(dat, aper, eps=0.01, nIsos=30, display=False):
    """
    Determine surface brights at points radius, r, outward using photoutils
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
            cen0, numpy.array; center point of galaxy
            rMax, int; max distance [pixels] from center that the isophotes will be calculated to
            nRings, int; number of isophotes to display later
    Output: isolist.sma, numpy.array; list of radii/sma values for isophotes
            isolist.intens, numpy.array; list of intensity/surface brightness values for isophotes
            isolist.int_err, numpy.array; 'The error of the mean intensity (rms / sqrt(# data points)).'
            isos, list; list of reconstructed ʻringsʻ/isophotes at some interval, r_max/n_rings
            cen_new, final calculated center of isophote calculating
    Notes:  Algorithum used is from Jedrzejewski (1987; MNRAS 226, 747)
            https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract
    """
    z1, z2 = ZScaleInterval().get_limits(values=dat)  # MJy/sr

    cen = [aper.positions[0], aper.positions[1]] # Grab updated centers

    # plt.imshow(dat, origin='lower', vmin=z1, vmax=z2) # Plot ALL data from fits, bounded
    # aper.plot(color='r')

    g = EllipseGeometry(x0=cen[0], y0=cen[1], sma=aper.a, eps=eps, pa=(aper.theta / 180.0) * np.pi)
    ellipse = Ellipse(dat, geometry=g)
    isolist = ellipse.fit_image(maxit=nIsos, maxsma=(aper.a*1.5)) # Creates isophotes using the geometry of 'g', so using above parameters as the bounds
    # print("Number of isophotes: ", len(isolist.to_table()['sma']))

    # # Plots the isophotes over some interval -- this part is PURELY cosmetic, it doesn't do anything
    # isos = [] # A list of isophote x-y positions to plot later
    # if nRings == -1:                            # nRings=-1 plots all the rings
    #     nRings = len(isolist.to_table()['sma'])
    # if nRings != 0 and len(isolist.to_table()['sma']) > 0: # Makes sure that there is data from the isophote fit
    #     rMax = isolist.to_table()['sma'][-1]  # Largest radius
    #     rings = np.arange(0, rMax, rMax / nRings)
    #     rings += rMax / nRings
    #     for sma in rings:
    #         iso = isolist.get_closest(sma) # Displayed isophotes are just the closest isophotes to a certain desired sma, but
    #                                        # there are more isophotes between the ones displayed.
    #         isos.append(iso.sampled_coordinates())
    #         plt.plot(iso.sampled_coordinates()[0], iso.sampled_coordinates()[1], color='g', linewidth=1)
    # if display:
    #     plt.show()

    return isolist.sma, isolist.intens, isolist.int_err, isolist.to_table()['ellipticity'], isolist

def data_visualizer(dat=None, cen0=np.array([0, 0]), cenFinal=np.array([0, 0]), rings=None, units=False, save=False):
    """
    Uses matplotlib.pyplot.imshow() to display the data.
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
            cen0, numpy.array; initial guess for the center point of galaxy
            cenFinal, numpy.array; final center point determine via function
                                   'photutils.isophote.Ellipse.ellipse.fit_image()'
            rings, numpy.array; x-y positions of isophotes
            units, bool; True=arcseconds, False=pixels
            save, bool; save visualization or not
    Output: None
    """
    # Calculate zscale data range | interval based on IRAF’s zscale
    z1, z2 = ZScaleInterval().get_limits(values=dat) # MJy/sr

    if units:
        # Fix Axies to arcseonds
        axis_scale = np.shape(data)
        axis_scale = np.array([0, axis_scale[1] * PIX_SCALE, 0, axis_scale[0] * PIX_SCALE])

        # Arcseconds
        plt.imshow(dat, vmin=z1, vmax=z2, origin='lower', cmap='magma', extent=axis_scale)

        plt.plot(cen0[0] * PIX_SCALE, cen0[1] * PIX_SCALE, 'r+',
                 label='Initial Center ' + str(cen0[0] * PIX_SCALE) + ", " + str(cen0[1] * PIX_SCALE))
        plt.plot(cenFinal[0] * PIX_SCALE, cenFinal[1] * PIX_SCALE, 'b+',
                 label='Final Center ' + str(round(cenFinal[0] * PIX_SCALE, 3)) + ", " + str(round(cenFinal[1] * PIX_SCALE, 3)))

        plt.title(SYS_TIME + '\n' + "Range: " + str(round((z1 * PIX_SCALE), 3)) + "-" + str(round((z2 * PIX_SCALE), 3)) + " MJy/sr" + " [arcsec]")
        plt.xlabel("[arcsec]"); plt.ylabel("[arcsec]")
    else:
        # Pixels
        plt.imshow(dat, vmin=z1, vmax=z2, origin='lower', cmap='magma')

        plt.plot(cen0[0], cen0[1], 'r+',
                 label='Initial Center: ' + str(cen0[0]) + ", " + str(cen0[1]))
        plt.plot(cenFinal[0], cenFinal[1], 'b+',
                 label='Final Center: ' + str(cenFinal[0]) + ", " + str(cenFinal[1]))

        plt.title(SYS_TIME + '\n' + "Range: " + str(round(z1, 3)) + "-" + str(round(z2, 3)) + " MJy/sr" + " [pixels]")
        plt.xlabel("[pixels]"); plt.ylabel("[pixels]")

    # Plot isophotes
        # The plot isn't of the actual isophote still, instead it finds a close neighbor and plots that
    if rings != None:
        for iso in rings:
            if units:
                plt.plot(iso[0] * PIX_SCALE, iso[1] * PIX_SCALE, color='g', linewidth=0.5)
            else:
                plt.plot(iso[0], iso[1], color='g', linewidth=0.5)

    # Save / Display
    plt.legend()
    if save:
        plt.savefig(r"results\GalaxyVisual_" + SYS_TIME + ".png")
    plt.show()

    return None

def plot_sb_profile(ID='', r=None, SB=None, err=None, sigma=10, r_forth=False, units=False, save=False):
    """
    Plot the surface brightness profile
    ---
    Input:  r, numpy.array; radius/semimajor axis of galaxy
            SB, numpy.array; surface brightness for each point of radius
            err, numpy.array; error of surface brightness calculation / meassurement error
            sigma, int; scales the error bars
            r_forth, bool; swaps mode to plot r to the 1/4
            units, bool; True=arcseconds, False=pixels
            save, bool; save plot or not
    Output: None
    """
    unit = '[pixels]'
    marker_size = 2
    if units:
        r = r * PIX_SCALE
        unit = '[arcsec]'

    # Plot SB vs radius [arcsec]
    plt.errorbar(r, SB, yerr=(err) * sigma, fmt='o', ms=marker_size)
    plt.xlabel("Radius, r " + unit)
    plt.title(SYS_TIME + " [" + str(ID) + "]" + '\nSurface Brightness Profile, ' + unit)
    plt.ylabel("Intensity, I [MJy/sr]")
    if save:
        plt.savefig(r"results\SBprofile_" + str(int(SYS_TIME)) + ".png")
    plt.show()

    return None

def plot_isophote_rings(isolist=None, nRings=10, c='g', display=True):
    # Plots the isophotes over some interval -- this part is PURELY cosmetic, it doesn't do anything
    isos = [] # A list of isophote x-y positions to plot later
    if nRings == -1:                            # nRings=-1 plots all the rings
        nRings = len(isolist.to_table()['sma'])
    if nRings != 0 and len(isolist.to_table()['sma']) > 0: # Makes sure that there is data from the isophote fit
        rMax = isolist.to_table()['sma'][-1]  # Largest radius
        rings = np.arange(0, rMax, rMax / nRings)
        rings += rMax / nRings
        for sma in rings:
            iso = isolist.get_closest(sma) # Displayed isophotes are just the closest isophotes to a certain desired sma, but
                                           # there are more isophotes between the ones displayed.
            isos.append(iso.sampled_coordinates())
            plt.plot(iso.sampled_coordinates()[0], iso.sampled_coordinates()[1], color=c, linewidth=1)
    if display:
        plt.show()
    return

def petrosian_radius(radius=None, SB=None):
    adda = 0.2  # Target constant
    sens = 0.005
    petro_r = -0.1 # Default Value, means no petrosian found

    for i in range(2, len(radius)):
        localSB = SB[i]
        integratedSB = petro_r_avg_SB(r=radius[:i], SB=SB[:i])

        if abs(integratedSB - (adda*localSB)) < sens:
            print(integratedSB, adda*localSB)
            petro_r = radius[i]
            break
    return petro_r

def petro_r_avg_SB(r=None, SB=None):

    SBtoR = simpson(y=SB, x=r) * 2 * np.pi
    area = np.pi*r[-1]**2

    avgSBwithinR = SBtoR/area

    return avgSBwithinR

if __name__ == "__main__":
    mainPath = r'downloads\jw01181-o098_t010_nircam_clear-f115w_i2d.fits'
    altPath=r'downloads/jw01181-c1009_t008_nircam_clear-f090w_i2d.fits'
    targetPath = r'targets.csv'
    realUnits = False
    bin = 'SCI'
    petroObjs = []
    crop = [100, 100]

    # OPEN FITS FILE
    print("Obtaining data from FITS...")
    with fits.open(altPath) as hdul:
        hdu = hdul[bin]
        data = hdu.data
        hdr = hdu.header
        datacoords = WCS(hdr)

    # OPEN TARGET LIST
    targets = np.genfromtxt(targetPath, delimiter=',', dtype=str)
    targets = targets[:, :5] # Clean columns

    for t in range(1, len(targets)):
        print(targets[t])
        print(targets[t][1])

        # tar = SkyCoord(float(targets[t][1]), float(targets[t][2]), unit="deg")
        tar = np.array([targets[t][1], targets[t][2]], dtype=np.float64)
        tarpix = datacoords.wcs_pix2world(tar, 0)
        print(tar)
        # print(datacoords.skycoord_to_pixel(tar))

        break

    # print(hdr)
    # quick_plot(data)
    # print(coords.pixel_to_world(30, 40))
















    # OBTAIN
    # print(queryMAST(targetPath))

    # print("Obtaining data from FITS...")
    # data, header = get_data_fits(path=mainPath, bin=bin)

    # print(data[3543, 1243])

    # data = data[0:3950, 0:2000] # Just temporarily manually cropping
    # data = data[3400:3950, 1220:2000] # Just temporarily manually cropping





    #
    # print("Segmenting image...")
    # sources_x, sources_y, sources_eps, apers = image_segmintation(data, threshold=0.9)
    #
    # '''
    # BREAK DOWN
    # '''
    # for i in range(len(apers)):
    #     print("[", i, "] Fiting isophotes...")
    #     tempObj = petrosianObject(ID=i, pos=(sources_x[i], sources_y[i]), iso_eps=float(sources_eps[i]), aper=apers[i])
    #     tempObj.iso_radii, tempObj.SB, tempObj.SBerr, tempObj.iso_eps, tempObj.isolist = isophote_fit_image_aper(data,
    #                                                                                                              aper=tempObj.aper,
    #                                                                                                              eps=tempObj.iso_eps,
    #                                                                                                              nIsos=10,
    #                                                                                                              display=False)
    #
    #     if len(tempObj.iso_radii) > 0:
    #         print("[", i, "] Calculating petrosian radii...")
    #         petro_r = petrosian_radius(radius=tempObj.iso_radii, SB=tempObj.SB)
    #         tempObj.petroR = petro_r
    #     else:
    #         print("[", i, "] No meaningful fit was possible.")
    #         tempObj.petroR = None
    #
    #     petroObjs.append(tempObj)
    #
    #
    # '''
    # DISPLAY
    # '''
    #
    # for i in range(len(petroObjs)):
    #     quick_plot(data, title="Isophotes [" + str(i) + "]", show=False)
    #     if len(petroObjs[i].iso_radii) > 0:
    #         print("[", i, "] Plotting isophote rings...")
    #         plot_isophote_rings(isolist=petroObjs[i].isolist, nRings=-1, c='g', display=False)
    #         plt.show()
    #
    #         print("[", i, "] Plotting surface brightness profile...")
    #         plot_sb_profile(ID=i, r=petroObjs[i].iso_radii, SB=petroObjs[i].SB, err=petroObjs[i].SBerr, sigma=10, units=realUnits, save=False)
    #
    #         print(petroObjs[i])
    #     time.sleep(0.5) # Pause between requests
    # #
    # # """
    # # Sanity Check:
    # # print(petro_r_avg_SB(a=radius[:50], eps=isoeps[:50], SB=SB[:50]))
    # # Averages <~10 are >>1 since its right at that bright center, but they quickly drop
    # # I looked at DS9 and saw the same
    # # DS9 avg @ 111: 0.4285, Mine: 0.0017414880080206133
    # # Weird, its report a waaaaaay too low value
    # # I found the issue, the integral isn't calculating right. I'll fix later.
    # # """
    # #
    # #
