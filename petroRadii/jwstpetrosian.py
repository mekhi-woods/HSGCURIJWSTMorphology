import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve

from photutils.aperture import CircularAperture
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog

SYS_TIME = str(int(time.time())) # System Time, for purposes of naming files uniquely
PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
SPEED_OF_LIGHT = 3e5 # km/s
H0 = 73.8 # km/s/Mpc
CMAP = 'viridis'

class PetrosianObject:
    """
    Container object for the properties of a given target
    -----------------------------------------------
    Member Variables:   ID, ID of the target taken from target file [str]
                        z, redshift of the target [float]
                        pos, pixel position of the target (numpy.float64, numpy.float64)
                        SB, intensity values as related to the radius [numpy.ndarray]
                        SBerr, error of intensity values as related to the radius [numpy.ndarray]
                        radius, distance from center of the target [numpy.ndarray]
                        petroR, Petrosian Radius of the target [float]
    Member Functions:   toKpc(), converts pixels distances to kpc using the redshift of the target
                        display_self(), displays data sheet of target
                            data, array of data from FITS file [numpy.ndarray]
                            nRingsDisp, number of rings to display [int]
                            crop, where to crop the data to zoom in on the target [int]
                            SBsigma, sigma parameter to make the errors appear better [int]
    """
    def __init__(self, SB=None, SBerr=None, radius=None, ID='None', z=0, pos=(0, 0), petroR=0.00):
        self.ID = ID
        self.z = z
        self.pos = pos
        self.SB = SB
        self.SBerr = SBerr
        self.radius = radius
        self.petroR = petroR
        return

    def __str__(self):
        return ("Petrosian object, " + str(self.ID) + " | Center Position: " + str(self.pos) + ", "
                + "Petrosian Radius: " + str(self.petroR) + ", " + "Redshift: " + str(self.z))

    def toKpc(self):
        return ((SPEED_OF_LIGHT*self.z) / H0) * self.petroR * PIX_SCALE * (np.pi/(180*3600)) * 1000

    # noinspection PyTypeChecker
    def display_self(self, data, nRingsDisp=10, crop=150, SBsigma=1):
        z1, z2 = ZScaleInterval().get_limits(values=data)

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('ID [' + str(self.ID) + '] ' +
                     'Center: (' + str(round(self.pos[0], 2)) + ', ' + str(round(self.pos[1], 2)) + ') | ' +
                     'Petrosian Radius: ' + str(self.petroR) + ' [pix] | ' +
                     'Redshift: ' + str(round(self.z, 2)))

        # Plot Raw Data
        ax1 = plt.subplot(223)
        ax1.imshow(data, origin="lower", cmap=CMAP, vmin=z1, vmax=z2) # BG
        ax1.set_xlim(self.pos[0] - crop, self.pos[0] + crop)
        ax1.set_ylim(self.pos[1] - crop, self.pos[1] + crop)
        ax1.set_xlabel('[pixels]')
        ax1.set_ylabel('[pixels]')

        # Plot Rings
        ax2 = plt.subplot(224)
        ax2.imshow(data, origin="lower", cmap=CMAP, vmin=z1, vmax=z2) # BG
        if nRingsDisp != 0 and nRingsDisp < len(self.radius):
            for i in range(len(self.radius)):
                if i % int(len(self.radius)/nRingsDisp) == 0:
                    aper = CircularAperture([self.pos[0], self.pos[1]], self.radius[i])
                    aper.plot(color='k', lw=1.5, alpha=0.5)

        aper = CircularAperture([self.pos[0], self.pos[1]], self.petroR) # Plots ring at petrosian
        aper.plot(color='r', lw=2, alpha=0.75, label='Petrosian Radius')

        ax2.set_xlim(self.pos[0] - crop, self.pos[0] + crop)
        ax2.set_ylim(self.pos[1] - crop, self.pos[1] + crop)
        ax2.set_xlabel('[pixels]')
        ax2.set_ylabel('[pixels]')
        # ax2.legend(loc='upper right')

        # Plot Surface Brightness
        ax3 = plt.subplot(211)
        ax3.axvline(x=self.petroR, color='r', label='Petrosian Radius = '+str(self.petroR)+' [pix]')
        ax3.errorbar(self.radius, self.SB, yerr=(self.SBerr * SBsigma))
        ax3.set_xlabel('Radius [pix]')
        ax3.set_ylabel('Intensity [MJy/sr]')
        ax3.legend(loc='upper right')

        plt.show()
        return

def FITS_info(path, wrkBin='SCI', errBin='ERR'):
    """
    Unpacks the FITS file
    -----------------------------------------------
    Input:  path, path to FITS file to unpack [str]
            wrkBin, FITS bin to use for unpacking data [str]
            errBin, FITS bin to use for unpacking errors [str]
    Output: hdr, header of FITS [astropy.io.fits.header.Header]
            data, data from FITS file [numpy.ndarray]
            err, error from FITS file [numpy.ndarray]
            datcrd, object that helps to find coords of certain points [astropy.wcs.wcs.WCS]
            z1, similar to DS9 z-scale; lower limit [numpy.float64]
            z2, similar to DS9 z-scale; upper limit [numpy.float64]
    """
    with fits.open(path) as hdul:
        hdu = hdul[wrkBin]
        data = hdu.data
        hdr = hdu.header
        err = hdul[errBin].data
        datcrd = WCS(hdr)
        z1, z2 = ZScaleInterval().get_limits(values=data)
    return hdr, data, err, datcrd, z1, z2

def world_to_pix(data, path, crds, z1, z2, show=False):
    """
    Converts the coordinates of certain points from world coordinates to pixels
    -----------------------------------------------
    Input:  path, path to file containing target list [str]
            crds, object that helps to find coords of certain points [astropy.wcs.wcs.WCS]
            z1, similar to DS9 z-scale; lower limit [numpy.float64]
            z2, similar to DS9 z-scale; upper limit [numpy.float64]
            show, display overlap plot? [bool]
    Output: allCoordPix, pixel coordinates of target points in pixels [list]
            targetIDs, target names (i.e. ID [158]) [numpy.ndarray]
            targetZs, target redshift [numpy.ndarray]
    """
    allCoordPix, targetIDs = [], []

    targets = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    targets = targets[:, :4]
    targetIDs = targets[:, 0]
    targetZs = targets[:, 3]

    if show:
        plt.imshow(data, origin="lower", cmap=CMAP, interpolation='antialiased', vmin=z1, vmax=z2)
        plt.title('Targets overlaid with data')

    for t in targets:
        RA = t[1]
        DEC = t[2]
        coordWorld = SkyCoord(ra=RA, dec=DEC, unit="deg")
        coordPix = coordWorld.to_pixel(crds, 0)
        allCoordPix.append(np.array([coordPix[0], coordPix[1]]))
        if show:
            plt.plot(coordPix[0], coordPix[1], marker='+', color='k')

    if show:
        plt.show()
    return allCoordPix, targetIDs, targetZs

def quick_plot(data=None, title="Default", color_map=CMAP, interpolation='antialiased', show=True):
    """
    Function to quickly plot data
    -----------------------------------------------
    Input:  data, data from FITS file [numpy.ndarray]
            title, title of plot
            map, which color map to use
            interpolation, type of interpolation to use
            show, display plot? [bool]
    Output: None
    """
    z1, z2 = ZScaleInterval().get_limits(values=data)

    plt.imshow(data, origin="lower", cmap=color_map, interpolation=interpolation, vmin=z1, vmax=z2)
    plt.title(title)
    if show:
        plt.show()
    return

def image_segmintation(data, display=True):
    """
    Capable of detecting sources/targets; prepares for isophote_fit()
    -----------------------------------------------
    Input:  data, data from FITS file [numpy.ndarray]
            display, display plots of the steps? [bool]
    Output: sources_x, x coordinates of the target's center [numpy.ndarray]
            sources_y, y coordinates of the target's center [numpy.ndarray]
            sources_eps, eccentricity of initial elliptic fit [numpy.ndarray]
            apers, photutils object used to later make isophotes [?]
    """
    convolved_FWHM = 1
    convolved_size = 5
    segment_npixels = 10
    seg_threshold = 0.3

    print("Plotting raw data...")
    quick_plot(data=data, title="Raw data")

    print("Convolving data with a 2D kernal...")
    kernel = make_2dgaussian_kernel(convolved_FWHM, size=convolved_size)
    convolved_data = convolve(data, kernel)
    if display:
        quick_plot(data=kernel, title="Kernal")

    print("Detecting sources in convolved data...")
    segment_map = detect_sources(convolved_data, seg_threshold, npixels=segment_npixels)
    if display:
        print(segment_map)

    print("Deblend overlapping sources...")
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
    if display:
        quick_plot(segment_map, title="Segmentation Image", color_map=segment_map.cmap, interpolation='nearest')

    print("Catalog sources...")
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

    print("Setting Kron apertures...")
    # norm = simple_norm(data, 'sqrt')
    quick_plot(segment_map, title='kron apertures', color_map=segment_map.cmap, show=False)
    cat.plot_kron_apertures({{'color': 'green'}, {'lw': 1.5}})
    plt.show()

    return sources_x, sources_y, sources_eps, apers

def isophote_fit(dat, aper, eps=0.01, perExtra=150):
    """
    Determine surface brights at points radius, r, outward using photoutils
    ---
    Input:  dat, intensity values for cropped area of fits file [numpy.ndarray]
            aper, photutils object used to make isophotes [?]
            eps, eccentricity of the isophote rings [float]
            perExtra, a percentatge of how far past the aperture isophote should be made [int]
    Output: isolist.sma, list of radii/sma values for isophotes [numpy.ndarray]
            isolist.intens, list of intensity/surface brightness values for isophotes [numpy.ndarray]
            isolist.int_err, 'The error of the mean intensity (rms / sqrt(# data points)).' [numpy.ndarray]
    Notes:  Algorithum used is from Jedrzejewski (1987; MNRAS 226, 747)
            https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract
    """
    z1, z2 = ZScaleInterval().get_limits(values=dat)  # MJy/sr

    cen = [aper.positions[0], aper.positions[1]] # Grab updated centers

    # plt.imshow(dat, origin='lower', vmin=z1, vmax=z2) # Plot ALL data from fits, bounded
    # aper.plot(color='r')

    g = EllipseGeometry(x0=cen[0], y0=cen[1], sma=aper.a, eps=eps, pa=(aper.theta / 180.0) * np.pi)
    ellipse = Ellipse(dat, geometry=g)
    isolist = ellipse.fit_image(maxsma=(aper.a*(perExtra/100))) # Creates isophotes using the geometry of 'g', so using above parameters as the bounds
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

    return isolist.sma, isolist.intens, isolist.int_err

def standard_fit(data, err, center, nRings=10, rLim=150, showText=True):
    """
    Fits rings aroung target to prepare for surface brightness; uses only circles
    -----------------------------------------------
    Input:  data, data from FITS file [numpy.ndarray]
            center, coordinates of the target (numpy.float64, numpy.float64)
            nRings, number of rings to form in fitting [int]
            rLim, max distance that rings can extend from center [int]
            showText, display activity of fitting function? [bool]
    Output: radius, array of distance moving from the center [numpy.ndarray]
            SB, average intensity within circle at certain radii [numpy.ndarray]
            SB_err, error corrisponding to SB array [numpy.ndarray]
    """
    radius, SB, SB_err = np.array([]), np.array([]), np.array([])

    r = np.arange(0.1, rLim, rLim/nRings)
    radius = np.copy(r)

    skipFirstErr = True
    for r_int in r:
        aper = CircularAperture([center[0], center[1]], r_int)
        sums, sums_err = aper.do_photometry(data=data, error=err)
        if sums > 0:
            SB = np.append(SB, sums/(np.pi*(r_int**2)))
            if skipFirstErr:
                SB_err = np.append(SB_err, 0)
                skipFirstErr = False
            else:
                SB_err = np.append(SB_err, sums_err / (np.pi * (r_int ** 2)))
            if showText:
                print(str(r_int)+' / '+str(rLim), sums/(np.pi*(r_int**2)), sums_err/(np.pi*(r_int**2)))
        else:
            print("Detected empty/invalid region, skipping...")
            break

    return radius, SB, SB_err

def calc_petro(radius, SB, sens=0.01):
    """
    Calculates the Petrosian Radius of given target
    -----------------------------------------------
    Input:  radius, array of distance moving from the center [numpy.ndarray]
            SB, average intensity within circle at certain radii [numpy.ndarray]
            sens, the tolerance given to calculating the value of the petro radius [float]
    Output: petro_r, Petrosian Radius of the target [float]
    """
    adda = 0.2  # Target constant
    petro_r = 0 # Initalizing petrosian value

    for i in range(2, len(radius)):
        localSB = SB[i]
        r = radius[:i]
        SB_i = SB[:i]

        SBtoR = simpson(y=SB_i, x=r) * 2 * np.pi
        area = np.pi * (r[-1]**2)

        integratedSB = SBtoR / area

        if abs(integratedSB - (adda*localSB)) < sens:
            petro_r = radius[i]
            break
    return petro_r
