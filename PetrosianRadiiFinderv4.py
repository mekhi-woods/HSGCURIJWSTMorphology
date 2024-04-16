"""
Created by Mekhi D. Woods
04/03/2024
Current Version: 1.0

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



SYS_TIME = str(int(time.time())) # System Time, for purposes of naming files uniquely
PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
DISPLAY_MODE = False

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
                                                     "Petrosian Radius: " + str(self.petroR))
        # return(str(self.ID) + str(self.pos) + str(self.iso_radii) + str(self.SB) +
        #        str(self.SBerr) + str(self.iso_eps) + str(self.aper) + str(self.petroR))

    def print_all(self):
        print("\n" +
              "ID: " + str(self.ID) + '\n' +
              "Pos: " + str(self.pos) + '\n' +
              "Radii length: " + str(len(self.iso_radii)) + '\n' +
              "SB length: " + str(len(self.SB)) + '\n' +
              "SB Err length: " + str(len(self.SBerr)) + '\n' +
              "Eps: " + str(len(self.iso_eps)) + '\n' +
              str(self.aper) + '\n' +
              "PetroR: " + str(self.petroR))
        return

def get_data_fits(path=None, bin=0):
    """
    Read in data from fits file
    ---
    Input:  path, string; path to the fits file
            bin, int; the bin to pull data from
    Output: data_cropped, numpy.array; intensity values for cropped area of fits file
            hdr, astropy.io.fits.header.Header; header of the fits file
    """
    with astropy.io.fits.open(path) as hdul:
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

def image_segmintation(data, threshold=0.5, display=True):
    print("Plotting raw data...")
    quick_plot(data=data, title="Raw data")

    print("Convolving data with a 2D kernal...")
    convolved_data, kernel = image_seg_convolve(data, FWHM=3.0, size=5, display=display)

    print("Detecting sources in convolved data...")
    segment_map = image_seg_detect_sources(convolved_data, threshold=threshold, npixels=10, display=display)

    print("Deblend overlapping sources...")
    segm_deblend = image_seg_deblend(convolved_data, segment_map, display=display)

    print("Catalog sources...")
    cat, sources_x, sources_y, sources_eps, apers = image_seg_cat(data, segm_deblend, convolved_data, display=display)

    print("Setting Kron apertures...")
    image_seg_kron_apertures(convolved_data, cat, display=display)

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

def isophote_fit_image_aper(dat, aper, eps=0.01):
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
    isolist = ellipse.fit_image(maxsma=(aper.a*1.5)) # Creates isophotes using the geometry of 'g', so using above parameters as the bounds
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
    plt.title(SYS_TIME + "_" + str(ID) + '\nSurface Brightness Profile, ' + unit)
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

def petrosian_radius(radius=None, SB=None, eps=None, sens=0.01):
    adda = 0.2  # Target constant
    petro_r = 0

    for i in range(2, len(radius)):
        localSB = SB[i]
        integratedSB = petro_r_avg_SB(a=radius[:i], eps=eps[:i], SB=SB[:i])

        if abs(integratedSB - (adda*localSB)) < sens:
            petro_r = radius[i]
            break
    return petro_r

def petro_r_avg_SB(a=None, eps=None, SB=None):
    b = a[-1] - (eps[-1]*a[-1]) # Semi-minor Axis

    SBtoR = simpson(y=SB, x=a) * 2 * np.pi
    area = np.pi*a[-1]*b

    avgSBwithinR = SBtoR/area

    return avgSBwithinR

def world_to_pix(data, crd, targetPath):
    # Plot/Organize coords
    allCoordPix = []
    targetIDs = []


    targets = np.genfromtxt(targetPath, delimiter=',', skip_header=1, dtype=float)
    targets = targets[:, :4]
    targetIDs = targets[:, 0]



    plt.imshow(data, origin="lower", cmap='magma', vmin=0, vmax=0.5)
    plt.title("Targets from Gus List over FITS-C1009-T008-NIRCAM-F090W")
    for t in targets:
        RA = t[1]
        DEC = t[2]
        coordWorld = SkyCoord(ra=RA, dec=DEC, unit="deg")
        coordPix = coordWorld.to_pixel(crd, 0)
        allCoordPix.append(np.array([coordPix[0], coordPix[1]]))
        plt.plot(coordPix[0], coordPix[1], marker='+', color='g')
    plt.show()
    return allCoordPix, targetIDs

def crop_sorter(targets=None, data=None, display=False):
    crop = 70
    cropedTargets = []
    cropedtargetsPix = []
    for i in range(len(targets)):
        y, x = int(targets[i][0]), int(targets[i][1])
        cropedy, cropedx = y*(crop/y), x*(crop/x)
        tempcrop = data[x - crop:x + crop, y - crop:y + crop]
        cropedTargets.append(tempcrop)
        cropedtargetsPix.append([x*(crop/x), y*(crop/y)])

        if display:
            print("Center: ", str(x) + ", " + str(y))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Gus Target file [' + str(i) + "]")
            ax1.imshow(data, origin='lower', cmap='magma', vmin=0, vmax=0.5)
            ax1.plot(y, x, marker='+', color='g')
            ax2.imshow(tempcrop, origin='lower', cmap='magma', vmin=0, vmax=0.5)
            ax2.plot(cropedy, cropedx, marker='+', color='g')
            plt.show()


    return cropedTargets, cropedtargetsPix

if __name__ == "__main__":
    # OPEN FITS FILE
    mainPath = r'downloads\jw01181-c1009_t008_nircam_clear-f090w_i2d.fits'
    altPath=r'downloads/jw01181-c1009_t008_nircam_clear-f090w_i2d.fits'
    bin = 'SCI'
    print("Obtaining data from FITS...")
    with fits.open(altPath) as hdul:
        hdu = hdul[bin]
        data = hdu.data
        hdr = hdu.header
        datacoords = WCS(hdr)

    # OPEN TARGET LIST
    targetPath = r'targets.csv'
    print("Sorting target list...")
    targetsPix, targetIDs = world_to_pix(data, datacoords, targetPath)

    # SOURCE DETECTION
    sourceThreshold = 0.65
    print("Detecting sources (segmenting image)...")
    sources_x, sources_y, sources_eps, apers = image_segmintation(data, threshold=sourceThreshold, display=False)
    positions = []
    for i in range(len(sources_x)):
        positions.append(np.array([sources_x[i], sources_y[i]]))

    # DETERMINE SOURCE OVERLAP WITH TARGET LIST
    overlapSens = 20
    overlappedPositions = []
    overlappedEps = []
    overlappedApers = []
    overlappedIDs = []
    for i in range(len(targetsPix)):
        for j in range(len(positions)):
            if abs(positions[j][0] - targetsPix[i][0]) < overlapSens and abs(positions[j][1] - targetsPix[i][1]) < overlapSens:
                overlappedPositions.append(np.array([positions[j][0], positions[j][1]]))
                overlappedEps.append(sources_eps[j])
                overlappedApers.append(apers[j])
                overlappedIDs.append(targetIDs[i])

    # CHECK OVERLAP
    quick_plot(data, title='Overlap from Gus Targets & Source Detection', show=False)
    for t in targetsPix:
        plt.plot(t[0], t[1], marker='+', color='g')
    for p in overlappedPositions:
        plt.plot(p[0], p[1], marker='x', color='b')
    plt.xlim(-1000, 4500); plt.ylim(2500, 8250)
    plt.show()

    # MAKE PETRO OBJECTS
    petroObjs = []
    for i in range(len(overlappedPositions)):
        tempObj = petrosianObject(ID = int(overlappedIDs[i]), pos = overlappedPositions[i], aper=overlappedApers[i], iso_eps=float(overlappedEps[i]))
        petroObjs.append(tempObj)
        # tempObj.print_all()

    # PROCESSING
    for i in range(len(petroObjs)):
        # ISOPHOTE FIT
        print("[", petroObjs[i].ID, "] Fiting isophotes...")
        petroObjs[i].iso_radii, petroObjs[i].SB, petroObjs[i].SBerr, petroObjs[i].iso_eps, petroObjs[i].isolist = isophote_fit_image_aper(dat=data,
                                                                                                                                          aper=petroObjs[i].aper,
                                                                                                                                          eps=petroObjs[i].iso_eps)

        # PETROSIAN RADIUS
        petroSens = 0.1
        if len(petroObjs[i].iso_radii) > 0:
            print("[", petroObjs[i].ID, "] Calculating petrosian radii...")
            petro_r = petrosian_radius(radius=petroObjs[i].iso_radii, SB=petroObjs[i].SB, eps=petroObjs[i].iso_eps, sens=petroSens)
            petroObjs[i].petroR = petro_r
        else:
            print("[", petroObjs[i].ID, "] No meaningful fit was possible.")
            petroObjs[i].petroR = 0

        petroObjs[i].print_all()


    # DISPLAY
    os.mkdir('images/' + str(SYS_TIME))
    for obj in petroObjs:
        z1, z2 = ZScaleInterval().get_limits(values=data)
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('ID [' + str(obj.ID) + ']' + '\n' +
                     'Center: (' + str(round(obj.pos[0], 2)) + ', ' + str(round(obj.pos[1], 2)) + ') | ' +
                     'Petrosian Radius: ' + str(round(obj.petroR, 6)))

        # Raw Data
        ax1 = plt.subplot(223)
        crop = 70
        ax1.imshow(data, origin="lower", cmap='magma', vmin=z1, vmax=z2)
        cenx, ceny = int(obj.pos[0]), int(obj.pos[1])
        ax1.set_xlim(cenx - crop, cenx + crop)
        ax1.set_ylim(ceny - crop, ceny + crop)
        ax1.set_xlabel('[pixels]')
        ax1.set_ylabel('[pixels]')

        # Isophote Rings
        ax2 = plt.subplot(224)
        nRings = 10
        ax2.imshow(data, origin="lower", cmap='magma', vmin=z1, vmax=z2)
        if nRings == -1:  # nRings=-1 plots all the rings
            nRings = len(obj.isolist.to_table()['sma'])
        if nRings != 0 and len(obj.isolist.to_table()['sma']) > 0:  # Makes sure that there is data from the isophote fit
            rMax = obj.isolist.to_table()['sma'][-1]  # Largest radius
            rings = np.arange(0, rMax, rMax / nRings)
            rings += rMax / nRings
            for sma in rings:
                iso = obj.isolist.get_closest(sma)  # Displayed isophotes are just the closest isophotes to a certain desired sma, but
                                                    # there are more isophotes between the ones displayed.
                ax2.plot(iso.sampled_coordinates()[0], iso.sampled_coordinates()[1], color='g', linewidth=1)
        ax2.set_xlim(cenx - crop, cenx + crop)
        ax2.set_ylim(ceny - crop, ceny + crop)
        ax2.set_xlabel('[pixels]')
        ax2.set_ylabel('[pixels]')

        # Surface Brightness plot
        ax3 = plt.subplot(211)
        sigma = 10
        marker_size = 2
        unit = '[pixels]'
        # if units:
        #     r = r * PIX_SCALE
        #     unit = '[arcsec]'
        ax3.errorbar(obj.iso_radii, obj.SB, yerr=(obj.SBerr) * sigma, fmt='o', ms=marker_size)
        ax3.set_xlabel('radius [pixels]')
        ax3.set_ylabel('Intensity [MJy/sr]')

        plt.savefig('images/' + str(SYS_TIME) + '/' + str(obj.ID) + '_' + str(SYS_TIME) + '.png', dpi=300)
        plt.show()

    # PRINT RADII & ID TO FILE
    with (open('petrosians.csv', 'w') as f):
        f.write('ID,PETROSIAN,PIXCENTERX,PIXCENTERY\n')
        for obj in petroObjs:
            line = str(obj.ID) + ',' + str(obj.petroR) + ',' + str(obj.pos[0]) + ',' + str(obj.pos[1]) + '\n'
            f.write(line)


