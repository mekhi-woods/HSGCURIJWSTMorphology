"""
Created by Mekhi D. Woods
02/22/2024
Version 7.3, GOALS: (1) Reliably ID centroids
                    (2) Identify r_max instead of specifying
    Initial Ideas:  (1) Just take the center of the crop and let photutils.isophote.EllipseGeometry.find_center() do the rest
                    (2) Iterate from center via circles, sum area, stop when the variation is past a certain limit
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse
from photutils.aperture import EllipticalAperture
from photutils.isophote import build_ellipse_model
from astropy.visualization import ZScaleInterval

time_s = str(int(time.time())) # System Time, for purposes of naming files uniquely
pix_scale = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera

def getDataFITS(path=None, crop_x=None, crop_y=None):
    """
    Read in data from fits file
    ---
    Input:  path, string; path to the fits file
            crop_x, list; x-axis crop
            crop_y, list; y-axis crop
    Output: data_cropped, numpy.array; intensity values for cropped area of fits file
            center_cropped, numpy.array; center of galaxy via max value
            hdr, astropy.io.fits.header.Header; header of the fits file
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul['SCI'].data
        err = hdul['ERR'].data

        data_cropped = data[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    return data_cropped, hdr

def findCenter(data=None):
    """
    Find center of galaxy
    ---
    Input:  data, numpy.array; intensity values for cropped area of fits file
    Output: cen, numpy.array; center of galaxy
    """
    cen = [np.where(data == np.max(data))[1][0],
           np.where(data == np.max(data))[0][0]] # Center = max flux

    return cen

def findMaxRadius(data=None, cen=None):
    """
    Find maximum radius to iterate over
    ---
    Input:  data, numpy.array; intensity values for cropped area of fits file
    Output: r_max, float; maximum radius to iterate over
    """
    z1, z2 = ZScaleInterval().get_limits(values=data) # z-scale for plot

    plt.imshow(data, vmin=z1, vmax=z2, origin='lower', cmap='magma') # Plot data
    plt.plot(cen[0], cen[1], 'r+') # Plot center point

    radii = np.arange(1, 250, 10)
    r_p = 0.0
    for r in radii:
        aper = CircularAperture(cen, r)
        r_c = aper.do_photometry(data)[0][0]
        print(r_c - r_p)
        r_p = r_c
        aper.plot()

    plt.show()

    return 10.0

def isophoteFitImage(data=None, cen_i=None, r_max=20, n_rings=0):
    """
    Determine surface brights at points radius, r, outward using photoutils
        1. Start at center_initial with esp=0, pa=0, sma=1
        2. Create isophote at position
        3. Keep track of esp, pa, sma/r, intens, int_err
        4. Plot ISOPHOTE
        5. Repeat 2-4 until r_max
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
            cen, numpy.array; center point of galaxy
            radius, int; initial radius of the ellipse to start solving for the isophotes
            n_rings, int; number of isophotes to display later
    Output: isolist.sma, numpy.array; list of radii/sma values for isophotes
            isolist.intens, numpy.array; list of intensity/surface brightness values for isophotes
            isolist.int_err, numpy.array; list of err from isophote sum
            isos, list; list of reconstructed ʻringsʻ/isophotes at some interval, r_max/n_rings
            cen_new, final calculated center of isophote calculating
    Notes:  Algorithum used is from Jedrzejewski (1987; MNRAS 226, 747)
            https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract
    """
    # Create isophote out to r_max
        # Works by starting in the center with a fit of a circle, then iterating at steps of 0.1
        # until it hits maxsma=r_max. Each iteration slightly changes the eps, pa, and x-y center.
    z1, z2 = ZScaleInterval().get_limits(values=data)  # MJy/sr

    g = EllipseGeometry(x0=cen_i[0], y0=cen_i[1], sma=r_max, eps=0.01, pa=(120 / 180.0) * np.pi)    # Make the outline of the initial guess using
                                                                                                    # above parameters
    g.find_center(data) # This runs a seperate fitting algorithim to adjust center from previous step.
                        # It then updates the values g.x0 and g.y0 automatically
                        # https://photutils.readthedocs.io/en/stable/_modules/photutils/isophote/geometry.html#EllipseGeometry.find_center
    cen_new = [g.x0, g.y0] # Grab updated centers
    ellipse = Ellipse(data, geometry=g) # make ellipse with area eqivalent to values in FITS data, but restricted to ellipse shape
    plt.imshow(data, origin='lower', vmin=z1, vmax=z2) # Plot ALL data from fits, bounded
    aper = EllipticalAperture((g.x0, g.y0), g.sma,
                              g.sma * (1 - g.eps),
                              g.pa)                         # Preps to plot graphical representation of 'g'
    aper.plot(color='r') # Plots 'g'
    isolist = ellipse.fit_image(maxsma=r_max) # Creates isophotes using the geometry of 'g', so using above parameters as the bounds
    print("Number of isophotes: ", len(isolist.to_table()['sma']))

    # Plots the isophotes over some interval
    isos = [] # A list of isophote x-y positions to plot later
    if n_rings != 0 and len(isolist.to_table()['sma']) > 0: # Makes sure that there is data from the isophote fit
        rings = np.arange(0, r_max, r_max/n_rings)
        rings += r_max/n_rings
        for sma in rings:
            iso = isolist.get_closest(sma) # Displayed isophotes are just the closest isophotes to a certain desired sma, but
                                           # there are more isophotes between the ones displayed.
            isos.append(iso.sampled_coordinates())
            plt.plot(iso.sampled_coordinates()[0], iso.sampled_coordinates()[1], color='k', linewidth=0.5)
    plt.show()

    return isolist.sma, isolist.intens, isolist.int_err, isos, cen_new

def data_visualizer(dat=None, cen_i=None, cen_f=None, rings=None, units=False, save=False):
    """
    Uses matplotlib.pyplot.imshow() to display the data.
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
            cen_i, numpy.array; intial guess for the center point of galaxy
            cen_f, numpy.array; final center point determine via function
                   'photutils.isophote.Ellipse.ellipse.fit_image()'
            units, bool; True-arcseconds, False-pixels
            save, bool; save visualization or not
    Output: None
    """
    # Calculate zscale data range | interval based on IRAF’s zscale
    z1, z2 = ZScaleInterval().get_limits(values=dat) # MJy/sr

    if units:
        # Fix Axies to arcseonds
        axis_scale = np.shape(data)
        axis_scale = np.array([0, axis_scale[1] * pix_scale, 0, axis_scale[0] * pix_scale])

        # Arcseconds
        plt.imshow(dat, vmin=z1, vmax=z2, origin='lower', cmap='magma', extent=axis_scale)

        plt.plot(cen_i[0] * pix_scale, cen_i[1] * pix_scale, 'r+',
                 label='Initial Center ' + str(cen_i[0] * pix_scale) + ", " + str(cen_i[1] * pix_scale))
        plt.plot(cen_f[0] * pix_scale, cen_f[1] * pix_scale, 'b+',
                 label='Final Center ' + str(round(cen_f[0] * pix_scale, 3)) + ", " + str(round(cen_f[1] * pix_scale, 3)))

        plt.title(time_s + '\n' + "Range: " + str(round((z1*pix_scale), 3)) + "-" + str(round((z2*pix_scale), 3)) + " MJy/sr" + " [arcsec]")
        plt.xlabel("[arcsec]"); plt.ylabel("[arcsec]")
    else:
        # Pixels
        plt.imshow(dat, vmin=z1, vmax=z2, origin='lower', cmap='magma')

        plt.plot(cen_i[0], cen_i[1], 'r+',
                 label='Initial Center: ' + str(cen_i[0]) + ", " + str(cen_i[1]))
        plt.plot(cen_f[0], cen_f[1], 'b+',
                 label='Final Center: '+ str(cen_f[0]) + ", " + str(cen_f[1]))

        plt.title(time_s + '\n' + "Range: " + str(round(z1, 3)) + "-" + str(round(z2, 3)) + " MJy/sr" + " [pixels]")
        plt.xlabel("[pixels]"); plt.ylabel("[pixels]")

    # Plot isophotes
        # The plot isn't of the actual isophote still, instead it finds a close neighbor and plots that
    if rings != None:
        for iso in rings:
            if units:
                plt.plot(iso[0] * pix_scale, iso[1] * pix_scale, color='g', linewidth=0.5)
            else:
                plt.plot(iso[0], iso[1], color='g', linewidth=0.5)

    # Save / Display
    plt.legend()
    if save:
        plt.savefig(r"results\GalaxyVisual_" + time_s + ".png")
    plt.show()

    return None

def plotSBProfile(r=None, SB=None, err=None, sigma=10, r_forth=False, units=False, save=False):
    """
    Plot the surface brightness profile
    ---
    Input:  r, numpy.array; radius/semimajor axis of galaxy
            SB, numpy.array; surface brightness for each point of radius
            err, numpy.array; error of surface brightness calculation / meassurement error
            sigma, int; scales the error bars
            r_forth, bool; swaps mode to plot r to the 1/4
            save, bool; save plot or not
    Output: None
    """
    unit = '[pixels]'
    marker_size = 2
    if units:
        r = r * pix_scale
        unit = '[arcsec]'

    if r_forth:
        # Plot SB vs radius [arcsec]
        plt.errorbar(r**(1/4), SB, yerr=(err) * sigma, fmt='o', ms=marker_size)
        plt.xlabel("Radius, r**1/4 " + unit)
    else:
        # Plot SB vs radius [arcsec]
        plt.errorbar(r, SB, yerr=(err) * sigma, fmt='o', ms=marker_size)
        plt.xlabel("Radius, r " + unit)

    plt.title(time_s + '\nSurface Brightness Profile, ' + unit)
    plt.ylabel("Intensity, I [MJy/sr]")
    if save:
        plt.savefig(r"results\SBprofile_"+str(int(time_s))+".png")
    plt.show()

    return None

if __name__ == "__main__":
    path = r"F:\School\HSGC URI JWST Morphology\downloads\140267160\mastDownload\JWST\jw01181-o098_t010_nircam_clear-f115w\jw01181-o098_t010_nircam_clear-f115w_i2d.fits"
    # crop_y, crop_x = [3625, 3750], [1525, 1700]
    # crop_y, crop_x = [3425, 3950], [1225, 2000] # Eliptical, but zoomed out more
    crop_y, crop_x = [2675, 2975], [275, 500] # Alt Galaxy, Spiral?
    real_units = False

    print("Obtaining data from FITS...")
    data, header = getDataFITS(path=path, crop_x=crop_x, crop_y=crop_y)

    print("Obtaining intital center..")
    cen = findCenter(data=data)

    # print("Obtaining maximum radius..")
    # r_max = findMaxRadius(data=data, cen=cen)

    print("Creating isophotes...")
    # # Elliptical
    # radius, SB, err, isophotes, cen_new = isophoteFitImage(data=data, cen_i=cen, r_max=80, n_rings=59) # Good fit

    # Spiral
    radius, SB, err, isophotes, cen_new = isophoteFitImage(data=data, cen_i=cen, r_max=120, n_rings=5) # Good fit
    radius, SB, err, isophotes, cen_new = isophoteFitImage(data=data, cen_i=cen, r_max=120, n_rings=59) # Good fit
    # plotSBProfile(r=radius, SB=SB, err=err, sigma=10, r_forth=False, units=real_units, save=False)

    print("Displaying data...")
    data_visualizer(dat=data, cen_i=cen, cen_f=cen_new, rings=isophotes, units=real_units, save=False) # With isophotes
    data_visualizer(dat=data, cen_i=cen, cen_f=cen_new, rings=None, units=real_units, save=False) # Without isophotes

    print("Plotting surface brightness profile...")
    plotSBProfile(r=radius, SB=SB, err=err, sigma=10, r_forth=False, units=real_units, save=False)
