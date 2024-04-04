"""
Created by Mekhi D. Woods
03/02/2024
Current Version: 7.4
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse
from photutils.aperture import EllipticalAperture
from astropy.visualization import ZScaleInterval
from scipy import integrate

SYS_TIME = str(int(time.time())) # System Time, for purposes of naming files uniquely
PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
DISPLAY_MODE = True

def get_data_fits(path=None, cropX=None, cropY=None):
    """
    Read in data from fits file
    ---
    Input:  path, string; path to the fits file
            cropX, list; x-axis crop
            cropY, list; y-axis crop
    Output: data_cropped, numpy.array; intensity values for cropped area of fits file
            hdr, astropy.io.fits.header.Header; header of the fits file
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul['SCI'].data

        data_cropped = data[cropY[0]:cropY[1], cropX[0]:cropX[1]]

    return data_cropped, hdr

def find_center(dat=None):
    """
    Find center of galaxy
    ---
    Input:  dat, numpy.array; intensity values for cropped area of fits file
    Output: cen, numpy.array; center of galaxy
    """
    cen = [np.where(dat == np.max(dat))[1][0],
           np.where(dat == np.max(dat))[0][0]] # Center = max flux

    return cen

def isophote_fit_image(dat=None, cen0=None, rInit=1, rMax=20, eps=0.01, pa=0, nRings=0):
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

    # Create isophote out to r_max
        # Works by starting in the center with a fit of a circle, then iterating at steps of 0.1
        # until it hits maxsma=r_max. Each iteration slightly changes the eps, pa, and x-y center.
    g = EllipseGeometry(x0=cen0[0], y0=cen0[1], sma=rInit, eps=eps, pa=(pa / 180.0) * np.pi)    # Make the outline of the initial guess using
                                                                                                    # above parameters
    g.find_center(dat) # This runs a seperate fitting algorithim to adjust center from previous step.
                        # It then updates the values g.x0 and g.y0 automatically
                        # https://photutils.readthedocs.io/en/stable/_modules/photutils/isophote/geometry.html#EllipseGeometry.find_center
    cen_new = [g.x0, g.y0] # Grab updated centers
    ellipse = Ellipse(dat, geometry=g) # make ellipse with area eqivalent to values in FITS data, but restricted to ellipse shape
    plt.imshow(dat, origin='lower', vmin=z1, vmax=z2) # Plot ALL data from fits, bounded
    aper = EllipticalAperture((g.x0, g.y0), g.sma,
                              g.sma * (1 - g.eps),
                              g.pa)                         # Preps to plot graphical representation of 'g'
    aper.plot(color='r') # Plots 'g'
    isolist = ellipse.fit_image(maxsma=rMax) # Creates isophotes using the geometry of 'g', so using above parameters as the bounds
    print("Number of isophotes: ", len(isolist.to_table()['sma']))

    # Plots the isophotes over some interval
    isos = [] # A list of isophote x-y positions to plot later
    if nRings != 0 and len(isolist.to_table()['sma']) > 0: # Makes sure that there is data from the isophote fit
        rings = np.arange(0, rMax, rMax / nRings)
        rings += rMax / nRings
        for sma in rings:
            iso = isolist.get_closest(sma) # Displayed isophotes are just the closest isophotes to a certain desired sma, but
                                           # there are more isophotes between the ones displayed.
            isos.append(iso.sampled_coordinates())
            plt.plot(iso.sampled_coordinates()[0], iso.sampled_coordinates()[1], color='k', linewidth=0.5)
    if DISPLAY_MODE:
        plt.show()

    return isolist.sma, isolist.intens, isolist.int_err, isos, cen_new

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

def plot_sb_profile(r=None, SB=None, err=None, sigma=10, r_forth=False, units=False, save=False):
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

    if r_forth:
        # Plot SB vs radius [arcsec]
        plt.errorbar(r**(1/4), SB, yerr=(err) * sigma, fmt='o', ms=marker_size)
        plt.xlabel("Radius, r**1/4 " + unit)
    else:
        # Plot SB vs radius [arcsec]
        plt.errorbar(r, SB, yerr=(err) * sigma, fmt='o', ms=marker_size)
        plt.xlabel("Radius, r " + unit)

    plt.title(SYS_TIME + '\nSurface Brightness Profile, ' + unit)
    plt.ylabel("Intensity, I [MJy/sr]")
    if save:
        plt.savefig(r"results\SBprofile_" + str(int(SYS_TIME)) + ".png")
    plt.show()

    return None

def calc_petrosian_radius(SB=None, radius=None):
    adda = 0.2 # Target constant

    # Ranges: 0.5, 0.75, 1 (50%, 75%, 100% of radius)
    rtarget = int(SB.size*0.75)-1
    nummerArr = [int(SB.size*0.5)-1, int(SB.size*1)-1]

    # Calc nummerator of pertrosian function, fancyR
    numerInterior = (2*np.pi*SB[nummerArr[0]:nummerArr[1]]) * radius[nummerArr[0]:nummerArr[1]]
    numerSB = integrate.trapezoid(numerInterior, radius[nummerArr[0]:nummerArr[1]])
    numerArea = (np.pi*(radius[nummerArr[1]]**2)) - (np.pi*(radius[nummerArr[0]]**2)) # Inner - Outer Area (ring @ 50% and 75% radius)
    numerator = numerSB/numerArea

    # Calc denomerator of pertrosian function, fancyR
    denomInterior = (2*np.pi*SB[:rtarget]) * radius[:rtarget]
    denomSB = integrate.trapezoid(denomInterior, radius[:rtarget]) # dx evolves with x=radius[:rtarget]
    denomArea = (np.pi*(radius[rtarget]**2)) # Area within radius = r_target
    denomenator = denomSB/denomArea

    # Final calculation of pertrosian function, fancyR
    fancyR = numerator/denomenator

    # Find at what radius is the SB closest to fancyR
    targetSB = fancyR*adda
    adjustedSB = np.absolute(SB - targetSB)
    targetIndex = adjustedSB.argmin()
    petrosianRadius = radius[targetIndex]

    ### DEBUG
    # print("Nearest element to the given values is : ", SB[index])
    # print("Index of nearest value is : ", index)
    # print(fancyR*adda)
    # print(SB)
    # print(SB.size)
    # print(radius.size)
    # print(int(SB.size*r))
    # print(nummerArr)

    return petrosianRadius

if __name__ == "__main__":
    mainPath = r'downloads\jw01181-o098_t010_nircam_clear-f115w_i2d.fits'
    realUnits = False

    print("Obtaining data from FITS...")
    # cropY, cropX = [3625, 3750], [1525, 1700]
    cropY, cropX = [3425, 3950], [1225, 2000] # Eliptical, but zoomed out more
    # cropY, cropX = [2675, 2975], [275, 500] # Alt Galaxy, Spiral
    data, header = get_data_fits(path=mainPath, cropX=cropX, cropY=cropY)
    # with fits.open(mainPath) as hdul:
    #     header = hdul[0].header
    #     data = hdul['SCI'].data
    #     data = data[cropY[0]:cropY[1], cropX[0]:cropX[1]]

    print("Obtaining initial center..")
    # cen = find_center(dat=data)
    cen = [np.where(data == np.max(data))[1][0],
           np.where(data == np.max(data))[0][0]] # Center = max flux

    print("Creating isophotes...")
    # Elliptical
    radius, SB, err, isophotes, cen_new = isophote_fit_image(dat=data, cen0=cen, rMax=150, nRings=50)

    # # Spiral
    # radius, SB, err, isophotes, cen_new = isophote_fit_image(dat=data, cen0=cen, rInit=130, rMax=150, eps=0.7, pa=120, nRings=50)

    if DISPLAY_MODE:
        print("Displaying data...")
        data_visualizer(dat=data, cen0=cen, cenFinal=cen_new, rings=isophotes, units=realUnits, save=False) # With isophotes
        data_visualizer(dat=data, save=False)

        print("Plotting surface brightness profile...")
        plot_sb_profile(r=radius, SB=SB, err=err, sigma=10, r_forth=False, units=realUnits, save=False)

    # print("Calculating Petrosian Radius...")
    # petrosianRadius = calc_petrosian_radius(SB=SB, radius=radius)
    # print("The Calculated Petrosian Radius is: ", petrosianRadius)
    # print("The end of SB array is r = ", radius[-1])