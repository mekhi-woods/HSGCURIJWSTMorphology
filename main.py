# M. D. Woods
# A. T. Coffey
# 11/7/2024

import time as systime
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture
from scipy.integrate import simpson
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.wcs import WCS
from photutils.isophote import EllipseGeometry, EllipseSample, Isophote
from photutils.aperture import EllipticalAperture, CircularAperture, EllipticalAnnulus, CircularAnnulus

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
    def __init__(self, ID: str = 'None',
                 pixCoords: list = [np.nan, np.nan], realCoords: list =[np.nan, np.nan],
                 z: float = np.nan):
        self.ID = ID
        self.pixCoords = pixCoords
        self.realCoords = realCoords
        self.z = z
        self.petroR = np.nan
        self.petroR_kpc = np.nan
        self.SB = np.array([])
        self.SB_err = np.array([])
        self.radius = np.array([])
        return
    def __str__(self):
        return f"[{self.ID}] {self.realCoords} | {self.pixCoords} | {self.z}"
    def SSDS_fit(self, data, err, r_step: int = 1, r_limit: int = 30, r_step_limit: int = 40):
        """
        Fits the SB profile using the SSDS method
        """
        self.radius, self.SB, self.SB_err, self.annulus_SB, self.annulus_SB_err, self.fancy_R = (
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        # Steps from center to radius limit
        r_range = np.arange(0.1, r_limit, r_limit / r_step_limit)
        for r in r_range:
            # Create geometries
            annulus = CircularAnnulus([self.pixCoords[0], self.pixCoords[1]], 0.8*r, 1.25*r)
            elipse = EllipticalAperture([self.pixCoords[0], self.pixCoords[1]], r, r)

            # Get photmetric data
            annulus_intensity, annulus_intensity_err = annulus.do_photometry(data=data, error=err)
            elipse_intensity, elipse_intensity_err = elipse.do_photometry(data=data, error=err)

            # Get instantaneous area
            annulus_area = np.pi*(((1.25*r)**2) - ((0.8*r)**2))
            elipse_area = np.pi*(r**2)

            # Store data
            self.radius = np.append(self.radius, r)
            self.SB = np.append(self.SB, elipse_intensity[0] / elipse_area)
            self.SB_err = np.append(self.SB_err, elipse_intensity_err[0])
            self.annulus_SB = np.append(self.annulus_SB, annulus_intensity[0] / annulus_area)
            self.annulus_SB_err = np.append(self.annulus_SB_err, annulus_intensity_err[0])
            self.fancy_R = np.append(self.fancy_R,
                                     float(get_constants()['ADDA'])*((elipse_intensity[0] / elipse_area) / (annulus_intensity[0] / annulus_area)))

        # Calculate petrosian radius
        radius_copy = np.copy(self.radius)
        difference_array = np.abs(self.SB - self.fancy_R)
        radius_copy = radius_copy[difference_array < float(get_constants()['PETRO_SENS'])]
        difference_array = difference_array[difference_array < float(get_constants()['PETRO_SENS'])]
        if len(difference_array) != 0:
            self.petroR = radius_copy[difference_array == np.min(difference_array)][0]
            self.petroR_kpc = pix_to_kpc(self.z, self.petroR)
        else:
            self.petroR, self.petroR_kpc = np.nan, np.nan

        return
    def plot(self, data, save_fig: bool = False):
        # Set up subplot stuff
        fig = plt.figure(figsize=(11, 5), constrained_layout=True)
        gs = plt.GridSpec(nrows=6, ncols=2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax21 = fig.add_subplot(gs[:3, 1])
        ax22 = fig.add_subplot(gs[3:, 1])

        # Replot image data
        aper = CircularAperture([self.pixCoords[0], self.pixCoords[1]], self.petroR)
        ax1.imshow(data,
                     vmin=np.median(data[~np.isnan(data)])-0.5,
                     vmax=np.median(data[~np.isnan(data)])+0.5)
        ax1.plot(self.pixCoords[0], self.pixCoords[1], marker='+', color='k',
                   label=(f'ID: {int(self.ID)}'+' | $z_{?}$'+f': {self.z}\n'+
                          f'RA: {round(self.realCoords[0], 4)}, DEC: {round(self.realCoords[1], 4)}'))
        aper.plot(ax=ax1, color='red', label='Petrosian Radius')
        ax1.set_xlabel('[pix]'); ax1.set_ylabel('[pix]')
        ax1.set_ylim(self.pixCoords[1] - np.max(self.radius),
                       self.pixCoords[1] + np.max(self.radius))
        ax1.set_xlim(self.pixCoords[0] - np.max(self.radius),
                       self.pixCoords[0] + np.max(self.radius))
        ax1.legend(loc='upper left')

        # Plot SB v. Radius
        ax21.errorbar(pix_to_kpc(self.z, self.radius), self.SB, yerr=self.SB_err, fmt='o', color='blue')
        ax21.axvline(self.petroR_kpc, color='red', label=f'Petrosian Radius: {round(self.petroR_kpc, 2)}kpc')
        ax21.legend(loc='best')
        ax21.set_xlabel('Radius [kpc]')
        ax21.set_ylabel('[counts/pix]')

        # Plot residuaul data
        ax22.scatter(pix_to_kpc(self.z, self.radius), self.SB, marker='o', color='g', label='SB')
        ax22.scatter(pix_to_kpc(self.z, self.radius), self.fancy_R, marker='+', color='k', label='Fancy R')
        ax22.scatter(pix_to_kpc(self.z, self.radius), np.abs(self.fancy_R - self.SB), marker='.', color='orange', label='Resid')
        ax22.set_xlabel('Radius [kpc]')
        ax22.set_ylabel('Surface Brightness')
        ax22.legend()

        if save_fig:
            plt.savefig(f'{self.ID}.png')

        plt.show()

        return
    def SB_plot(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].errorbar(self.radius, self.SB, yerr=self.SB_err, color='r', fmt='o')
        axs[0].axvline(self.petroR)
        axs[1].scatter(self.radius, self.SB, marker='o', color='g', label='SB')
        axs[1].scatter(self.radius, self.fancy_R, marker='+', color='k', label='Fancy R')
        axs[1].scatter(self.radius, np.abs(self.fancy_R - self.SB), marker='.', color='orange', label='Resid')
        axs[1].legend()
        plt.show()

# General Functions ================================================================================================= #
def get_constants(constant_loc='constants.txt'):
    CONSTANTS = {}
    with open(constant_loc, 'r') as f:
        temp = f.readlines()
        for line in temp:
            line = line[:-1].split(', ')
            if len(line) != 2:
                continue
            CONSTANTS.update({line[0]: line[1]})
    return CONSTANTS
def get_info(path: str, wrkBin: str = 'SCI', errBin: str = 'ERR'):
    """
    Unpacks the FITS file
    :param path: str; path to FITS file to unpack
    :param wrkBin: str; FITS bin to use for unpacking data
    :param errBin: str; FITS bin to use for unpacking errors
    :return array(hdr: astropy.io.fits.header.Header, header of FITS
                  data: numpy.ndarray; data from FITS file
                  err: numpy.ndarray; error from FITS file
                  datcrd: astropy.wcs.wcs.WCS; object that helps to find coords of certain points
                  z1: numpy.float64; similar to DS9 z-scale, lower limit
                  z2: numpy.float64; similar to DS9 z-scale, upper limit)
    """
    with fits.open(path) as hdul:
        hdu = hdul[wrkBin]
        data = hdu.data
        hdr = hdu.header
        err = hdul[errBin].data
        data_coords = WCS(hdr)
        z1, z2 = ZScaleInterval().get_limits(values=data)
    return hdr, data, err, data_coords, z1, z2
def get_targets(data: np.array, target_path: str, coords: WCS, show: bool = False, quiet: bool = False) -> dict:
    real_targets = np.genfromtxt(target_path, delimiter=',', skip_header=1, dtype=float)[:, :4]
    target_data = {'ids': [], 'z': [],
                   'pixCoords': np.full((1, 2), np.nan),
                   'realCoords': np.full((1, 2), np.nan)}

    for t in real_targets:
        realCoord = SkyCoord(ra=t[1], dec=t[2], unit="deg")
        PixCoord = realCoord.to_pixel(coords, 0)

        # Check validity
        if (np.isnan(PixCoord[0]) or np.isnan(PixCoord[1]) or  # Check for nan value
            int(PixCoord[0]) < 0 or int(PixCoord[1]) < 0 or  # Check for negatives
            int(PixCoord[0]) > data.shape[0] or int(PixCoord[1]) > data.shape[1]):  # Check for out of bounds
            if not quiet:
                print(f"[!!!] Target '{t[1], t[2]} | {PixCoord[0], PixCoord[1]}' is at an invalid position! Removing...")
            continue

        # Save valid data
        target_data['realCoords'] = np.vstack((target_data['realCoords'], np.array([t[1], t[2]])))
        target_data['pixCoords'] = np.vstack((target_data['pixCoords'], np.array([PixCoord[0], PixCoord[1]])))
        target_data['ids'].append(t[0])
        target_data['z'].append(t[3])

    target_data['realCoords'] = target_data['realCoords'][1:, :]  # Remove empty head of array
    target_data['pixCoords'] = target_data['pixCoords'][1:, :]

    if show:
        plt.imshow(data,
                   vmin=np.median(data[~np.isnan(data)]) - 0.5,
                   vmax=np.median(data[~np.isnan(data)]) + 0.5)
        for p in target_data['pixCoords']:
            plt.plot(p[0], p[1], marker='+', color='r')
        plt.title(f'Current FITS w/ Targets overlaid [{len(target_data['pixCoords'])}]')
        plt.show()

    return target_data
def pix_to_kpc(z: float, d: float):
    CONSTANTS = get_constants()
    return (((float(CONSTANTS['SPEED_OF_LIGHT'])*z) / float(CONSTANTS['H0'])) *
                                   d * float(CONSTANTS['PIX_SCALE']) * (np.pi/(180*3600)) * 1000)

# Main Functions ==================================================================================================== #
def run(fits_path: str, target_path: str = '', fit_type: str = 'std',
        r_step: int = 1, r_limit: int = 30, r_step_limit: int = 40) -> None:
    """
    Main process.
    :param fits_path: path to fits file
    :param target_path: path to target file
    """
    # Load file information
    print("[+++] Obtaining data from FITS...")
    hdr, data, err, coords, z1, z2 = get_info(path=mainPath, wrkBin='SCI', errBin='ERR')

    # Source detection
    if len(target_path) == 0:
        print('[+++] Detecting sources in image...')
        return
    else:
        print('[+++] Using given target list to identify sources...')
        target_data = get_targets(data, target_path, coords, True, True)
    if len(target_data['pixCoords']) == 0:
        raise ValueError('[!!!] No valid souces found!')

    # Initialize objects
    print("[+++] Initializing Petrosian Objects...")
    allPetrosians = []  # Container for petrosian objects
    for i in range(len(target_data['ids'])):
        allPetrosians.append(PetrosianObject(ID=target_data['ids'][i], realCoords=target_data['realCoords'][i],
                                             pixCoords=target_data['pixCoords'][i], z=target_data['z'][i]))

    # Fitting SB profiles
    print("[+++] Fitting data with SSDS fit...")
    if fit_type == 'std':
        for obj in allPetrosians:
            obj.SSDS_fit(data, err, r_step, r_limit, r_step_limit)

    # Display results
    print('[+++] Displaying data...')
    for obj in allPetrosians:
        obj.plot(data, True)

    # Save data
    print('[+++] Saved data to... results.txt')
    with open('saved/results.txt', 'w') as f:
        f.write('# M.D. Woods -- using sens=0.1\n')
        f.write('ID,RA,DEC,Z,PETRO_PIX,PETRO_KPC\n')
        for obj in allPetrosians:
            f.write(f"{obj.ID},{obj.realCoords[0]},{obj.realCoords[1]},{obj.z},{obj.petroR},{obj.petroR_kpc}\n")

    return

if __name__ == '__main__':
    # mainPath = r'fits/jw02736-o001_t001_nircam_clear-f090w_i2d.fits' # All nan
    # mainPath = r'fits/jw01181-o009_t009_nircam_clear-f090w_i2d.fits' # All off image
    mainPath = r'fits/jw02514125001_03201_00002_nrca2_i2d.fits' # Some on image
    targetPath = r'files/targets.csv'

    fitting_params = {'r_step': 0.1, 'r_limit': 30, 'r_step_limit': 30}
    run(mainPath, targetPath, 'std', **fitting_params)
