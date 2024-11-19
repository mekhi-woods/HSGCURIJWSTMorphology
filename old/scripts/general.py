import numpy as np  # default_imshow(), get_targets()
import matplotlib.pyplot as plt  # default_imshow()
from astropy.io import fits  # get_info()
from astropy.wcs import WCS  # get_info(), get_targets()
from astropy.visualization import ZScaleInterval  # get_info()
from astropy.coordinates import SkyCoord  # get_targets()


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
def default_imshow(data: np.array):
    plt.imshow(data, vmin=np.median(data[~np.isnan(data)]) - 0.5, vmax=np.median(data[~np.isnan(data)]) + 0.5)
    plt.show()
    return
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