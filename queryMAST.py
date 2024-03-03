import numpy as np
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os


def query(filter='F115W', dec=0, ra=0, sensitivity=0.01):
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


def query_count(filter='F115W', dec=0, ra=0, sensitivity=0.01):
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


def manifest_download(id=None, download_path="downloads\\"):
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
    return None



if __name__ == "__main__":
    # Construct targets list
    print("Constructing targets list...")
    targets_txt = np.genfromtxt(fname="targets.txt", delimiter="\t", dtype="float")
    targets = SkyCoord(ra=(targets_txt[:, 0]) * u.degree, dec=(targets_txt[:, 1]) * u.degree)
    print("Targets list constructed.")

    # manifest_download(id=140267160)


    for i in range(len(targets)):
        manifest_download(id=(query(ra=targets[i].ra.degree, dec=targets[i].dec.degree)))