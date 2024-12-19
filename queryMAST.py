import csv
import numpy as np
import time as systime
from astroquery.mast import Observations

# # ONLY SET TO TRUE IF YOU ARE CERTAIN YOU WANT TO DOWNLOAD THE FILES FROM YOUR SEARCH
# download = False  # Opts in or out to downloading fits files for your search
#
# # Iterating over list of objects
# f = open('files/NED_results_2000_0000.txt', 'r')
# reader = csv.DictReader(f, delimiter='|')
# for row in reader:
#
#     # Pulling the RA and Dec for each target with SLS distinction
#     if 'SLS' in row['Redshift Flag']:
#         ID = row['No.']
#         RA = float(row['RA'])
#         dec = float(row['DEC'])
#
#         # Creating box size for coordinate search
#         fov = 0.02  # Range for RA and Dec values
#         decRange = (float(dec) - fov/2, float(dec) + fov/2)
#         RARange = (float(RA) - fov/2, float(RA) + fov/2)
#
#         # Searching MAST for RA and Dec range
#         obs_table = Observations.query_criteria(s_dec=decRange,
#                                                 s_ra=RARange,
#                                                 intentType="science",
#                                                 dataproduct_type='IMAGE',
#                                                 obs_collection='JWST;HST',
#                                                 dataRights="public",
#                                                 calib_level=3,
#                                                 )
#
#         # Checking for observations
#         productIDs = list(obs_table['obsid'])
#         if len(productIDs) > 0:
#             print(len(productIDs))
#             print(RA, dec, ID)
#
#             # Downloading observations
#             if download:
#                 Observations.download_products(productIDs,
#                                                productType=["SCIENCE"],
#                                                extension="fits",
#                                                mrp_only=True)
#
def read_data() -> dict[np.ndarray[str]]:
    """
    Opens data from '|'-seperated text file (NED Results file).
    """
    hdr = ('No.|Object Name|RA|DEC|Type|Velocity|Redshift|Redshift Flag|Magnitude and Filter|Separation|References|'
           'Notes|Photometry Points|Positions|Redshift Points|Diameter Points|Associations').split('|')
    data = np.genfromtxt('files/NEDTargets.txt', delimiter='|', dtype=str)
    new_data = {}
    for p in hdr:
        data[:, hdr.index(p)] = np.char.strip(data[:, hdr.index(p)])  # Removes whitespace
        new_data.update({p: data[:, hdr.index(p)]})
    return new_data
def verify(ra: np.ndarray[str], dec: np.ndarray[str], save_loc: str = 'target_validation.txt') -> np.ndarray[str]:
    # Verify known values
    known_ra, known_dec, all_ids = [], [], []
    with open(save_loc, 'r') as f:
        hdr = f.readline()
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split(',')
            known_ra.append(float(data[i][0]))
            known_dec.append(float(data[i][1]))
            if len(data[i][2]) > 3:
                l = ''.join(data[i][2:])
                l = l[1:-2].replace("'", "").split(' ')
                for n_l in l:
                    all_ids.append(n_l)

    with open(save_loc, 'a') as f:
        if len(known_ra) == 0: f.write('ra,dec,obsid\n') # Prevents adding header if data in file already

        # Check of RA & DEC already queried
        for n in range(len(ra)):
            if float(ra[n]) in known_ra or float(dec[n]) in known_dec:
                print(f"{n}: RA & DEC previously queried!")
                continue

            try:
                # Give range of RA and DEC values
                fov = 2
                RARange = [float(ra[n]) - fov/2, float(ra[n]) + fov/2]
                decRange = [float(dec[n]) - fov/2, float(dec[n]) + fov/2]

                # Query by criteria
                obs_table = Observations.query_criteria(dataproduct_type=["image"],
                                                        obs_collection='JWST',
                                                        s_ra=RARange,
                                                        s_dec=decRange)

                # Write to file
                f.write(f"{ra[n]},{dec[n]},{list(obs_table['obsid'])}\n")

                # Save Observation IDs
                for id in list(obs_table['obsid']):
                    all_ids.append(id)

                # Readout
                if len(obs_table) == 0:
                    print(f"{n}: No observations found!")
                else:
                    print(f"{n}: Observations found! [{len(obs_table)}]")

            except Exception as e:
                print(f"Error Encounterd: {e}\n Pausing for 5 seconds...")
                systime.sleep(5)
    return np.unique(all_ids)
def download(ids: np.ndarray[str]) -> None:
    if len(ids) > 0:
        Observations.download_products(ids,
                                       productType=["SCIENCE"],
                                       extension="fits",
                                       mrp_only=True)
    return
def main():
    data = read_data()
    ids = verify(data['RA'], data['DEC'], save_loc='target_validation.txt')
    print('# of Observation IDs:', len(ids), '\n',ids)
    # download(ids)
    return


if __name__ == '__main__':
    main()
