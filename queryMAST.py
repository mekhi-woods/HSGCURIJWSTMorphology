import csv
import shutil
import numpy as np
import time as systime
from astroquery.mast import Observations
def read_data(path: str = 'files/NEDTargets.txt') -> dict[np.ndarray[str]]:
    """
    Opens data from '|'-seperated text file (NED Results file).
    """
    with open(path, 'r') as f: hdr = f.readline().split('|')
    data = np.genfromtxt(path, skip_header=1, delimiter='|', dtype=str)
    new_data = {}
    for p in hdr:
        data[:, hdr.index(p)] = np.char.strip(data[:, hdr.index(p)])  # Removes whitespace
        new_data.update({p: data[:, hdr.index(p)]})
    return new_data
def retrieve_known_ids(path: str = 'target_validation.txt') -> (list[str], list[str], list[str]):
    known_ra, known_dec, all_ids = [], [], []
    with open(path, 'r') as f:
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
    return known_ra, known_dec, all_ids
def remove_duplicate_ids(og_obs_id: list[str], og_obsid: list[str]) -> list[str]:
    obs_dict = {}
    for i in range(len(og_obs_id)):
        short_obs_id = og_obs_id[i].split('-')[0]
        if short_obs_id in obs_dict: continue
        obs_dict.update({short_obs_id: og_obsid[i]})
    final_ids = []
    for j in obs_dict:
        final_ids.append(obs_dict[j])
    return final_ids
def query(ra: float, dec: float, fov: float = 1.0) -> list[str] | None:
    try:
        # Query by criteria
        obs_table = Observations.query_criteria(dataproduct_type=["image"],
                                                obs_collection='JWST',
                                                calib_level=3,
                                                dataRights="public",
                                                s_ra=[ra - fov/2, ra + fov/2],
                                                s_dec=[dec - fov/2, dec + fov/2])

        # Readout
        if len(obs_table) == 0:
            print(f"No observations found!")
            return []
        else:
            valid_obsid = remove_duplicate_ids(list(obs_table['obs_id']), list(obs_table['obsid']))
            print(f"Observations found! [{len(valid_obsid)}]")
            return valid_obsid
    except Exception as e:
        print(f"Error Encounterd: {e}\n Pausing for 5 seconds...")
        systime.sleep(5)
        return None
def verify(ra: np.ndarray[str], dec: np.ndarray[str], save_loc: str = 'target_validation.txt') -> np.ndarray[str]:
    # Verify known values
    known_ra, known_dec, all_ids = retrieve_known_ids(save_loc)

    # Check if all values are already known
    if len(ra) == len(known_ra):
        print(f'All RA & DEC have previously been queried! Return saved data...')
        return all_ids

    with open(save_loc, 'a') as f:
        if len(known_ra) == 0: f.write('ra,dec,obsid\n') # Prevents adding header if data in file already

        # Check of RA & DEC already queried
        for n in range(len(ra)):
            print(n, end=': ')
            n_ids = query(float(ra[n]), float(dec[n]), fov = 1.0)
            if n_ids is not None:
                f.write(f'{ra[n]},{dec[n]},{n_ids}\n')
                for id in n_ids:
                    all_ids.append(id)
            else:
                f.write(f'{ra[n]},{dec[n]},[]\n')

    return all_ids
def stats(ids: np.ndarray[str]) -> list[str]:
    # Get duplicate frequencies
    all_ids, all_freq = [], []
    for i in set(ids):
        all_ids.append(i)
        all_freq.append(ids.count(i))

    # Read out
    print('Number of Observation IDs:', len(all_freq))

    # Most common IDs
    top_ids = []
    n = 5
    s = np.array(list(zip(all_ids, all_freq)))
    sorted_ids = [x for _, x in sorted(zip(all_freq, all_ids))]
    print(f"Top {n} Most Common Observation IDs:")
    for a in range(len(sorted_ids)-1, len(sorted_ids)-(n+1), -1):
        print(f"{len(sorted_ids) - a}. {sorted_ids[a]}, {s[np.where(s[:, 0] == sorted_ids[a])[0][0], 1]} counts")
        top_ids.append(sorted_ids[a])
    return top_ids
def download(id: str, save_loc: str = 'fits/') -> str:
    manifest = Observations.download_products(id,
                                              productType=["SCIENCE"],
                                              extension="fits",
                                              mrp_only=True)
    shutil.copyfile(manifest['Local Path'][0], save_loc+manifest['Local Path'][0].split('/')[-1])
    print(f"[*] Saved {id} to... {manifest['Local Path'][0]}")
    print(f"[*] Addtionally copied to... {save_loc+manifest['Local Path'][0].split('/')[-1]}")
    return manifest['Local Path'][0].split('/')[-1].split('.')[0]
def id_target_file(id: str, tar_data: dict[np.ndarray[str]], save_loc: str,
                   path: str = 'target_validation.txt') -> (list[str], list[str]):
    valid_ra, valid_dec = [], []
    with open(path, 'r') as f:
        hdr = f.readline()
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split(',')
            if len(data[i][2]) > 3:
                l = ''.join(data[i][2:])
                l = l[1:-2].replace("'", "").split(' ')
                if id in l:
                    # print(f"{id}: {l}")
                    valid_ra.append(float(data[i][0]))
                    valid_dec.append(float(data[i][1]))
    with open(save_loc, 'w') as f:
        f.write(f'# Data values for objects in {id}\n')
        f.write(f"# RA Range: {np.min(valid_ra)} - {np.max(valid_ra)}\n")
        f.write(f"# DEC Range: {np.min(valid_dec)} - {np.max(valid_dec)}\n")
        f.write('No.,RA,DEC,Redshift (z),Object Name,,\n')
        for i in range(len(valid_ra)):
            index = np.where(tar_data['RA'].astype(float) == valid_ra[i])[0][0]
            f.write(f"{i},{tar_data['RA'][index]},{tar_data['DEC'][index]},{tar_data['Redshift'][index]},{tar_data['Object Name'][index]},,\n")

    return valid_ra, valid_dec
def main():
    start = systime.time()  # Runtime tracker
    data = read_data('files/NEDTargets.txt')
    ids = verify(data['RA'], data['DEC'], save_loc='target_validation.txt')
    stats(ids)

    # # LARGE DOWNLOADS, verify ids list before running!
    # for n_id in top_ids[:10]:
    #     download_name = download(n_id)
    #     id_target_file(n_id, data, save_loc=f'id_files/targetFile_{download_name}.txt', path='target_validation.txt')
    return

if __name__ == '__main__':
    start = systime.time()  # Runtime tracker
    main()
    print('|---------------------------|\n Run-time: ', round(systime.time() - start, 4), 'seconds\n|---------------------------|')
