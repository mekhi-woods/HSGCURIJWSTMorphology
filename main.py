# M. D. Woods
# A. T. Coffey
# 10/23/2024

import time as systime
import petroRadii.jwstpetrosian as jp
from scripts import general as gen
CONSTANTS = gen.get_constants()

def run(fits_path: str, target_path: str) -> None:
    """
    Main process.
    :param fits_path: path to fits file
    :param target_path: path to target file
    """
    print("Obtaining data from FITS...")
    hdr, data, err, datcrd, z1, z2 = jp.FITS_info(path=mainPath, wrkBin='SCI', errBin='ERR')

    print("Displaying targets against data...")
    targetPixCords, targetIDs, targetZs = jp.world_to_pix(data=data, path=targetPath, crds=datcrd, z1=z1, z2=z2, show=True)
    targetPixCords = targetPixCords[:10]  # SLICED FOR TESTING

    print("Initializing Petrosian Objects...")
    allPetrosians = []  # Container for petrosian objects
    for i in range(len(targetPixCords)):
        allPetrosians.append(
            jp.PetrosianObject(ID=targetIDs[i], z=targetZs[i], pos=(targetPixCords[i][0], targetPixCords[i][1])))
    #
    print("Fitting data (standard/circular fit)...")
    tempContainer = []
    for i in range(len(allPetrosians)):
        print("[" + str(i + 1) + '/' + str(len(allPetrosians)) + "] Standard fit...")  # Enumerates through the loop
        radius, SB, SB_err = jp.standard_fit(data=data, err=err, center=allPetrosians[i].pos, nRings=100, rLim=150,
                                             showText=False)
        print(" Fit finished.\n")
        if len(SB) != 0:
            allPetrosians[i].radius = radius
            allPetrosians[i].SB = SB
            allPetrosians[i].SBerr = SB_err
            tempContainer.append(allPetrosians[i])
    allPetrosians = tempContainer  # Update petrosian object container with valid objects

    print("Finding petrosian radii...")
    allPetroR = []
    for i in range(len(allPetrosians)):
        print("[" + str(i + 1) + '/' + str(
            len(allPetrosians)) + "] Calculating radius...")  # Enumerates through the loop
        petroR = jp.calc_petro(radius=allPetrosians[i].radius, SB=allPetrosians[i].SB, sens=CONSTANTS['PETRO_SENS'])
        print('[' + str(i + 1) + '/' + str(len(allPetrosians)) + '] Petrosian Radius = ', petroR)
        allPetrosians[i].petroR = petroR

    print('Displaying data...')
    for i in range(len(allPetrosians)):
        print("[" + str(i + 1) + '/' + str(len(allPetrosians)) + "] Plotting...")  # Ennumerates through the loop
        allPetrosians[i].display_self(data)

        print('Pausing for ' + str(CONSTANTS['PAUSE_TIME']) + ' seconds...\n')
        systime.sleep(CONSTANTS['PAUSE_TIME'])  # Pause is necessary or matplotlib gets angry, if you get display errors then increase the time
    return


if __name__ == '__main__':
    # mainPath = r'fits/jw02736-o001_t001_nircam_clear-f090w_i2d.fits' # All nan
    # mainPath = r'fits/jw01181-o009_t009_nircam_clear-f090w_i2d.fits' # All off image
    mainPath = r'fits/jw02514125001_03201_00002_nrca2_i2d.fits' # Some on image
    targetPath = r'files/targets.csv'

    run(mainPath, targetPath)
