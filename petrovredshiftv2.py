import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cos

PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
c = 3e5 # km/s
H0 = 73.8 #km/s/Mpc, using current WMAP data
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def lamda_rest(lamdaObs, z=0):
    return lamdaObs / (z+1)

def filter_selector(filterProps, z=0):
    return


def filters():
    filterRoot = 'filters/'
    filterTargets = ['F090W.txt', 'F115W.txt', 'F182M.txt', 'F210M.txt', 'F356W.txt', 'F444W.txt']
    filterRange = []

    plt.figure(figsize=(8, 4))
    for i in range(len(filterTargets)):
        dat = np.genfromtxt(filterRoot+filterTargets[i], skip_header=1, delimiter=' ')
        plt.plot(dat[:, 0], dat[:, 1], COLORS[i], label=filterTargets[i])
        plt.legend()
        
        filterRange.append([filterTargets[i], np.min(dat[:, 0]), np.max(dat[:, 0])])
    plt.title('Filters @ rest (z=0)')
    plt.ylim(0, 0.8)
    plt.show()
    return filterRange

if __name__ == "__main__":
    # filterProps = filters()








    data = np.genfromtxt('petrosians/vTEST_c1009_t008_petrosians.csv', delimiter=',', skip_header=1)
    IDs = data[:, 0]
    petroR = data[:, 1]
    petroRkpc = data[:, 2]
    centers = []
    for i in range(len(data[:, 3])):
        centers.append(np.array([data[i, 3], data[i, 4]]))
    redshift = data[:, 5]

    # petroRkpc = ((c*redshift) / H0) * petroR * PIX_SCALE * (np.pi/(180*3600)) * 1000

    plt.figure(figsize=(12, 6))
    plt.scatter(redshift, petroRkpc, c='m', s=50, marker='o')
    plt.title('Petrosian Radius vs. Redshift')
    plt.xlabel('Redshift, z'); plt.ylabel('Petrosian Radius, r [kpc]')
    plt.xlim(0.4, 1.6)
    plt.legend(labels=["$H_{0}$"+' = 73.8, using WMAP7'], loc='upper right')
    plt.show()




