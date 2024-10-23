import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cos

PIX_SCALE = 0.031 # arcsec/pix, from https://jwst-docs.stsci.edu/jwst-near-infrared-camera
c = 3e5 # km/s
H0 = 73.8 #km/s/Mpc, using current WMAP data

if __name__ == "__main__":
    data = np.genfromtxt('../petrosians/vTEST_c1009_t008_petrosians.csv', delimiter=',', skip_header=1)
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







    # print('IDs: ', IDs)
    # print('petroR: ', petroR)
    # print('centers: ', centers)
    # print('redshift: ', redshift)

    # plt.figure(figsize=(12, 6))
    # for i in range(len(IDs)):
    #     plt.plot(redshift[i], petroRkpc[i], '.', markersize=20,
    #              label=str(IDs[i])+', ['+str(centers[i][0])+', '+str(centers[i][1]))
    #
    #     # plt.legend(loc='upper left', fontsize=8)
    #     # plt.xlim([0, 2.5]); plt.ylim([6, 30])
    #     # plt.show()
    # plt.title('Petrosian Radius vs. Redshift')
    # plt.xlabel('Redshift, z'); plt.ylabel('Petrosian Radius, r [kpc]')
    # # plt.gca().invert_xaxis()
    # # plt.ylim(-1, 120)
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()



    # d_l = cz / H0
    # arc = d_L * R_adda