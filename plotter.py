import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas
import galaxies as gals
import sys
import time

# data = pandas.read_csv('files/finalFile.csv')
#
# idList = data['ID'].tolist()
# raList = data['RA'].tolist()
# decList = data['dec'].tolist()
# zList = data['z'].tolist()
# AListList = data['absAList'].tolist()
# sqrdAListList = data['sqrdAList'].tolist()
# errAListList = data['errAList'].tolist()
# minAList = data['absAMin'].tolist()
# minSqrdAList = data['sqrdAMin'].tolist()
# minAErrList = data['absAMinErr'].tolist()
#
# # print(minAErrList)
# plt.errorbar(zList, minAList, yerr=minAErrList, fmt='o')
# # plt.scatter(zList, minAList)
# plt.show()
# plt.clf()
#
# plt.scatter(zList, minSqrdAList)
# plt.show()
# plt.clf()

absAList = []
absErrList = []
sqrdAList = []
sqrdErrList = []
multiAbsList = []
multiAbsErrList = []
multiSqrdList = []
multiSqrdErrList = []
zList = []
f = open('files/targets.csv', 'r')
reader = csv.DictReader(f)
for row in reader:
    ID = row['No.']
    RA = row['RA']
    dec = row['Dec']
    z = float(row['Redshift (z)'])
    name = row['Object Name']

    galaxy = gals.RegularGalaxy(ID, RA, dec, z, name)
    bool = galaxy.isolate(display=False)
    if bool:
        abs, sqrd = galaxy.asymmetry(sqrd=True)
        absA, errAbsA = abs
        sqrdA, errSqrdA = sqrd
        absAList.append(absA)
        absErrList.append(errAbsA)
        sqrdAList.append(sqrdA)
        sqrdErrList.append(errSqrdA)
        zList.append(z)

        multiAbsA, multiAbsErr = galaxy.asymmetry(multi=True)
        multiSqrdA, multiSqrdErr = galaxy.asymmetry(abso=False, sqrd=True, multi=True)
        multiAbsList.append(multiAbsA)
        multiAbsErrList.append(multiAbsErr)
        multiSqrdList.append(multiSqrdA)
        multiSqrdErrList.append(multiSqrdErr)

plt.scatter(absAList, sqrdAList)
plt.title('Squared A vs Absolute A')
plt.ylabel('Squared A')
plt.xlabel('Absolute A')
plt.show()
plt.clf()

plt.scatter(zList, absAList)
plt.title("Absolute Asymmetry")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

plt.errorbar(zList, absAList, yerr=10*np.array(absErrList), fmt='o')
plt.title("Absolute Asymmetry With Error")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

plt.scatter(zList, sqrdAList)
plt.title("Squared Asymmetry")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

plt.errorbar(zList, sqrdAList, yerr=10*np.array(sqrdErrList), fmt='o')
plt.title("Squared Asymmetry With Error")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

multiZList = []
for i in range(len(zList)):
    list = [zList[i]]*len(multiAbsList[i])
    multiZList.append(list)

for i in range(len(multiZList)):
    plt.scatter(multiZList[i], multiAbsList[i])
# plt.scatter(multiZList, multiAbsList)
plt.title('Varying Center Absolute Asymmetry')
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

for i in range(len(multiZList)):
    plt.scatter(multiZList[i], multiSqrdList[i])
plt.title('Varying Center Squared Asymmetry')
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()


