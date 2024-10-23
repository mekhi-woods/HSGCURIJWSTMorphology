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

problemChildren = ['137']
# absAList = []
# absErrList = []
# sqrdAList = []
# sqrdErrList = []
multiAbsList = []
multiAbsErrList = []
multiSqrdList = []
multiSqrdErrList = []
zList = []

finalFile = open('files/asymmFile.csv', 'w')
writer = csv.writer(finalFile)
header = ['ID', 'RA', 'dec', 'z', 'absAList',
          'absErrList', 'sqrdAList', 'sqrdErrList']
writer.writerow(header)

targets = open('files/targets.csv', 'r')
reader = csv.DictReader(targets)
for row in reader:
    ID = row['No.']
    RA = row['RA']
    dec = row['Dec']
    z = float(row['Redshift (z)'])
    name = row['Object Name']

    if ID in problemChildren:
        continue

    galaxy = gals.RegularGalaxy(ID, RA, dec, z, name)
    bool = galaxy.isolate()
    # time.sleep(0.5)
    if bool:
        # abs, sqrd = galaxy.asymmetry(sqrd=True)
        # absA, errAbsA = abs
        # sqrdA, errSqrdA = sqrd
        # absAList.append(absA)
        # absErrList.append(errAbsA)
        # sqrdAList.append(sqrdA)
        # sqrdErrList.append(errSqrdA)
        zList.append(z)

        multiAbsA, multiAbsErr = galaxy.asymmetry(multi=True)
        multiSqrdA, multiSqrdErr = galaxy.asymmetry(abso=False, sqrd=True, multi=True)
        multiAbsList.append(multiAbsA)
        multiAbsErrList.append(multiAbsErr)
        multiSqrdList.append(multiSqrdA)
        multiSqrdErrList.append(multiSqrdErr)

        infoList = [ID, RA, dec, z, ' '.join(map(str, multiAbsA)), ' '.join(map(str, multiAbsErr)),
                    ' '.join(map(str, multiSqrdA)), ' '.join(map(str, multiSqrdErr))]
        writer.writerow(infoList)

targets.close()
finalFile.close()


for i in range(len(multiAbsList)):
    absArr = np.array(multiAbsList[i][:-1])
    j = np.where(absArr == absArr.min())[0][0]
    plt.scatter(absArr[j], multiSqrdList[i][:-1][j], color='blue')
plt.title('Squared A vs Absolute A')
plt.ylim(0, 0.2)
plt.xlim(0.025, 0.2)
plt.ylabel('Squared A')
plt.xlabel('Absolute A')
plt.show()
plt.clf()

for i in range(len(multiAbsList)):
    plt.scatter(zList[i], min(multiAbsList[i][:-1]), color='blue')
plt.title("Absolute Asymmetry")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

for i in range(len(multiAbsList)):
    absArr = np.array(multiAbsList[i][:-1])
    j = np.where(absArr == absArr.min())[0][0]
    plt.errorbar(zList[i], absArr[j], yerr=3*multiAbsErrList[i][:-1][j], fmt='o', color='blue')
plt.title("Absolute Asymmetry With Error")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

for i in range(len(multiSqrdList)):
    plt.scatter(zList[i], min(multiSqrdList[i][:-1]), color='blue')
plt.title("Squared Asymmetry")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

for i in range(len(multiAbsList)):
    absArr = np.array(multiSqrdList[i][:-1])
    j = np.where(absArr == absArr.min())[0][0]
    plt.errorbar(zList[i], absArr[j], yerr=3*multiSqrdErrList[i][:-1][j], fmt='o', color='blue')
plt.title("Squared Asymmetry With Error")
plt.ylabel('A')
plt.xlabel('z')
plt.show()
plt.clf()

multiZList = []
for i in range(len(zList)):
    list = [zList[i]]*len(multiAbsList[i][:-1])
    multiZList.append(list)

for i in range(len(multiZList)):
    plt.scatter(multiZList[i], multiAbsList[i][:-1], label=multiAbsList[i][-1])
# plt.scatter(multiZList, multiAbsList)
plt.title('Varying Center Absolute Asymmetry')
plt.ylabel('A')
plt.xlabel('z')
plt.legend(ncol=2, bbox_to_anchor=(1.1, 1.05))
plt.show()
plt.clf()

for i in range(len(multiZList)):
    plt.scatter(multiZList[i], multiSqrdList[i][:-1], label=multiSqrdList[i][-1])
plt.title('Varying Center Squared Asymmetry')
plt.ylabel('A')
plt.xlabel('z')
plt.legend(ncol=2, bbox_to_anchor=(1.1, 1.05))
plt.show()
plt.clf()


