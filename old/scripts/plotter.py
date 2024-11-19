import matplotlib.pyplot as plt
import csv

zList = []
absAListList = []
absErrListList = []
sqrdAListList = []
sqrdErrListList = []
f = open('files/asymmFile.csv', 'r')
reader = csv.DictReader(f)
for row in reader:
    zList.append(row['z'])
    absAListList.append(row['absAList'].split(' '))
    absErrListList.append(row['absErrList'].split(' '))
    sqrdAListList.append(row['sqrdAList'].split(' '))
    sqrdErrListList.append(row['sqrdErrList'].split(' '))




