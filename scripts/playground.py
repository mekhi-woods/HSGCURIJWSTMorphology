import utilities as utils
import numpy as np
from PIL import Image as im
import random
import matplotlib.pyplot as plt
import data


###############################Unit converter#######################################
print(utils.time_to_degrees('12:36:53'))
print(utils.dms_to_degrees('62:12:57'))
print(utils.degrees_to_time('110.788652'))
print(utils.degrees_to_dms('-73.472292'))

################################centerPoint testing##################################
galaxyArr = np.load('files/testerGalaxy.npy')
origImage = im.fromarray(galaxyArr)

# Finding center of galaxy
originVal = galaxyArr.max()
(x, y) = np.where(galaxyArr == originVal)
originXY = (x[0], y[0])
randXY = (random.randint(0, 100), random.randint(0, 40))
gusXY = (77, 54)
# print(originXY)

croppedArr = utils.centerPoint(galaxyArr, gusXY)
croppedImage = im.fromarray(croppedArr)

# origImage.show()

croppedImage = croppedImage.convert('L')
# croppedImage.show()


croppedImage.save('wakawaka.jpg')

###########################errorSum Test#########################
array = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
print(array.shape)
print(utils.errorSum(array))

##########################mockGalaxy Test#########################
plt.imshow(utils.mockGalaxy(4, 6, 90, 39, 33))
plt.show()



