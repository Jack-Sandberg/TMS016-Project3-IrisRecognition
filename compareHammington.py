import enum
from math import dist
from analyzeRatings import getRatingsLRAndNameDict, getfilteredRatingsList, constructDataset
from loadData import getOutputDataPaths
from gaborWavelets import defined_gabor, getBitBlocks
import numpy as np
import matplotlib.pyplot as plt

ratingsLRAndNameDict = getRatingsLRAndNameDict()
filteredNames = getfilteredRatingsList(ratingsLRAndNameDict)
_, outputDataPaths = getOutputDataPaths()
normalizedData, labels = constructDataset(outputDataPaths, filteredNames)

numberOfRelativeRotations = 7
maxRelativeRotationAngle = 90
thetaDimSize = normalizedData.shape[-1]
rotations = np.round(np.linspace(-maxRelativeRotationAngle/360 * thetaDimSize, 
                        maxRelativeRotationAngle/360 * thetaDimSize, 
                        numberOfRelativeRotations)).astype(int)

relativeRotatedData = np.zeros((numberOfRelativeRotations, *normalizedData.shape))
for rot_idx, rotation in enumerate(rotations):
    relativeRotatedData[rot_idx] = np.roll(normalizedData, rotation, axis = -1)
relativeRotatedData = relativeRotatedData.swapaxes(0,1)

N = len(labels)

bitblockData = []
for i in range(N):
    rotatedBlockData = []
    for j in range(numberOfRelativeRotations):
        normalized = relativeRotatedData[i,j,:,:]
        filterReal, _ = defined_gabor(normalized, 1.5, 3.0, 1.5)
        bitblock = getBitBlocks(filterReal, size = 8)
        rotatedBlockData.append(bitblock)
    bitblockData.append(np.stack(rotatedBlockData, 0))

bitblockData = np.stack(bitblockData, 0)

distances = np.zeros((N,N))
allDistances = []
sameLabelDistances = []
differentLabelDistances = []
for i in range(N):
    for j in range(N):
        distance = np.min(np.mean(np.abs(bitblockData[i, numberOfRelativeRotations//2+1] - bitblockData[j]), axis = -1))
        distances[i,j] = distance
        if i != j: 
            allDistances.append(distance)
            if labels[i] == labels[j]: sameLabelDistances.append(distance)
            else: differentLabelDistances.append(distance)
allDistances = np.array(allDistances)
sameLabelDistances = np.array(sameLabelDistances)
differentLabelDistances = np.array(differentLabelDistances)

#plt.hist([sameLabelDistances, differentLabelDistances], bins = 30, label = ['same', 'different'])
plt.hist(sameLabelDistances, bins = 30, label = 'same', alpha = 0.5, density = True)
plt.hist(differentLabelDistances, bins = 30, label = 'different', alpha = 0.5, density = True)

plt.legend()
plt.show()