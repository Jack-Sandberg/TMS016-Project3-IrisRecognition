import enum
from math import dist
from analyzeRatings import getRatingsLRAndNameDict, getfilteredRatingsList, constructDataset
from loadData import getOutputDataPaths
from gaborWavelets import defined_gabor, getBitBlocks
import numpy as np
import matplotlib.pyplot as plt
from irisNormalization import ImageEnhancement

ratingsLRAndNameDict = getRatingsLRAndNameDict()
filteredNames = getfilteredRatingsList(ratingsLRAndNameDict)
_, outputDataPaths = getOutputDataPaths()
normalizedData, labels = constructDataset(outputDataPaths, filteredNames)

normalized = normalizedData[0]

filterReal, filterImag = defined_gabor(normalized, .5, 3.0, 1.5)
bitblockReal = getBitBlocks(filterReal, size = 8)
bitblockImag = getBitBlocks(filterReal, size = 8)
#fig, ax = plt.subplots(2)
fig, ax = plt.subplots(1)
ax.imshow(normalizedData[0], cmap = 'gray')
print('hello')
fig, ax = plt.subplots(1)
ax.imshow(filterReal, cmap = 'gray')

fig, ax = plt.subplots(1)
ax.imshow(filterReal, cmap = 'gray')
for x in range(8, 512, 8):
    ax.axvline(x = x, c = 'r', linewidth = 0.2)
for y in range(8, 64, 8):
    ax.axhline(y = y, c = 'r', linewidth = 0.2)


fig, ax = plt.subplots(1)
ax.imshow(bitblockReal.reshape(32,16))

plt.show()

print('hello')