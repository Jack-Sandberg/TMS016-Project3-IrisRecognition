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
        filterReal, filterImag = defined_gabor(normalized, .5, 3.0, 1.5)
        bitblockReal = getBitBlocks(filterReal, size = 8)
        bitblockImag = getBitBlocks(filterImag, size = 8)
        
        rotatedBlockData.append(np.concatenate([bitblockReal, bitblockImag]))
    bitblockData.append(np.stack(rotatedBlockData, 0))

bitblockData = np.stack(bitblockData, 0)

distances = np.zeros((N,N))
allDistances = []
sameLabelDistances = []
differentLabelDistances = []
for i in range(N):
    for j in range(N):
        #if labels[i][-1] == 'r' and labels[j][-1] == 'r':
        distance = np.min(np.mean(np.abs(bitblockData[i, numberOfRelativeRotations//2+1] - bitblockData[j]), axis = -1))
        distances[i,j] = distance
        if i != j: 
            allDistances.append(distance)
            if labels[i] == labels[j]: sameLabelDistances.append(distance)
            else: differentLabelDistances.append(distance)
allDistances = np.array(allDistances)
np.fill_diagonal(distances, np.inf)
classification = labels[np.argmin(distances, axis = 0)]
accuracy = np.mean(np.equal(classification, labels))

oneSampleAccuracyRepetitions = 30
oneSampleAccuracy = np.zeros(oneSampleAccuracyRepetitions)
for n in range(oneSampleAccuracyRepetitions):
    unique_labels, counts = np.unique(labels, return_counts=True)
    N_labels = len(unique_labels)
    assert np.equal(counts[0], counts).all(), "All labels must occur equally often."
    N_imagesamples = N // N_labels
    databaseLabelIdx = np.random.choice(N_imagesamples, size = N_labels, replace = True)
    reshapedBitBlock = bitblockData.reshape(N_labels, N_imagesamples, *bitblockData.shape[1:])
    databaseData = np.zeros((N_labels, *bitblockData.shape[1:]))
    nonDatabase = []
    for i in range(N_labels):
        databaseData[i] = reshapedBitBlock[i,databaseLabelIdx[i]]
        nonDatabase.extend([reshapedBitBlock[i, j] for j in range(N_imagesamples) if j != databaseLabelIdx[i]])
    nonDatabase = np.stack(nonDatabase)
    
    distances = np.zeros((N-N_labels, N_labels))
    for i in range(N-N_labels):
        for j in range(N_labels):
            distance = np.min(np.mean(np.abs(nonDatabase[i, numberOfRelativeRotations//2+1] - databaseData[j]), axis = -1))
            distances[i,j] = distance
    true_labels = np.repeat(np.arange(N_labels), N_imagesamples-1)
    predicted_labels = np.argmin(distances,axis = 1)
    oneSampleAccuracy[n] = np.mean(np.equal(true_labels, predicted_labels))

print(f"Accuracy: {accuracy*100:.3f}%")
print(f"OneSampleAccuracy: {oneSampleAccuracy.mean()*100:.3f}%, std: {oneSampleAccuracy.std()*100:.3f}%")


sameLabelDistances = np.array(sameLabelDistances)
differentLabelDistances = np.array(differentLabelDistances)

#plt.hist([sameLabelDistances, differentLabelDistances], bins = 30, label = ['same', 'different'])
plt.hist(sameLabelDistances, bins = 30, label = 'Equal', alpha = 0.5, density = True)
plt.hist(differentLabelDistances, bins = 30, label = 'Different', alpha = 0.5, density = True)
plt.xlabel('Hamming distance')
plt.legend()
plt.show()