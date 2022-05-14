import numpy as np
from loadData import getOutputDataPaths

def getRatingsLRAndNameDict():
    ratingsLRAndNameDict = {}
    with open('dataRatings.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            individual, imgName, rating = line.split(', ')
            leftOrRight = imgName[-2]
            rating = rating[0]
            if individual in ratingsLRAndNameDict:
                ratingsLRAndNameDict[individual][0].append(rating)
                ratingsLRAndNameDict[individual][1].append(leftOrRight)
                ratingsLRAndNameDict[individual][2].append(imgName)
            else:
                ratingsLRAndNameDict[individual] = [[rating], [leftOrRight], [imgName]]
    return ratingsLRAndNameDict

def getfilteredRatingsList(ratingsLRAndNameDict, edgeIsBad = False):
    filteredNames = []
    for individual, ratingsLRAndName in ratingsLRAndNameDict.items():
        ratings, leftOrRights, names = ratingsLRAndName
        hasABad = 'b' in ratings
        hasAnEdge = 'e' in ratings
        for name in names:
            if not hasABad:
                if edgeIsBad and hasAnEdge:
                    continue
                else:
                    filteredNames.append(name)
    return filteredNames

def constructDataset(dataPaths, names):
    normalizedData = []
    labels = []
    unwrappedPaths = [p for person in dataPaths for lr in person for p in lr]
    for path in unwrappedPaths:
        for name in names:
            if name in path:
                x = np.load(path)
                y = name[:-1]
                normalizedData.append(x)
                labels.append(y)
    normalizedData = np.stack(normalizedData, 0)
    labels = np.array(labels, dtype = object)
    return normalizedData, labels


if __name__ == '__main__':
    ratingsLRAndNameDict = getRatingsLRAndNameDict()
    numberOfRatings = 0
    totalGood, totalBad, totalEdge = 0,0,0
    onlyGood, atleastOneBad, atleastOneEdge = 0,0,0
    noBad = 0

    for indivial, ratingsLRAndName in ratingsLRAndNameDict.items():
        ratings = ratingsLRAndName[0]
        if 'b' in ratings: atleastOneBad += 1
        if 'e' in ratings: atleastOneEdge += 1

        if not 'b' in ratings and not 'e' in ratings:
            onlyGood += 1
        if not 'b' in ratings:
            noBad += 1

        for r in ratings:
            if r == 'g': totalGood += 1
            elif r == 'b': totalBad += 1
            elif r == 'e': totalEdge += 1
            numberOfRatings += 1

    print(f"Good images: {totalGood}, Bad images: {totalBad}, Edge cases: {totalEdge}")
    print(f"Good images: {totalGood/numberOfRatings*100:.2f}%, Bad images: {totalBad/numberOfRatings*100:.2f}%, Edge cases: {totalEdge/numberOfRatings*100:.2f}%")

    print(f"Number of individuals with: ")
    print(f"Only good: {onlyGood}, at least one bad: {atleastOneBad}, at least one edge: {atleastOneEdge}")
    print(f"No bad: {noBad}")
    
    filteredNames = getfilteredRatingsList(ratingsLRAndNameDict)
    _, outputDataPaths = getOutputDataPaths()
    normalizedData, labels = constructDataset(outputDataPaths, filteredNames)
