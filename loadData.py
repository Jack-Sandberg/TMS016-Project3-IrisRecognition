import os


def getDatasetImagePaths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'MMU-Iris-Database\\')
    numberOfIndividuals = (len(next(os.walk(data_dir))[1]))
    imagePaths = []
    for i in range(1,numberOfIndividuals+1):
        individualPathLeft = os.path.join(data_dir, f'{i}\\left\\')
        leftFiles = list(filter(lambda file: file.endswith('.bmp'), os.listdir(individualPathLeft)))
        leftFiles = [os.path.join(individualPathLeft, f) for f in leftFiles]
        individualPathRight = os.path.join(data_dir, f'{i}\\right\\')
        rightFiles = list(filter(lambda file: file.endswith('.bmp'), os.listdir(individualPathRight)))
        rightFiles = [os.path.join(individualPathRight, f) for f in rightFiles]
        
        imagePaths.append([leftFiles, rightFiles])
    return imagePaths

def getOutputDataPaths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'OutputDataNew\\')
    visualizationPaths, dataPaths = [], []
    numberOfIndividuals = (len(next(os.walk(data_dir))[1]))
    for i in range(1,numberOfIndividuals+1):
        individualPathLeft = os.path.join(data_dir, f'{i}\\left\\')
        individualPathRight = os.path.join(data_dir, f'{i}\\right\\')
        if not os.path.exists(individualPathLeft) or not os.path.exists(individualPathRight): continue
        leftVisFiles = list(filter(lambda file: file.endswith('.pdf'), os.listdir(individualPathLeft)))
        leftVisFiles = [os.path.join(individualPathLeft, f) for f in leftVisFiles]

        leftDataFiles = list(filter(lambda file: file.endswith('.npy'), os.listdir(individualPathLeft)))
        leftDataFiles = [os.path.join(individualPathLeft, f) for f in leftDataFiles]

        rightVisFiles = list(filter(lambda file: file.endswith('.pdf'), os.listdir(individualPathRight)))
        rightVisFiles = [os.path.join(individualPathRight, f) for f in rightVisFiles]

        rightDataFiles = list(filter(lambda file: file.endswith('.npy'), os.listdir(individualPathRight)))
        rightDataFiles = [os.path.join(individualPathRight, f) for f in rightDataFiles]

        visualizationPaths.append([leftVisFiles, rightVisFiles])
        dataPaths.append([leftDataFiles, rightDataFiles])
    return visualizationPaths, dataPaths
if __name__ == '__main__':
    imagePaths = getDatasetImagePaths()

    print(f"Number of indivduals: {len(imagePaths)}") 
    print(f"left-images of individual 1: {len(imagePaths[0][0])}")
    print(f"right-images of individual 1: {len(imagePaths[0][1])}")