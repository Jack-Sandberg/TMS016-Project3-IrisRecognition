from newCalculateIris import *
from loadData import getDatasetImagePaths

imagePaths = getDatasetImagePaths()

name = 'hock'
leftOrRight = 'l'
idx = 1
imagePath = None
for persons in imagePaths:
    for lr in persons:
        for img in lr:
            if (name + str(leftOrRight) + str(idx)) in img:
                imagePath = img

assert imagePath != None, f"Image {name + str(leftOrRight) + str(idx)}.bmp not found"

computeNormalizedIrisAndSave(imagePath, closeFigAfter=False, saveResults=False)

fig, ax = plt.subplots(2,2)
image = np.array(Image.open(imagePath))
ax[0,0].imshow(cv2.GaussianBlur(image, (5,5), 3,3))
ax[0,1].imshow(cv2.GaussianBlur(image, (15,15), 10,10))
ax[1,0].imshow(cv2.GaussianBlur(image, (11,11), 8,8))
ax[1,1].imshow(cv2.GaussianBlur(image, (9,9), 6,6))

plt.show()