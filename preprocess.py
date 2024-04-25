'''
This file preprocess the images and masks
Completed by Rucha Pendharkar on 4/24/24 

'''
import cv2
from PIL import Image
import numpy as np
from patchify import patchify 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random 

# Calculates the values for cropping the image
def calculateCropSize(image, patchSize):

    sizeX = (image.shape[1] // patchSize) * patchSize
    sizeY = (image.shape[0] // patchSize) * patchSize

    return sizeX, sizeY

# Converts the images into patches
def patchImages(image, patchSize):

    # Convert image to numpy array
    image = np.array(image)

    # Split the image into small patches specified by patch size
    patchedImage = patchify(image, (patchSize, patchSize, 3), step=patchSize)

    return patchedImage

# Apply Min-Max scaling and normalize the image
def normalizeImage(patchedImage):
    scaler = MinMaxScaler()
    scaled_patch = scaler.fit_transform(patchedImage.reshape(-1, 3)).reshape(patchedImage.shape)
    scaled_patch = scaled_patch[0] 
    # Ensure pixel values are within [0, 1] range
    scaled_patch = np.clip(scaled_patch, 0, 1)
    return scaled_patch

# Preprocess Images and Masks  
def processImages(imageType, imageExtension, datasetFolder, datasetName):


    dataset = []

    for tile in range(1, 9):
        print(f"Processing Tile {tile}")
        for image_id in range(1,10):
            image_path = f'{datasetFolder}/{datasetName}/Tile {tile}/{imageType}/image_part_00{image_id}.{imageExtension}'
            testImage = cv2.imread(image_path)
            if testImage is not None:
                if imageType == 'masks':
                    testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
                sizeX, sizeY = calculateCropSize(testImage, 256)
                image = Image.fromarray(testImage)
                cropped_image = image.crop((0, 0, sizeX, sizeY))
                patched_image = patchImages(cropped_image, 256)
                for i in range(patched_image.shape[0]):
                    for j in range(patched_image.shape[1]):
                        patchedImage = patched_image[i, j, :, :]
                        scaledImage = normalizeImage(patchedImage)
                        #print(scaledPatch.shape)
                        dataset.append(scaledImage)
    
    return dataset

# main function
def main():
    
    #File paths 
    datasetFolder = '/Users/ruchinitsure/RuchaCV'
    datasetName = 'Semantic segmentation dataset'

    imageDataset = processImages('images', 'jpg', datasetFolder, datasetName)
    maskDataset = processImages('masks', 'png', datasetFolder,datasetName )

    # Save pre processed images and masks
    #np.save('/home/rucha/CS5330/Final Project/image_dataset.npy', imageDataset)
    #np.save('/home/rucha/CS5330/Final Project/mask_dataset.npy', maskDataset)

    print("Done!")

    #Test plot
    random_image_id = random.randint(0,len(maskDataset))
    plt.figure(figsize=(14,8))
    plt.subplot(121) 
    plt.title("Image")
    plt.imshow(imageDataset[random_image_id])
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(maskDataset[random_image_id])
    plt.show()

if __name__ == "__main__":
    main()