'''
This file preprocess the data 
Completed by Rucha Pendharkar on 4/24/24 

'''
import cv2
from PIL import Image
import numpy as np
from patchify import patchify 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
def processImages(imageType, imageExtension):

    #File paths 
    datasetFolder = '/home/rucha/Segmentation-using-Deep-Networks'
    datasetName = 'Semantic segmentation dataset'

    dataset = []

    for tile in range(9):
        print(f"Processing Tile {tile}")
        for image_id in range(1,10):
            image_path = f'{datasetFolder}/{datasetName}/Tile {tile}/{imageType}/image_part_00{image_id}.{imageExtension}'
            testImage = cv2.imread(image_path)
            if testImage is not None:
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

    imageDataset = processImages('images', 'jpg')
    maskDataset = processImages('masks', 'png')
    print(imageDataset)

    # Save pre processed images and masks
    np.save('/home/rucha/Segmentation-using-Deep-Networks/image_dataset.npy', imageDataset)
    np.save('/home/rucha/Segmentation-using-Deep-Networks/mask_dataset.npy', maskDataset)

    print("Done!")

    #Split the data into training and testing data
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(imageDataset, maskDataset, 
                                                                  test_size = 0.2, 
                                                                  shuffle = True, 
                                                                  random_state = 3)
    
    print(len(train_imgs))
    print(len(test_imgs))
    print(len(train_masks))
    print(len(test_masks))

    plt.subplot(1, 2, 1)
    plt.imshow(train_imgs[1])

    plt.subplot(1, 2, 2)
    plt.imshow(train_masks[1])
    plt.show()

    #Save the data
    print("...saving the data...")
    np.save('/home/rucha/Segmentation-using-Deep-Networks.npy', train_imgs)
    np.save('/home/rucha/Segmentation-using-Deep-Networks.npy', train_masks)
    np.save('/home/rucha/Segmentation-using-Deep-Networks.npy', test_imgs)
    np.save('/home/rucha/Segmentation-using-Deep-Networks.npy', test_masks)

    print("Saved Testing and Training Data!")

if __name__ == "__main__":
    main()