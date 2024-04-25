'''
labels.py
This file creates the labels and generates the training and testing data 
Completed by Rucha Pendharkar on 4/24/24 

'''
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Function that converts RGB values to labels
def RGB2Label(label):
        
    # Convert the Hex codes into RGB colors    
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

    label_segment = np.zeros(label.shape,dtype = np.uint8)
    label_segment[np.all(label == Water, axis=-1)] = 0
    label_segment[np.all(label == Land, axis=-1)] = 1
    label_segment[np.all(label == Road, axis=-1)] = 2
    label_segment[np.all(label == Building, axis=-1)] = 3
    label_segment[np.all(label == Vegetation, axis=-1)] = 4
    label_segment[np.all(label == Unlabeled, axis=-1)] = 5
    label_segment = label_segment[:,:,0]

    return label_segment

#main function
def main():
    
    print('...Loading datasets!...')
    imageDataset = np.load('/home/rucha/CS5330/Final Project/image_dataset.npy')
    maskDataset = np.load('/home/rucha/CS5330/Final Project/mask_dataset.npy')

    print("...Loaded datasets!...")

    labels = []

    #Convert RGB to Labels
    for i in range(maskDataset.shape[0]):
        print(i)
        label = RGB2Label(maskDataset[i])
        labels.append(label)

    labels = np.array(labels)
    labels = np.expand_dims(labels,axis=3)

    print("Total unique values {}".format(np.unique(labels)))
    #print(labels.shape)

    #Get number of classes for the data
    total_classes = len(np.unique(labels))
    #print(total_classes)


    labelsDataset = to_categorical(labels,num_classes=total_classes)
    #print(labelsDataset.shape)
    #print(labelsDataset[0][0][0])

    #Split the data into training and testing images
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(imageDataset, labelsDataset, 
                                                                  test_size = 0.4, 
                                                                  shuffle = True, 
                                                                  random_state = 3)
    
    #Verify shapes of training and testing images
    print(len(train_imgs))
    print(len(test_imgs))
    print(len(train_masks))
    print(len(test_masks))

    #Save training and testing data
    np.save('/home/rucha/CS5330/Final Project/train_images.npy', train_imgs)
    np.save('/home/rucha/CS5330/Final Project/train_masks.npy', train_masks)
    np.save('/home/rucha/CS5330/Final Project/test_images.npy', test_imgs)
    np.save('/home/rucha/CS5330/Final Project/test_masks.npy', test_masks)

    print("Saved Testing and Training Data!")

if __name__ == "__main__":
    main()