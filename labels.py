'''
This file preprocess the labels 
Completed by Rucha Pendharkar on 4/24/24 

'''
import numpy as np 
import json
import random
import matplotlib.pyplot as plt
import cv2


def hexToRGB(hexColor): 

    #Converts a hexadecimal color code to an RGB tuple
    hexColor = hexColor.lstrip('#')

    return tuple(int(hexColor[i:i+2], 16) for i in (0, 2, 4))

def colorsToRGB(color_codes):
    #Converts a list of hexadecimal color codes to RGB tuples

    rgb_dict = {}
    for color_name, hex_code in color_codes.items():
        rgb_dict[color_name] = np.array(hexToRGB(hex_code))

    return rgb_dict


def main():
    print('...Loading datasets!...')
    imageDataset = np.load('image_dataset.npy')
    maskDataset = np.load('mask_dataset.npy')
    print("...Loaded datasets!...")

    random_image_id = random.randint(0, len(imageDataset))


    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.imshow(imageDataset[random_image_id])
    plt.subplot(122)
    plt.imshow(maskDataset[random_image_id])
    plt.show()

    # Load your mask image
    mask_image = cv2.imread('/home/rucha/CS5330/Final Project/archive/Semantic segmentation dataset/Tile 3/masks/image_part_004.png')

    # Check the shape of the image to verify it's in RGB format
    print("Image shape:", mask_image.shape)

    # Access individual pixel values
    # For example, to check the RGB values at pixel (x, y)
    x, y = 100, 200  # Example coordinates
    pixel_value = mask_image[y, x]
    print("RGB values at pixel (x={}, y={}):".format(x, y), pixel_value)

    # Load color codes from JSON file
    with open('/home/rucha/CS5330/Final Project/archive/Semantic segmentation dataset/classes.json', 'r') as f:
        data = json.load(f)

    # Extract class titles and colors
    class_colors = {}
    for item in data['classes']:
        title = item['title']
        color = item['color']
        # Convert color code to RGB tuple
        rgb_tuple = hexToRGB(color)
        class_colors[title] = rgb_tuple
        print(class_colors)

    class_building = '#D0021B'
    class_building = class_building.lstrip('#')
    class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))
    print(class_building)

    class_land = '#F5A623'
    class_land = class_land.lstrip('#')
    class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))
    print(class_land)

    class_road = '#DE597F'
    class_road = class_road.lstrip('#')
    class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))
    print(class_road)

    class_vegetation = '#417505'
    class_vegetation = class_vegetation.lstrip('#')
    class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))
    print(class_vegetation)

    class_water = '#50E3C2'
    class_water = class_water.lstrip('#')
    class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))
    print(class_water)

    class_unlabeled = '#9B9B9B'
    class_unlabeled = class_unlabeled.lstrip('#')
    class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))
    print(class_unlabeled)

    def rgb_to_label(label):
        label_segment = np.zeros(label.shape, dtype=np.uint8)
        label_segment[np.all(label == class_water, axis = -1)] = 0
        label_segment[np.all(label == class_land, axis = -1)] = 1
        label_segment[np.all(label == class_road, axis = -1)] = 2
        label_segment[np.all(label == class_building, axis = -1)] = 3
        label_segment[np.all(label == class_vegetation, axis=-1)] = 4
        label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
        return label_segment
    label = mask_image
    label = rgb_to_label(mask_image)
    print("Label for RGB values [41, 169, 226]:", label)

  # Convert RGB masks to label masks
    labels = []
    for i in range(maskDataset.shape[0]):
        label = rgb_to_label(maskDataset[i])
        labels.append(label)
     
    print(len(labels))
    # Convert list of labels to numpy array
    labels = np.array(labels)

    # Print unique labels
    print("Total unique labels based on masks:", np.unique(labels))

    # Display random image and its corresponding mask
    random_image_id = random.randint(0, len(imageDataset))

    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.imshow(imageDataset[random_image_id])
    plt.subplot(122)
    plt.imshow(labels[random_image_id][:,:,0])
    plt.show()


if __name__ == "__main__":
    main()