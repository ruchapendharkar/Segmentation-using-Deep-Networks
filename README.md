# Semantic Segmentation of Aerial Images using Deep Networks

## Created by Rucha Pendharkar on 4/24/24

The goal of this project is to build and train a deep network capable of performing semantic segmenation of aerial images. This project explores the application of the U-Net convolutional neural network architecture for the task of aerial image segmentation of the Dubai landscape. The unique architecture of U-Net, which integrates both contracting and expansive pathways, makes it a promising candidate for this task. Experimentation with model architecture and transfer learning are used to compare and contrast the results. 

## Link to Dataset - https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/

## Link to Trained Models - https://drive.google.com/drive/folders/1VR021d0radMMGMiILTESiozlQY_2I_WI?usp=sharing

## Features
- **preprocess.py**: For preprocessing images and masks in the dataset
- **labels.py** : Processes labels and creates training and testing data
- **model.py** : Contains definition of various networks
- **compare.py**: For training a pretrained model for transfer learning
- **predict.py** : For visualizing the results of all the trained models
- **train.py** : For training the networks

## Environment 
The scripts were authored using VS Code, and code compilation took place in the Ubuntu 20.04.06 LTS environment, utilizing CMake through the terminal.
