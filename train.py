'''
train.py
This file calls the model and trains it. It also generates accuracy and loss plots 
Completed by Rucha Pendharkar on 4/24/24 

'''
from model import getNetwork, getNetworkWithAttention
import numpy as np
import matplotlib.pyplot as plt
import os
import segmentation_models as sm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_network(train_imgs_path, train_masks_path, test_imgs_path, test_masks_path):
    #model = getNetwork()
    #model.summary()

    model1 = getNetworkWithAttention()
    model1.summary()

    print("... Loading training data....")
    train_imgs = np.load(train_imgs_path)
    train_masks = np.load(train_masks_path)
    print("... Loading testing data....")
    test_imgs = np.load(test_imgs_path)
    test_masks = np.load(test_masks_path)
    print("Finished loading training and testing data!")

    model1.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model1.fit(np.array(train_imgs), np.array(train_masks),
                        batch_size=16,
                        verbose=1,
                        epochs=70,
                        validation_data=(np.array(test_imgs), np.array(test_masks)),
                        shuffle=False)

    # Save the trained model
    model1.save('Segmentation-Model-Attention-CC-Loss.h5')
    print("Model Saved!")
    
    return history

# Plot the accuracy and losses
def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
    plt.title("Training Vs Validation Accuracy")
    plt.xlabel("Epochs") 
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("Accuracy_Plot.png")
    plt.show()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label="Training Loss")
    plt.plot(epochs, val_loss, 'r', label="Validation Loss")
    plt.title("Training Vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Loss_Plot.png")
    plt.show()

def main():
    train_imgs_path = r'C:\Users\Rucha Pendharkar\Desktop\RuchaCV\train_images.npy'
    train_masks_path = r'C:\Users\Rucha Pendharkar\Desktop\RuchaCV\train_masks.npy'
    test_imgs_path = r'C:\Users\Rucha Pendharkar\Desktop\RuchaCV\test_images.npy'
    test_masks_path = r'C:\Users\Rucha Pendharkar\Desktop\RuchaCV\test_masks.npy'
    history = train_network(train_imgs_path, train_masks_path, test_imgs_path, test_masks_path)
    plot(history)

if __name__ == "__main__":
    main()
