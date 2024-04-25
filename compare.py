'''
compare.py 
Trains a pretrained model used for benchmarking performance
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import segmentation_models as sm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Plot the validation and training accuracy and losses
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
    plt.savefig("Accuracy_Plot_Pretrained.png")
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
    plt.savefig("Loss_Plot_Pretrained.png")
    plt.show()

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
train_imgs_path = './train_images.npy'
train_masks_path = './train_masks.npy'
test_imgs_path = './test_images.npy'
test_masks_path = './test_masks.npy'

print("... Loading training data....")
train_imgs = np.load(train_imgs_path)
train_masks = np.load(train_masks_path)
print("... Loading testing data....")
test_imgs = np.load(test_imgs_path)
test_masks = np.load(test_masks_path)
print("Finished loading training and testing data!")

# preprocess input
X_train_prepr = preprocess_input(train_imgs)
X_test_prepr = preprocess_input(test_imgs)


# define model
model_resnet = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=6, activation='softmax')
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

print(model_resnet.summary())

history=model_resnet.fit(X_train_prepr, 
          train_masks,
          batch_size=16, 
          epochs=70,
          verbose=1,
          validation_data=(X_test_prepr, test_masks))

model_resnet.save('Segmentation-Model-ResNet-CC.h5')
print("Model Saved!")
plot(history)
