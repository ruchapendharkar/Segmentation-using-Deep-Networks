from keras.callbacks import EarlyStopping
from model import *
import numpy as np
import matplotlib.pyplot as plt 

def show_result(id):
    fig, ax = plt.subplots(1, 3, figsize = (12, 7))
    
    ax[0].set_title('original')
    ax[0].imshow(test_imgs[id])
    ax[0].axis("off")
    
    ax[1].set_title('predict')
    ax[1].imshow(images_predict[id])
    ax[1].axis("off")
    
    ax[2].set_title('ground_truth')
    ax[2].imshow(test_masks[id])
    ax[2].axis("off")


earlystopping = EarlyStopping(monitor = 'val_loss',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')


input = Input(shape = (256, 256, 3))
output = UNet(input, dropout = 0.1)

model = Model(inputs = input, outputs = output)
model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

train_imgs = np.load('/home/rucha/CS5330/Final Project/train_images.npy')
train_masks = np.load('/home/rucha/CS5330/Final Project/train_masks.npy')
np.savez_compressed('train_data.npz', images=train_imgs, masks=train_masks)

test_imgs = np.load('/home/rucha/CS5330/Final Project/test_images.npy')
test_masks = np.load('/home/rucha/CS5330/Final Project/test_masks.npy')


history = model.fit(np.array(train_imgs), np.array(train_masks),
                    batch_size = 8, 
                    epochs = 2, 
                    validation_data = (np.array(test_imgs), np.array(test_masks)),
                    callbacks = [earlystopping])

plt.style.use('seaborn')
plt.figure(figsize = (11, 8))
plt.plot(history.history['loss'], c = 'orange', label = 'train_loss')
plt.plot(history.history['val_loss'], c = 'blue', label = 'val_loss')
plt.legend()
plt.show()

plt.style.use('seaborn')
plt.figure(figsize = (11, 8))
plt.plot(history.history['accuracy'], c = 'orange', label = 'train_acc')
plt.plot(history.history['val_accuracy'], c = 'blue', label = 'val_acc')
plt.legend()
plt.show()


images_predict = model.predict(np.array(test_imgs))

for id in range(5):
    show_result(id)

