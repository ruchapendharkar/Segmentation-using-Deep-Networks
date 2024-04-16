import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def show_result(id):
    fig, ax = plt.subplots(1, 3, figsize = (12, 7))
    
    ax[0].set_title('Original Image')
    ax[0].imshow(test_imgs[id])
    ax[0].axis("off")
    
    ax[1].set_title('Prediction')
    ax[1].imshow(images_predict[id])
    ax[1].axis("off")
    
    ax[2].set_title('Ground Truth')
    ax[2].imshow(test_masks[id])
    ax[2].axis("off")
    plt.show()

model = load_model('/home/rucha/Segmentation-using-Deep-Networks/trained_model.h5')
print("...Loading data....")
test_imgs = np.load('/home/rucha/Segmentation-using-Deep-Networks/test_images.npy')
test_masks = np.load('/home/rucha/Segmentation-using-Deep-Networks//test_masks.npy')

images_predict = model.predict(np.array(test_imgs))

for id in range(5):
    show_result(id)
