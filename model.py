'''
This file contains the model definition
Completed by Rucha Pendharkar on 4/24/24 

'''
import tensorflow as tf
from keras.layers import Input
from keras.layers import Conv2D,  MaxPooling2D, BatchNormalization,  Dropout
from keras.models import Model
from keras.layers import Concatenate, UpSampling2D


def encode(x, dropout, filters):
  x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu')(x)
  x = BatchNormalization()(x)
  x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu')(x)
  x = Dropout(dropout)(x)
  pool = MaxPooling2D(pool_size = (2, 2))(x)

  return x, pool

def decode(x, skip_connect, dropout, filters):
  x = Concatenate(axis = -1)([skip_connect, UpSampling2D(size = (2, 2))(x)])
  x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu')(x)
  x = Dropout(dropout)(x)
  x = BatchNormalization()(x)
  x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu')(x)

  return x


def UNet(x, dropout, conv = [32, 64, 128, 256, 512]):

  # encoder
  conv1, pool1 = encode(x, dropout, conv[0]) 
  conv2, pool2 = encode(pool1, dropout, conv[1])
  conv3, pool3 = encode(pool2, dropout, conv[2]) 
  conv4, pool4 = encode(pool3, dropout, conv[3]) 

  conv5 = Conv2D(filters = conv[4], kernel_size = 3, padding = 'same', activation = 'relu')(pool4)
  conv5 = Dropout(dropout)(conv5)
  conv5 = BatchNormalization()(conv5)

  # decoder
  conv6 = decode(conv5, conv4, dropout, conv[4]) 
  conv7 = decode(conv6, conv3, dropout, conv[3]) 
  conv8 = decode(conv7, conv2, dropout, conv[2]) 
  conv9 = decode(conv8, conv1, dropout, conv[1]) 
  # output
  result = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(conv9)

  return result

if __name__ == '__main__': 

    input = Input(shape = (256, 256, 3))
    output = UNet(input, dropout = 0.1)

    model = Model(inputs = input, outputs = output)
    model.summary()

