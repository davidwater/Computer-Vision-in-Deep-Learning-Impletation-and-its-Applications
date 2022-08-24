from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Concatenate, add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.resnet_v2 import ResNet50V2
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import get_file
import random
import os
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime, os

# enable and test GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# data preprocessing
from keras.preprocessing.image import ImageDataGenerator
image_width = 224
image_height = 224
image_size = (image_width, image_height)

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        'mytrain',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=8,
        class_mode='categorical')

validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(
        'myvalid',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=8,
        class_mode='categorical')

# build structure
net = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(image_width, image_height, 3))
x = net.output
x = AveragePooling2D((7, 7), name='avg_pool')(x)
x = Flatten()(x)

x = Dropout(0.5)(x)

output_layer = Dense(2, activation='softmax', name='softmax')(x)

base_model = Model(inputs=net.input, outputs=output_layer)

for layer in base_model.layers[:2]:
    layer.trainable = False
for layer in base_model.layers[2:]:
    layer.trainable = True

lr = 0.00001
base_model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
base_model.summary()

# train
from keras.callbacks import TensorBoard

# Load the TensorBoard notebook extension

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
base_model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//8,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//8,
        epochs=25,
        callbacks=[tensorboard_callback])

base_model.save(ResNet50_final.h5)