'''
This file will be used to find the initial learning rate 
to be used by model for training
'''

# import all required libraries
import keras
from .model import build_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, SGD
import numpy as np
import os
from pathlib import Path
from .find_lr import find_lr

# path to dataset
path = Path('data/food-101/images')

# create a generator to feed images for training
# batch size
batch_size = 64

# ImageData Generator with various transformations like
# zoom, rescale, horizontal flip, etc 
# and splitting 20% of data into validation set
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             preprocessing_function=preprocess_input,
                             validation_split=0.2,
                             horizontal_flip=True)

# train generator
train_generator = datagen.flow_from_directory(path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              subset='training',
                                              class_mode='categorical')
# validation generator
val_generator = datagen.flow_from_directory(path,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            subset='validation',
                                            class_mode='categorical')

# create a base model of xception architecture
model = build_model('xception')

# show each layer in model and whether it is trainable or not
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

# create adam optimizer using amsgrad = True
AdamW = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=AdamW, metrics=['acc', 'top_k_categorical_accuracy'])

# find the best lr to train the model
logs,losses = find_lr(model, train_generator, batch_size)

# plot the lr and smoothed losses
plt.plot(logs[10:-5],losses[10:-5])
plt.show()