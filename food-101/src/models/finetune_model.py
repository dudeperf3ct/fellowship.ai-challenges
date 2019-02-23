'''
This file is used to finetune a trained model and 
save its weights used for predictions 
'''


import keras
from keras_contrib.callbacks.cyclical_learning_rate import CyclicLR
from .model import finetune_model, build_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import random

# name of the model to use
name = 'xception'

print ('Creating {} model..'.format(name))
model = build_model(name)
model.load_weights(name+'_model-224_weights.h5')
print ('Model weights loaded...')

# create xception model
model = finetune_model(model, name)

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

# create adam optimizer using amsgrad = True using the lr obtained from find_lr_model.py
AdamW = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True)

# compile the model
model.compile(loss='categorical_crossentropy',
	optimizer=AdamW, 
	metrics=['acc', 'top_k_categorical_accuracy'])

# create a checkpoint to save model
mc = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                     monitor='val_acc', verbose=1, save_best_only=True)

# to avoid overfitting use earlystopping
es = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=3, min_lr=0.001, verbose=1)

# clr = CyclicLR(base_lr=1e-3, max_lr=1e-2, step_size=2500., mode='triangular1')
# clr = CyclicLR(base_lr=1e-3, max_lr=1e-2, step_size=2500., mode='triangular2')
# clr = CyclicLR(base_lr=0.001, max_lr=0.007, step_size=2500., mode='exp_range', gamma=0.99994)

# lr_manager = OneCycleLR(num_samples=train_generator.samples, 
#                         num_epochs=8, 
#                         batch_size=batch_size, 
#                         max_lr=0.1,
#                         end_percentage=0.1,
#                         maximum_momentum=0.95, 
#                         minimum_momentum=0.85,
#                         verbose=False)

# train the model for 15 epochs
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples // batch_size,
                              epochs=15,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // batch_size,
                              callbacks=[mc, es])


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# save the model weights
model.save_weights(name+'_finetune_model-224_weights.h5')
print ('Model weights saved...')