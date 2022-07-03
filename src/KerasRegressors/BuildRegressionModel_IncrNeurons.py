#!/usr/bin/env python
#
## BuildRegressionModel.py
# Create a convolutional neural network regressor to predict the
# gradability of an image based on grouped quality scoring.
#
# Tested on Yale Farnam
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import datasets,layers,models

# keras imports
from tensorflow import keras

# sklearn imports
import sklearn
from sklearn.model_selection import train_test_split

# skimage imports
import skimage
from skimage.io import imread

# utility imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import random

######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('calc/regression/cnn'):
    os.makedirs('calc/regression/cnn')

######################################################
## DEFINE CONSTANTS
######################################################

IMG_PIXELS_X = 1024
IMG_PIXELS_Y = 1024
IMG_CHANNELS = 1

OUTPUT_CHANNELS = 2

######################################################
## READ DATA
######################################################

merged_ratings = pd.read_csv('./calc/merged_ratings.csv')
merged_ratings['Gradable.sum'] = merged_ratings['Gradable.r1'] + merged_ratings['Gradable.r2']

img_fns = os.listdir('./data/images')
img = []
for fn in merged_ratings['ID'].tolist():
    tmpimg = imread(
        './data/images/superficial_{}.bmp'.format(fn),
        as_gray=True
    )
    img.append(np.reshape(tmpimg, (1024,1024,1)))

img = np.array(img)

######################################################
## SPLIT DATA
######################################################

X = img
y = pd.Series(np.ravel([float(x) for x in np.ravel(merged_ratings[['Gradable.sum']])]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

######################################################
## Define Model
######################################################
#
# See: https://www.tensorflow.org/tutorials/images/cnn
#      https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
#

model = models.Sequential()

# Simple model
model.add(layers.Conv2D(16, (3, 3), activation='relu',
    input_shape=(IMG_PIXELS_X, IMG_PIXELS_Y, IMG_CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))

######################################################
## FIT / TEST
######################################################

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_absolute_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size=10,
                    epochs=30)

training_history = pd.DataFrame({
    'loss' : history.history['loss'],
    'root_mean_squared_error' : history.history['root_mean_squared_error'],
    'val_loss' : history.history['val_loss'],
    'val_root_mean_squared_error' : history.history['val_root_mean_squared_error']
})

training_history.to_csv('./calc/regression/cnn/training_history_incrneuron.csv', index=False)

######################################################
## TEST
######################################################

test_results = model.evaluate(X_test, y_test)
prediction_vals = model.predict(X_test).flatten()

predictions = pd.DataFrame({
    'predictions': prediction_vals,
    'truth': y_test,
    'ID': y_test.index.tolist()
})

predictions.to_csv('./calc/regression/cnn/predictions_incrneuron.csv', index=False)

np.corrcoef(prediction_vals, y_test)

print('All done.')