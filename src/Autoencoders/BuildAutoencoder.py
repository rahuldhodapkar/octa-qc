#!/usr/bin/env python
## BuildAutoencoder.py
#
# Build an Autoencoder using TensorFlow and Keras,
# exporting both the dense encoding for all images as well as
# UMAP projections from dense-layer representation. 
#

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import datasets,layers,models,losses

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

from imageio import imwrite

######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('calc/autoencoder'):
    os.makedirs('calc/autoencoder')

if not os.path.exists('calc/autoencoder/reconstructions'):
    os.makedirs('calc/autoencoder/reconstructions')

if not os.path.exists('calc/autoencoder/internalrep'):
    os.makedirs('calc/autoencoder/internalrep')

######################################################
## DEFINE CONSTANTS
######################################################

IMG_PIXELS_X = 1024
IMG_PIXELS_Y = 1024
IMG_CHANNELS = 1

TEST_TRAIN_FRACTION = 0.2

LATENT_DIM = 512

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
    X, y, test_size=TEST_TRAIN_FRACTION, random_state=42)

######################################################
## Define Model
######################################################
#
# See: https://www.tensorflow.org/tutorials/generative/autoencoder
#

latent_dim = LATENT_DIM
autoencoder = tf.keras.Sequential([
            layers.Input(shape=(IMG_PIXELS_X, IMG_PIXELS_Y, IMG_CHANNELS)),
            layers.Conv2D(4, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(4, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
            layers.Dense(512),
            layers.Reshape((4,4,32)),
            layers.Conv2DTranspose(32, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(16, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(16, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(8, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(8, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(4, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            layers.Conv2DTranspose(4, (3,3), activation='relu'),
            layers.UpSampling2D((2,2)),
            # extra layer needed due to rounding errors in output dimension
            layers.Conv2DTranspose(16, (5,5), activation='relu'),
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
        ])


autoencoder.compile(optimizer='adam', loss=losses.MeanAbsoluteError())


######################################################
## FIT / TEST
######################################################

autoencoder_history = autoencoder.fit(X_train, X_train,
                epochs=100,
                shuffle=True,
                batch_size=10,
                validation_data=(X_test, X_test))

######################################################
## WRITE HISTORY TO FILE
######################################################

training_history = pd.DataFrame({
    'loss' : autoencoder_history.history['loss'],
    'val_loss' : autoencoder_history.history['val_loss']
})

training_history.to_csv('./calc/autoencoder/training_history_autoencoder.csv', index=False)

######################################################
## WRITE RECONSTRUCTIONS TO FILE
######################################################

test_reconstructions = autoencoder.predict(X_test)
for i in range(len(y_test)):
    imwrite('calc/autoencoder/reconstructions/superficial_{}_test_recon.bmp'.format(
            y_test.index[i]
        ), test_reconstructions[i]*255)

train_reconstructions = autoencoder.predict(X_train)
for i in range(len(y_train)):
    imwrite('calc/autoencoder/reconstructions/superficial_{}_train_recon.bmp'.format(
            y_train.index[i]
        ), train_reconstructions[i]*255)

######################################################
## WRITE DENSE INTERNAL REPRESENTATIONS TO FILE
######################################################
# get values of dense internal layer
dense_output = autoencoder.get_layer('dense').output
m = keras.Model(inputs=autoencoder.input, outputs=dense_output)

test_dense = m.predict(X_test)
for i in range(len(y_test)):
    np.savetxt(
        'calc/autoencoder/internalrep/superficial_{}_test_dimred.csv'.format(
            y_test.index[i]
        ), test_dense[i], delimiter=',')


train_dense = m.predict(X_train)
for i in range(len(y_train)):
    np.savetxt(
        'calc/autoencoder/internalrep/superficial_{}_train_dimred.csv'.format(
            y_train.index[i]
        ), train_dense[i], delimiter=',')

######################################################
## Write MAE per image for outlier detection
######################################################

mae_gen = tf.keras.metrics.MeanAbsoluteError()

test_mae = []
for i in range(len(y_test)):
    mae_gen.update_state(test_reconstructions[i], X_test[i])
    mae = mae_gen.result().numpy()
    test_mae.append(mae)

train_mae = []
for i in range(len(y_train)):
    mae_gen.update_state(train_reconstructions[i], X_train[i])
    mae = mae_gen.result().numpy()
    train_mae.append(mae)

mae_df = pd.DataFrame({
    'ID': y_test.index.tolist() + y_train.index.tolist(),
    'MAE': test_mae + train_mae,
    'Group': ['test']*len(y_test) + ['train']*len(y_train)
})

mae_df.to_csv('calc/autoencoder/mae_per_image.csv', index=False)


