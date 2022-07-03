#!/usr/bin/env python
#
## UnsupervisedAnalysis.py
# Perform basic unsupervised analysis of data.
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

# sklearn imports
from sklearn.preprocessing import StandardScaler

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
import umap

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
    nx, ny = tmpimg.shape
    img.append(tmpimg.reshape(nx*ny))

img = np.array(img)

######################################################
## CAST DATA TYPES
######################################################

X = img
y = np.ravel([float(x) for x in np.ravel(merged_ratings[['Gradable.sum']])])

######################################################
## CREATE UMAP PROJECTION
######################################################

reducer = umap.UMAP(random_state=42)

scaled_X = StandardScaler().fit_transform(X)
embedding = reducer.fit_transform(scaled_X)

umap_df = pd.DataFrame({
    'ID' : merged_ratings['ID'].tolist(),
    'UMAP1' : embedding[:,0],
    'UMAP2' : embedding[:,1]})

umap_df.to_csv('./calc/umap_embeddings.csv', index=False)