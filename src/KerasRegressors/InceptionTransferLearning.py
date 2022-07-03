#!/usr/bin/env python
#
## InceptionTransferLearning.py
# Create a convolutional neural network with transfer learning from the
# InceptionV3 model 
#
# Tested on Yale Farnam
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.06.08
#

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import datasets,layers,models

# keras imports
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

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

input_tensor = Input(shape=(1024, 1024, 3))
model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
