#!/usr/bin/env python
## RunExternalDatasets.py
# 
# Run code on external datasets to benchmark performance for additional
# validation of robustness.
#

# fastai imports
from fastai.vision.all import PILImage
from fastai.learner import load_learner

# utility imports
import numpy as np
import pandas as pd
import os
import re
import random

######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('calc/fastai/resnet'):
    os.makedirs('calc/fastai/resnet')

if not os.path.exists('calc/fastai/resnet/validation'):
    os.makedirs('calc/fastai/resnet/validation')

######################################################
## Read In Models
######################################################

hisens_model = load_learner('./calc/fastai/hisens_model.pkl')
hisens_model_cnn = load_learner('./calc/fastai/hisens_model_alexnet.pkl')

hispec_model = load_learner('./calc/fastai/hispec_model.pkl')
hispec_model_cnn = load_learner('./calc/fastai/hispec_model_alexnet.pkl')

######################################################
## Read Data
######################################################

ratings = pd.read_csv('./data/ext/6x6_Superficial/quality_ratings/rater1.csv')
rating2 = pd.read_csv('./data/ext/6x6_Superficial/quality_ratings/rater2.csv')

ratings['Rater1Gradable'] = ratings['Gradable']
ratings['Rater2Gradable'] = rating2['Gradable']
ratings['Gradable'] = ratings['Rater1Gradable'] + ratings['Rater2Gradable']

ratings['File'] = ["./data/ext/6x6_Superficial/images/{}.bmp".format(x) for x in ratings['ID']]

def PredictImageQuality(model, metadata):
    truth = []
    pred = []
    lo_qual_prob = []
    hi_qual_prob = []
    for i in range(len(metadata['File'])):
        img = PILImage.create(metadata['File'][i])
        is_valid, _, probs = model.predict(img)
        truth.append(metadata['Gradable'][i])
        pred.append(is_valid)
        lo_qual_prob.append(probs[0].item())
        hi_qual_prob.append(probs[1].item())
    predictions = pd.DataFrame({
        'predictions': hi_qual_prob,
        'truth': truth,
        'ID': metadata['ID']
    })
    return(predictions)

p_hisens = PredictImageQuality(hisens_model, ratings)
np.corrcoef(p_hisens['predictions'], p_hisens['truth'])
np.corrcoef(p_hisens['predictions'], p_hisens['truth'] >= 3)

p_hisens_cnn = PredictImageQuality(hisens_model_cnn, ratings)
np.corrcoef(p_hisens_cnn['predictions'], p_hisens_cnn['truth'])
np.corrcoef(p_hisens_cnn['predictions'], p_hisens_cnn['truth'] >= 3)

p_hispec = PredictImageQuality(hispec_model, ratings)
np.corrcoef(p_hispec['predictions'], p_hispec['truth'])
np.corrcoef(p_hispec['predictions'], p_hispec['truth'] >= 1)

p_hispec_cnn = PredictImageQuality(hispec_model_cnn, ratings)
np.corrcoef(p_hispec_cnn['predictions'], p_hispec_cnn['truth'])
np.corrcoef(p_hispec_cnn['predictions'], p_hispec_cnn['truth'] >= 1)

######################################################
## Save Data
######################################################

p_hisens.to_csv(
    "./calc/fastai/resnet/validation/hisens_6x6_superficial.csv", index=False)
p_hisens_cnn.to_csv(
    "./calc/fastai/resnet/validation/hisens_cnn_6x6_superficial.csv", index=False)

p_hispec.to_csv(
    "./calc/fastai/resnet/validation/hispec_6x6_superficial.csv", index=False)
p_hispec_cnn.to_csv(
    "./calc/fastai/resnet/validation/hispec_cnn_6x6_superficial.csv", index=False)

