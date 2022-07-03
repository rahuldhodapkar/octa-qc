#!/usr/bin/env python
#
## ResNetModel.py
# Create a neural network based on ResNet using Fast.ai framework
# Tested on Yale Farnam
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

# fastai imports
from fastai.vision.all import *

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# utility imports
import numpy as np
import pandas as pd
import os
import re
import random


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)


######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('calc/fastai/resnet'):
    os.makedirs('calc/fastai/resnet')

######################################################
## DEFINE DATA SPLITS
######################################################

merged_ratings = pd.read_csv('./calc/merged_ratings.csv')
merged_ratings['Gradable.sum'] = merged_ratings['Gradable.r1'] + merged_ratings['Gradable.r2']

merged_ratings['hispec'] = merged_ratings['Gradable.sum'] >= 4
merged_ratings['hisens'] = merged_ratings['Gradable.sum'] >= 2

ids = []
labels_hispec = []
labels_hisens = []
fns = []
for i in range(len(merged_ratings)):
    ids += [merged_ratings['ID'][i]] * 4
    labels_hispec += [str(merged_ratings['hispec'][i])] * 4
    labels_hisens += [str(merged_ratings['hisens'][i])] * 4
    fns.append(Path(
        './data/images/superficial_{}.bmp'.format(
        merged_ratings['ID'][i]
    )))
    fns.append(Path(
        './calc/transform/vertflip/superficial_{}.bmp'.format(
        merged_ratings['ID'][i]
    )))
    fns.append(Path(
        './calc/transform/horzflop/superficial_{}.bmp'.format(
        merged_ratings['ID'][i]
    )))
    fns.append(Path(
        './calc/transform/flipflop/superficial_{}.bmp'.format(
        merged_ratings['ID'][i]
    )))


######################################################
## DEFINE DATA SPLITS
######################################################


def TrainTransferLearner(
    fns, labels, outfile, loss_outfile,
    model=resnet34, lr=0.04, ps=0.19, wd=0.04,
    n_epochs=50, n_freeze_epochs=10):
    #
    '''
    n_0 = np.sum([x == 'True' for x in labels])
    n_1 = np.sum([x == 'False' for x in labels])
    w_0 = (n_0 + n_1) / n_0
    w_1 = (n_0 + n_1) / n_1
    class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    loss_func = CrossEntropyLossFlat(weight=class_weights)
    '''
    loss_func = CrossEntropyLossFlat()
    #
    X_train, X_val, y_train, y_val = train_test_split(
        fns, labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    path = Path('./data/images')
    dls = ImageDataLoaders.from_lists(
        path=path,
        fnames=X_train, labels=y_train,
        val_pct=0.2, seed=42, 
        item_tfms=Resize(224),
        bs=12)
    learn = vision_learner(dls, model, metrics=accuracy,
        lr=lr, ps=ps, wd=wd,
        loss_func=loss_func)
    #
    # Perform "fine_tune" directly, so can capture full training history
    #
    base_lr=2e-3
    lr_mult=100
    pct_start=0.3
    div=5.0
    set_seed(99, True)
    learn.freeze()
    learn.fit_one_cycle(n_freeze_epochs, slice(base_lr), pct_start=0.99)
    loss_df_frozen = pd.DataFrame(learn.recorder.values, 
        columns=['train_loss', 'val_loss', 'accuracy'], dtype=float)
    base_lr /= 2
    learn.unfreeze()
    learn.fit_one_cycle(n_epochs, slice(base_lr/lr_mult, base_lr),
        pct_start=pct_start, div=div)
    loss_df_unfrozen = pd.DataFrame(learn.recorder.values, 
        columns=['train_loss', 'val_loss', 'accuracy'], dtype=float)
    loss_df = pd.concat((loss_df_frozen, loss_df_unfrozen), axis=0)
    # quantify model performance
    truth = []
    pred = []
    lo_qual_prob = []
    hi_qual_prob = []
    for i in range(len(y_val)):
        img = PILImage.create(X_val[i])
        is_valid, _, probs = learn.predict(img)
        truth.append(y_val.iloc[i])
        pred.append(is_valid)
        lo_qual_prob.append(probs[0].item())
        hi_qual_prob.append(probs[1].item())
    predictions = pd.DataFrame({
        'predictions': hi_qual_prob,
        'truth': y_val,
        'ID': y_val.index.tolist()
    })
    predictions.to_csv(outfile, index=False)
    loss_df.to_csv(loss_outfile, index=False)
    #
    print("AUC: {}".format(roc_auc_score(predictions['truth'],
                                         predictions['predictions'])))
    #
    return(learn)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


learn_hisens = TrainTransferLearner(fns=fns, 
                     labels=pd.Series(
                        labels_hisens,
                        index=ids), 
                     outfile='./calc/fastai/resnet/hisens_predictions.csv',
                     loss_outfile='./calc/fastai/resnet/hisens_history.csv',
                     lr=0.0470, ps=0.354, wd=0.188,
                     n_epochs=40, n_freeze_epochs=10,
                     model=resnet152)
# AUC = 0.9831025951466021

learn_hisens_alexnet = TrainTransferLearner(fns=fns, 
                     labels=pd.Series(
                        labels_hisens,
                        index=ids), 
                     outfile='./calc/fastai/resnet/alexnet_hisens_predictions.csv',
                     loss_outfile='./calc/fastai/resnet/alexnet_hisens_history.csv',
                     lr=0.0955, ps=0.498, wd=0.188,
                     n_epochs=40, n_freeze_epochs=10,
                     model=alexnet)
# AUC = 0.9530452176659473

learn_hispec = TrainTransferLearner(fns=fns, 
                     labels=pd.Series(
                        labels_hispec,
                        index=ids), 
                     outfile='./calc/fastai/resnet/hispec_predictions.csv',
                     loss_outfile='./calc/fastai/resnet/hispec_history.csv',
                     lr=0.0417, ps=0.532, wd=0.000445,
                     n_epochs=40, n_freeze_epochs=10,
                     model=resnet152)
# AUC = 0.9734228971962617

learn_hispec_alexnet = TrainTransferLearner(fns=fns, 
                     labels=pd.Series(
                        labels_hispec,
                        index=ids), 
                     outfile='./calc/fastai/resnet/alexnet_hispec_predictions.csv',
                     loss_outfile='./calc/fastai/resnet/alexnet_hispec_history.csv',
                     lr=0.0323, ps=0.558, wd=0.343,
                     n_epochs=40, n_freeze_epochs=10,
                     model=alexnet)
# AUC = 0.944436331775701


# Save fastai models for later usage.
learn_hisens.export(os.path.abspath('./calc/fastai/hisens_model.pkl'))
learn_hisens_alexnet.export(os.path.abspath('./calc/fastai/hisens_model_alexnet.pkl'))

learn_hispec.export(os.path.abspath('./calc/fastai/hispec_model.pkl'))
learn_hispec_alexnet.export(os.path.abspath('./calc/fastai/hispec_model_alexnet.pkl'))

print("All done!")
