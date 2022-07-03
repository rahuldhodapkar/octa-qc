#!/usr/bin/env python
#
## OptimizeHyperparameters.py
# Optimize hyperparameters for OCTA image quality analysis
# Tested on Yale Farnam
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

# fastai imports
from fastai.vision.all import *

# hyperparam optimization
from bayes_opt import BayesianOptimization

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# utility imports
import numpy as np
import pandas as pd
import os
import re
import random

######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('calc/fastai/hyperparam'):
    os.makedirs('calc/fastai/hyperparam')

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
## Tune Hyperparams
######################################################


def OptimizeHyperparams(model=resnet34,
        hps={
            'base_lr': (2e-5, 2e-1),
            'lr_mult': (50.0, 200.0),
            'div': (1.0, 40.0),
            'pct_start': (0.1, 0.6)
        },
        type='hisens'):
    if (type == 'hisens'):
        labels=pd.Series(labels_hisens, index=ids)
    elif (type == 'hispec'):
        labels=pd.Series(labels_hispec, index=ids)
    X_train, X_val, y_train, y_val = train_test_split(
        fns, labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    #
    # *** SET FOR HYPERPARAM OPTIM ***
    n_epochs=10
    n_freeze_epochs=1
    #
    #
    print(hps)
    def TrainWithParams(base_lr, lr_mult, div, pct_start):
        print(base_lr, lr_mult, div, pct_start)
        path = Path('./data/images')
        dls = ImageDataLoaders.from_lists(
            path=path,
            fnames=X_train, labels=y_train,
            val_pct=0.2, seed=42, 
            item_tfms=Resize(224),
            bs=12)
        learn = vision_learner(dls, model, metrics=accuracy)
        set_seed(99, True)
        learn.freeze()
        learn.fit_one_cycle(
            n_freeze_epochs,
            slice(base_lr),
            pct_start=0.99)
        base_lr /= 2
        learn.unfreeze()
        learn.fit_one_cycle(n_epochs, slice(base_lr/lr_mult, base_lr),
            pct_start=pct_start,
            div=div)
        # quantify model performance
        truth = []
        pred = []
        lo_qual_prob = []
        hi_qual_prob = []
        for i in range(len(y_test)):
            img = PILImage.create(X_test[i])
            is_valid, _, probs = learn.predict(img)
            truth.append(y_test.iloc[i])
            pred.append(is_valid)
            lo_qual_prob.append(probs[0].item())
            hi_qual_prob.append(probs[1].item())
        predictions = pd.DataFrame({
            'predictions': hi_qual_prob,
            'truth': y_test,
            'ID': y_test.index.tolist()
        })
        return(roc_auc_score(predictions['truth'], predictions['predictions']))
    #
    #
    #
    optim = BayesianOptimization(
        f = TrainWithParams, # our fit function
        pbounds = hps, # our hyper parameters to tune
        verbose = 2, # 1 prints out when a maximum is observed, 0 for silent
        random_state=1
    )
    optim.maximize(init_points=2, n_iter=10)
    print(optim.max)
    return(optim)



def OptimizeHyperparamsClassWeighted(model=resnet34,
        hps={
            'base_lr': (2e-5, 2e-1),
            'lr_mult': (50, 200),
            'wd': (4e-4, 0.4),
            'div': (1, 40),
            'pct_start': (0.1, 0.6),
        },
        type='hisens'):
    if (type == 'hisens'):
        labels=pd.Series(labels_hisens, index=ids)
    elif (type == 'hispec'):
        labels=pd.Series(labels_hispec, index=ids)
    #
    n_0 = np.sum([x == 'True' for x in labels])
    n_1 = np.sum([x == 'False' for x in labels])
    w_0 = (n_0 + n_1) / n_0
    w_1 = (n_0 + n_1) / n_1
    class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    loss_func = CrossEntropyLossFlat(weight=class_weights)
    #
    X_train, X_val, y_train, y_val = train_test_split(
        fns, labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    print(hps)
    def TrainWithParams(lr, wd, ps):
        print(lr, wd, ps)
        path = Path('./data/images')
        dls = ImageDataLoaders.from_lists(
            path=path,
            fnames=X_train, labels=y_train,
            val_pct=0.2, seed=42, 
            item_tfms=Resize(224),
            bs=12)
        learn = vision_learner(
            dls, model, metrics=accuracy,
            loss_func=loss_func,
            lr=float(lr), wd=float(wd), ps=float(ps))
        set_seed(99, True)
        learn.fine_tune(epochs=10, freeze_epochs=1)
        loss_df = pd.DataFrame(learn.recorder.values, 
            columns=['train_loss', 'val_loss', 'accuracy'], dtype=float)
        # quantify model performance
        truth = []
        pred = []
        lo_qual_prob = []
        hi_qual_prob = []
        for i in range(len(y_test)):
            img = PILImage.create(X_test[i])
            is_valid, _, probs = learn.predict(img)
            truth.append(y_test.iloc[i])
            pred.append(is_valid)
            lo_qual_prob.append(probs[0].item())
            hi_qual_prob.append(probs[1].item())
        predictions = pd.DataFrame({
            'predictions': hi_qual_prob,
            'truth': y_test,
            'ID': y_test.index.tolist()
        })
        return(roc_auc_score(predictions['truth'], predictions['predictions']))
    #
    #
    #
    optim = BayesianOptimization(
        f = TrainWithParams, # our fit function
        pbounds = hps, # our hyper parameters to tune
        verbose = 2, # 1 prints out when a maximum is observed, 0 for silent
        random_state=1
    )
    optim.maximize(init_points=2, n_iter=10)
    print(optim.max)
    return(optim)

######################################################
## Optimize Hyperparameters without Class Weighting
######################################################

hisens_optim = OptimizeHyperparams(model=resnet34,
                            type='hisens')
# {'target': 0.9698300870285952, 'params': {'base_lr': 0.029368243045606267, 'div': 4.601205195983114, 'lr_mult': 77.93903170665064, 'pct_start': 0.2727803635215239}}


hispec_optim = OptimizeHyperparams(model=resnet34,
                            type='hispec')
# {'target': 0.9506704980842912, 'params': {'base_lr': 0.08341606050042076, 'div': 29.092655244244167, 'lr_mult': 50.017156222601734, 'pct_start': 0.25116628631591986}}


hisens_optim_alexnet = OptimizeHyperparams(model=alexnet,
                            type='hisens')
# {'target': 0.9360132615002072, 'params': {'base_lr': 0.08341606050042076, 'div': 29.092655244244167, 'lr_mult': 50.017156222601734, 'pct_start': 0.25116628631591986}}

hispec_optim_alexnet = OptimizeHyperparams(model=alexnet,
                            type='hispec')
# {'target': 0.9137931034482758, 'params': {'base_lr': 0.004260830605052716, 'div': 32.50736834544213, 'lr_mult': 194.72833116547824, 'pct_start': 0.45103651362179775}}


######################################################
## Optimize Hyperparameters with Class Weighting
######################################################

hisens_optim_weighted = OptimizeHyperparamsClassWeighted(
                            model=resnet34,
                            hps={
                                    'lr': (1e-05, 1e-01),
                                    'wd': (4e-4, 0.4),
                                    'ps': (0.1, 0.7)
                            },
                            type='hisens')
# {'target': 0.9680895151263986, 'params': {'lr': 0.04702996660070071, 'ps': 0.3540956268190799, 'wd': 0.18897630305465052}}

hisens_optim_weighted_alexnet = OptimizeHyperparamsClassWeighted(
                            model=alexnet,
                            hps={
                                    'lr': (1e-05, 1e-01),
                                    'wd': (4e-4, 0.4),
                                    'ps': (0.1, 0.7)
                            },
                            type='hisens')
# {'target': 0.9368421052631579, 'params': {'lr': 0.09554500309836413, 'ps': 0.49844284950522855, 'wd': 0.18849996771570127}}

hispec_optim_weighted = OptimizeHyperparamsClassWeighted(model=resnet34,
                            hps={
                                    'lr': (1e-05, 1e-01),
                                    'wd': (4e-4, 0.4),
                                    'ps': (0.1, 0.7)
                            },
                            type='hispec')
# {'target': 0.9378591954022988, 'params': {'lr': 0.04170803025021038, 'ps': 0.5321946960652949, 'wd': 0.00044570417701101674}}

hispec_optim_weighted_alexnet = OptimizeHyperparamsClassWeighted(model=alexnet,
                            hps={
                                    'lr': (1e-05, 1e-01),
                                    'wd': (4e-4, 0.4),
                                    'ps': (0.1, 0.7)
                            },
                            type='hispec')
# {'target': 0.9147509578544061, 'params': {'lr': 0.032254051216090984, 'ps': 0.5581212692663862, 'wd': 0.34284188156449513}}


# as we did not observe any significant changes to optimization parameters or fitting performance
# when using weighted cross entropy as compared to vanilla cross-entropy, will go forward with
# unweighted cross entropy weighting.

