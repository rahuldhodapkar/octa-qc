#!/usr/bin/env python
#
##VisualizeLearning.py
#
# Visualize the learning process using Shapley values
# and/or class activation maps (CAM)
#

import matplotlib.pyplot as plt

# fastai imports
from fastai.vision.all import PILImage, Path, ImageDataLoaders, Resize
from fastai.learner import load_learner
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import torch
import torchvision

# utility imports
import numpy as np
import pandas as pd
import os
import re
import random
from tqdm import tqdm

# grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

######################################################
## GENERATE OUTPUT STRUCTURE
######################################################

if not os.path.exists('fig/fastai/viz'):
    os.makedirs('fig/fastai/viz')

if not os.path.exists('fig/fastai/viz_ext'):
    os.makedirs('fig/fastai/viz_ext')

######################################################
## Read In Models
######################################################

hisens_model = load_learner('./calc/fastai/hisens_model.pkl')
hisens_model_cnn = load_learner('./calc/fastai/hisens_model_alexnet.pkl')

hispec_model = load_learner('./calc/fastai/hispec_model.pkl')
hispec_model_cnn = load_learner('./calc/fastai/hispec_model_alexnet.pkl')

models_and_targets = {
    'hisens_model': hisens_model.model,
    'hisens_model_cnn': hisens_model_cnn.model,
    'hispec_model': hispec_model.model,
    'hispec_model_cnn': hispec_model_cnn.model
}

######################################################
##
######################################################

def visualize_first_layer(learn):
    conv1 = list(learn.model.children())[0][0]
    weights = conv1.weight.data.cpu().numpy()
    weights_shape = weights.shape
    weights = minmax_scale(weights.ravel()).reshape(weights_shape)
    fig, axes = plt.subplots(8, 8, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.rollaxis(weights[i], 0, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

#visualize_first_layer(hisens_model)

######################################################
## Read Data
######################################################

merged_ratings = pd.read_csv('./calc/merged_ratings.csv')
merged_ratings['Gradable.sum'] = merged_ratings['Gradable.r1'] + merged_ratings['Gradable.r2']

merged_ratings['hispec'] = merged_ratings['Gradable.sum'] >= 4
merged_ratings['hisens'] = merged_ratings['Gradable.sum'] >= 2

ids = []
labels_hispec = []
labels_hisens = []
labels_raw = []
fns = []
for i in range(len(merged_ratings)):
    ids += [merged_ratings['ID'][i]] * 4
    labels_hispec += [str(merged_ratings['hispec'][i])] * 4
    labels_hisens += [str(merged_ratings['hisens'][i])] * 4
    labels_raw += [merged_ratings['Gradable.sum'][i]] * 4
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
## Generate SHAP Image
######################################################
#
# shap only works with raw pytorch so will need to modify
# the options here a bit.
#

labels = labels_raw

X_train, X_val, y_train, y_val = train_test_split(
    fns, labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        super(LabeledImageDataset, self).__init__()
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torchvision.transforms.ToTensor()(PILImage.create(img))
        return (
            img_tensor,
            self.labels[idx])

torch.manual_seed(777)
test_loader = torch.utils.data.DataLoader(
    LabeledImageDataset(X_test, y_test),
    batch_size=32, shuffle=True)

batch = next(iter(test_loader))
images, targets = batch

# find images for each of the 5 possible gradability states
repimg_0 = images[np.min(np.where(targets == 0)):
                  np.min(np.where(targets == 0))+1]
repimg_1 = images[np.min(np.where(targets == 1)):
                  np.min(np.where(targets == 1))+1]
repimg_2 = images[np.min(np.where(targets == 2)):
                  np.min(np.where(targets == 2))+1]
repimg_3 = images[np.min(np.where(targets == 3)):
                  np.min(np.where(targets == 3))+1]
repimg_4 = images[np.min(np.where(targets == 4)):
                  np.min(np.where(targets == 4))+1]

images_to_process = [
    repimg_0, repimg_1, repimg_2, repimg_3, repimg_4
]

###########################################################################
## Test Images GradCAM
###########################################################################

raw_image_fn = './fig/fastai/viz/model-{}_quality-{}_raw.bmp'
cam_image_fn = './fig/fastai/viz/model-{}_quality-{}_target-{}_cam.bmp'
integrated_image_fn = './fig/fastai/viz/model-{}_quality-{}_target-{}_integrated_cam.bmp'
filtered_image_fn = './fig/fastai/viz/model-{}_quality-{}_target-{}_filtered_cam.png'

for model_name, model in models_and_targets.items():
    target_layers = [model[0][0]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    print("===== Processing Model: {} =====".format(model_name))
    for i in tqdm(range(len(images_to_process))):
        input_tensor = images_to_process[i]
        target_names = {
            'low_quality': ClassifierOutputTarget(0),
            'high_quality': ClassifierOutputTarget(1)
        }
        rgb_img = np.float32(input_tensor[0, :].detach().numpy())
        rgb_img = np.moveaxis(rgb_img, 0, -1)
        Image.fromarray(np.uint8(rgb_img * 255)).save(
            raw_image_fn.format(
                model_name, str(i)
            ))
        for output, target in target_names.items():
            grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
            grayscale_cam_img = grayscale_cam[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_img, use_rgb=True)
            Image.fromarray(visualization).save(integrated_image_fn.format(
                model_name, str(i), output
            ))
            Image.fromarray(np.uint8(grayscale_cam_img * 255), 'L').save(cam_image_fn.format(
                model_name, str(i), output
            ))
            filtered_image = Image.composite(
                image1=Image.fromarray(np.uint8(rgb_img * 255)),
                image2=Image.fromarray(np.uint8(np.zeros((1024, 1024))), 'L'),
                mask=Image.fromarray(np.uint8(grayscale_cam_img * 255), 'L')
            )
            filtered_image.save(filtered_image_fn.format(
                model_name, str(i), output
            ))


###########################################################################
## 6x6 Images GradCAM
###########################################################################

ratings = pd.read_csv('./data/ext/6x6_Superficial/quality_ratings/rater1.csv')
rating2 = pd.read_csv('./data/ext/6x6_Superficial/quality_ratings/rater2.csv')

ratings['Rater1Gradable'] = ratings['Gradable']
ratings['Rater2Gradable'] = rating2['Gradable']
ratings['Gradable'] = ratings['Rater1Gradable'] + ratings['Rater2Gradable']

ratings['File'] = ["./data/ext/6x6_Superficial/images/{}.bmp".format(x) for x in ratings['ID']]

torch.manual_seed(777)
diffsize_loader = torch.utils.data.DataLoader(
    LabeledImageDataset(ratings['File'], ratings['Gradable']),
    batch_size=32, shuffle=True)

batch = next(iter(diffsize_loader))
images, targets = batch

# find images for each of the 5 possible gradability states
repimg_0 = images[np.min(np.where(targets == 0)):
                  np.min(np.where(targets == 0))+1]
repimg_1 = images[np.min(np.where(targets == 1)):
                  np.min(np.where(targets == 1))+1]
repimg_2 = images[np.min(np.where(targets == 2)):
                  np.min(np.where(targets == 2))+1]
repimg_3 = images[np.min(np.where(targets == 3)):
                  np.min(np.where(targets == 3))+1]
repimg_4 = images[np.min(np.where(targets == 4)):
                  np.min(np.where(targets == 4))+1]

images_to_process = [
    repimg_0, repimg_1, repimg_2, repimg_3, repimg_4
]

#########
## Run the actual external CAM models.
#########

raw_image_fn = './fig/fastai/viz_ext/model-{}_quality-{}_raw.bmp'
cam_image_fn = './fig/fastai/viz_ext/model-{}_quality-{}_target-{}_cam.bmp'
integrated_image_fn = './fig/fastai/viz_ext/model-{}_quality-{}_target-{}_integrated_cam.bmp'
filtered_image_fn = './fig/fastai/viz_ext/model-{}_quality-{}_target-{}_filtered_cam.png'

for model_name, model in models_and_targets.items():
    target_layers = [model[0][0]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    print("===== Processing Model: {} =====".format(model_name))
    for i in tqdm(range(len(images_to_process))):
        input_tensor = images_to_process[i]
        target_names = {
            'low_quality': ClassifierOutputTarget(0),
            'high_quality': ClassifierOutputTarget(1)
        }
        rgb_img = np.float32(input_tensor[0, :].detach().numpy())
        rgb_img = np.moveaxis(rgb_img, 0, -1)
        Image.fromarray(np.uint8(rgb_img * 255)).save(
            raw_image_fn.format(
                model_name, str(i)
            ))
        for output, target in target_names.items():
            grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
            grayscale_cam_img = grayscale_cam[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_img, use_rgb=True)
            Image.fromarray(visualization).save(integrated_image_fn.format(
                model_name, str(i), output
            ))
            Image.fromarray(np.uint8(grayscale_cam_img * 255), 'L').save(cam_image_fn.format(
                model_name, str(i), output
            ))
            filtered_image = Image.composite(
                image1=Image.fromarray(np.uint8(rgb_img * 255)),
                image2=Image.fromarray(np.uint8(np.zeros((1024, 1024))), 'L'),
                mask=Image.fromarray(np.uint8(grayscale_cam_img * 255), 'L')
            )
            filtered_image.save(filtered_image_fn.format(
                model_name, str(i), output
            ))

print('All done!')
