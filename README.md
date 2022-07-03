# OCT-A Machine Learning Quality Control

@author Rahul Dhodapkar <rahul.dhodapkar@yale.edu> [ORCID: 0000-0002-2014-7515
](https://orcid.org/0000-0002-2014-7515)
@version 2022.07.03

This project contains code to prototype the automated quality control of 
en-face images for optical coherence tomography-angiography (OCT-A).

## Plug and Play
To use the pre-trained high quality and low quality models, simply load the `fastai` models
from their respective pickle files into a python environment.

For a sample bitmap image at `./testimg.bmp`, one could run:

    #####
    # load libraries
    #
    from fastai.vision.all import PILImage
    from fastai.learner import load_learner
    #####
    # load models
    #
    low_quality_model = load_learner('./calc/fastai/hisens_model.pkl')
    high_quality_model = load_learner('./calc/fastai/hispec_model.pkl')
    #####
    # load image
    #
    img = PILImage.create('./testimg.bmp')
    #####
    # run loaded models
    #
    lq_is_valid, _, lq_probs = low_quality_model.predict(img)
    hq_is_valid, _, hq_probs = high_quality_model.predict(img)

The relative confidence of each model is encoded in the returned objects from the `.predict()` call.

All data and model files are available upon request from the authors *jay (dot) wang (at) yale (dot) edu* or *rahul (dot) dhodapkar (at) yale (dot) edu*.
