#!/usr/bin/env Rscript
#
## CNNRegressionHighSensitivity.R
# benchmark convolutional neural network regression against Gradable.sum
# with a target of "high sensitivity"
#
# Tested on | R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out"
#

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(shades)
library(reshape)
library(cowplot)
library(stringr)
library(hashmap)
library(pROC)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/fastai"), dir.create("./fig/fastai"), FALSE)
ifelse(!dir.exists("./fig/fastai/resnet"), dir.create("./fig/fastai/resnet"), FALSE
