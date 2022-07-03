#!/usr/bin/env Rscript
#
## BenchmarkResnetHiSensitivity
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
library(pROC)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/fastai"), dir.create("./fig/fastai"), FALSE)
ifelse(!dir.exists("./fig/fastai/resnet"), dir.create("./fig/fastai/resnet"), FALSE)

#####################################################
## Load data
#####################################################

merged.ratings <- read.csv('./calc/merged_ratings.csv')
metadata <- read.csv('./data/metadata.csv')
merged.metadata <- merge(merged.ratings, metadata, by='ID')

predictions.hisens <- read.csv('./calc/fastai/resnet/hisens_predictions.csv')
predictions.cnn <- read.csv('./calc/fastai/resnet/alexnet_hisens_predictions.csv')

##############################################
## Generate plots
##############################################

##########
# SIGNAL STRENGTH
##########

id2truth <- merged.metadata$Gradable.r1 + merged.metadata$Gradable.r2
names(id2truth) <- merged.metadata$ID

predictions.signalstrength <- data.frame(
    predictions=merged.metadata$SignalStrength,
    truth=merged.metadata$Gradable.r1 + merged.metadata$Gradable.r2,
    ID=merged.metadata$ID,
    model='Signal Strength'
) %>% na.omit()

cor.data.signalstrength <- cor.test(
                     x = predictions.signalstrength$predictions, 
                     y = predictions.signalstrength$truth,
                     method = 'pearson')

##########
# CNN
##########
# now look at predictions

predictions.cnn$truth <- id2truth[predictions.cnn$ID + 1]
predictions.cnn$model <- 'Convolutional Neural Network'

cor.data.cnn <- cor.test(
                     x = predictions.cnn$predictions, 
                     y = predictions.cnn$truth,
                     method = 'pearson')

##########
# RESNET
##########
predictions.hisens$truth <- id2truth[predictions.hisens$ID + 1]
predictions.hisens$model <- 'ResNet Transfer Learning'
cor.data.resnet <- cor.test(
                     x = predictions.hisens$predictions, 
                     y = id2truth[predictions.hisens$ID + 1],
                     method = 'pearson')

##########
# Merge Datsets
##########
merged.predictions <- rbind(predictions.signalstrength,
                            predictions.cnn,
                            predictions.hisens)

merged.predictions$model <- factor(merged.predictions$model,
                                    levels=c(
                                        'Signal Strength',
                                        'Convolutional Neural Network',
                                        'ResNet Transfer Learning'
                                    ))

##################################################################
## Examine PPV with cutoff points.
##################################################################

# now plot ROC curves for models in grid facet format as below:
#
#   |  TP  |  FN  |
#   |  FP  |  TN  |
#

# binarization cutoff from human classification data.
HUMAN.GRADABLE.CUTOFF <- 2
STEP.WIDTH <- 0.01

test.threshold.iter <- seq(
    min(merged.predictions$predictions),
    max(merged.predictions$predictions) + STEP.WIDTH*10,
    by=STEP.WIDTH)

roc.data <- NULL
for (t in test.threshold.iter) {
    x <- merged.predictions %>%
            group_by(model) %>%
            summarise(
                tp = sum(truth >= HUMAN.GRADABLE.CUTOFF
                         & predictions >= t), 
                fn = sum(truth >= HUMAN.GRADABLE.CUTOFF
                         & predictions < t),
                fp = sum(truth < HUMAN.GRADABLE.CUTOFF
                         & predictions >= t),
                tn = sum(truth < HUMAN.GRADABLE.CUTOFF
                         & predictions < t)
            ) %>%
        as.data.frame()
    x$threshold <- t

    if (is.null(roc.data)) {
        roc.data <- x
    } else {
        roc.data <- rbind(roc.data, x)
    }
}


# calculate AUC for each model
m <- predictions.signalstrength
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label.signalstrength <- sprintf(
    "Signal Strength: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)
pROC_obj.signalstrength <- pROC_obj

m <- predictions.hisens
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label.hisens <- sprintf(
    "ResNet Transfer Learning: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)
pROC_obj.hisens <- pROC_obj

roc.test(pROC_obj.hisens, pROC_obj.signalstrength)

m <- predictions.cnn
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label.cnn <- sprintf(
    "Convolutional Neural Network: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)

roc.test(pROC_obj.hisens, pROC_obj)

auc.caption = paste(
    auc.label.signalstrength,
    auc.label.cnn,
    auc.label.hisens,
    sep = '\n'
)

# now make plots of ROC space.

ggplot(roc.data, aes(x = fp / (fp + tn), 
                     y = tp / (tp + fn), 
                     color = model)) +
    geom_path(alpha=0.5, size=1.5) + 
    geom_abline(slope = 1, intercept = 0, linetype='dashed') +
    scale_color_manual(name='', values=rev(brewer.pal(4, 'Set1'))) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    labs(caption=auc.caption) +
    theme_cowplot() + background_grid() +
    theme(
        legend.position='top',
        legend.justification='center'
    )

ggsave('./fig/fastai/resnet/multimodel_roc_gradability_cutoff_2.png', height=8, width=8)


# create two-line plot of ROC space
roc.data.2l <- subset(roc.data, model %in% c('Signal Strength', 'Convolutional Neural Network', 'ResNet Transfer Learning'))
auc.caption.2l <- paste(
    auc.label.signalstrength,
    auc.label.cnn,
    auc.label.hisens,
    sep = '\n'
)

ggplot(roc.data.2l, aes(x = fp / (fp + tn), 
                     y = tp / (tp + fn), 
                     color = model)) +
    geom_path(alpha=0.5, size=1.5) + 
    geom_abline(slope = 1, intercept = 0, linetype='dashed') +
    scale_color_manual(name='', values=rev(brewer.pal(4, 'Set1'))) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    labs(caption=auc.caption.2l) +
    theme_cowplot() + background_grid() +
    theme(
        legend.position='top',
        legend.justification='center'
    )

ggsave('./fig/fastai/resnet/roc_gradability_hisens.png', height=8, width=8)

#########
# Generate Accuracy Estimates
#########

roc.data$accuracy <- (roc.data$tp + roc.data$tn) / (roc.data$tp + roc.data$tn + roc.data$fp + roc.data$fn)
roc.data$kappa <- (2 * (roc.data$tp * roc.data$tn - roc.data$fn * roc.data$fp) ) / (
                        (roc.data$tp + roc.data$fp) * (roc.data$fp + roc.data$tn) +
                        (roc.data$tp + roc.data$fn) * (roc.data$fn + roc.data$tn)
                    )
roc.data.resnet <- subset(roc.data, model == 'ResNet Transfer Learning')
roc.data.signalstrength <- subset(roc.data, model == 'Signal Strength')
roc.data.cnn <- subset(roc.data, model == 'Convolutional Neural Network')

print("ResNet Transfer Learning Accuracy and Threshold")
resnet.opt.acc <- roc.data.resnet[which.max(roc.data.resnet$accuracy),]
resnet.opt.acc
resnet.x.pt <- resnet.opt.acc$fp[[1]] / (
            resnet.opt.acc$fp[[1]] + resnet.opt.acc$tn[[1]])

print("Signal Strength Accuracy and Threshold")
signalstrength.opt.acc <- roc.data.signalstrength[which.max(roc.data.signalstrength$accuracy),]
signalstrength.opt.acc
signalstrength.x.pt <- signalstrength.opt.acc$fp[[1]] / (
            signalstrength.opt.acc$fp[[1]] + signalstrength.opt.acc$tn[[1]])

print("Convolutional Neural Network Accuracy and Threshold")
cnn.opt.acc <- roc.data.cnn[which.max(roc.data.cnn$accuracy),]
cnn.opt.acc
cnn.x.pt <- signalstrength.opt.acc$fp[[1]] / (
            signalstrength.opt.acc$fp[[1]] + signalstrength.opt.acc$tn[[1]])


ggplot(roc.data.2l, aes(x = fp / (fp + tn), 
                     y = tp / (tp + fn), 
                     color = model)) +
    geom_path(alpha=0.5, size=1.5) + 
    geom_abline(slope = 1, intercept = 0, linetype='solid') +
    scale_color_manual(name='', values=rev(brewer.pal(4, 'Set1'))) +
    geom_vline(xintercept = signalstrength.x.pt, linetype='dashed',
               color = rev(brewer.pal(4, 'Set1'))[[1]]) +
    geom_vline(xintercept = resnet.x.pt, linetype='dashed',
               color = rev(brewer.pal(4, 'Set1'))[[2]]) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    labs(caption=auc.caption.2l) +
    theme_cowplot() + background_grid() +
    theme(
        legend.position='top',
        legend.justification='center'
    )

ggsave('./fig/fastai/resnet/roc_gradability_hisens_accuracycutoffs.png', height=8, width=8)
