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
ifelse(!dir.exists("./fig/regression"), dir.create("./fig/regression"), FALSE)
ifelse(!dir.exists("./fig/regression/cnn"), dir.create("./fig/regression/cnn"), FALSE)

#####################################################
## Load data
#####################################################

training.history <- read.csv('./calc/regression/cnn/training_history.csv')
training.history$epoch <- 1:nrow(training.history)
predictions <- read.csv('./calc/regression/cnn/predictions.csv')

training.history.incrneuron <- read.csv('./calc/regression/cnn/training_history_incrneuron.csv')
training.history.incrneuron$epoch <- 1:nrow(training.history.incrneuron)
predictions.incrneuron <- read.csv('./calc/regression/cnn/predictions_incrneuron.csv')

training.history.incrneuron.dropout <- read.csv('./calc/regression/cnn/training_history_incrneuron_dropout.csv')
training.history.incrneuron.dropout$epoch <- 1:nrow(training.history.incrneuron.dropout)
predictions.incrneuron.dropout <- read.csv('./calc/regression/cnn/predictions_incrneuron_dropout.csv')

merged.ratings <- read.csv('./calc/merged_ratings.csv')
metadata <- read.csv('./data/metadata.csv')

##############################################
## Generate plots
##############################################

# now look at predictions

predictions$model <- 'Initial'

predictions.incrneuron$model <- 'Incr. Neurons'

## increased neurons model
cor.data.incrneuron <- cor.test(x = predictions.incrneuron$predictions, 
                     y = predictions.incrneuron$truth,
                     method = 'pearson')

cor.label.incrneuron.text <- sprintf(
    'Pearson Correlation = %.2f (95%% CI [%.2f - %.2f]) p = %.2e',
    cor.data.incrneuron$estimate, 
    cor.data.incrneuron$conf.int[1],
    cor.data.incrneuron$conf.int[2],
    cor.data.incrneuron$p.value)

## increased neurons + dropout model
predictions.incrneuron.dropout$model <- 'Incr. Neurons + Dropout'

cor.data.incrneuron.dropout <- cor.test(
                     x = predictions.incrneuron.dropout$predictions, 
                     y = predictions.incrneuron.dropout$truth,
                     method = 'pearson')

cor.label.incrneuron.dropout.text <- sprintf(
    'Pearson Correlation = %.2f (95%% CI [%.2f - %.2f]) p = %.2e',
    cor.data.incrneuron.dropout$estimate, 
    cor.data.incrneuron.dropout$conf.int[1],
    cor.data.incrneuron.dropout$conf.int[2],
    cor.data.incrneuron.dropout$p.value)

## SignalStrength model
merged.metadata <- merge(merged.ratings, metadata, by='ID')

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
    

cor.label.signalstrength.text <- sprintf(
    'Pearson Correlation = %.2f (95%% CI [%.2f - %.2f]) p = %.2e',
    cor.data.signalstrength$estimate, 
    cor.data.signalstrength$conf.int[1],
    cor.data.signalstrength$conf.int[2],
    cor.data.signalstrength$p.value)

merged.predictions <- rbind(predictions,
                            predictions.incrneuron,
                            predictions.incrneuron.dropout,
                            predictions.signalstrength)

merged.predictions$model <- factor(merged.predictions$model,
                                    levels=c(
                                        'Initial',
                                        'Incr. Neurons',
                                        'Incr. Neurons + Dropout',
                                        'Signal Strength'
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
HUMAN.GRADABLE.CUTOFF <- 4
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

# calculate AUC for each model.
m <- predictions
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label <- sprintf(
    "CNN: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)

m <- predictions.incrneuron
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label.incrneuron <- sprintf(
    "Incr. Neurons: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)

m <- predictions.incrneuron.dropout
m$labels <- as.numeric(m$truth >= HUMAN.GRADABLE.CUTOFF)
pROC_obj <- roc(m$labels,m$predictions,
            smoothed = TRUE,
            # arguments for ci
            ci=TRUE, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)
auc.label.incrneuron.dropout <- sprintf(
    "Incr. Neurons + Dropout: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
    pROC_obj$auc,
    pROC_obj$ci[1],
    pROC_obj$ci[3]
)

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

auc.caption = paste(
    auc.label,
    auc.label.incrneuron,
    auc.label.incrneuron.dropout,
    auc.label.signalstrength,
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

ggsave('./fig/regression/cnn/multimodel_roc_gradability_cutoff_4.png', height=8, width=8)

# plot only initial and signalspace
roc.data.to.plot <- subset(roc.data, model %in% c('Initial', 'Signal Strength'))
roc.data.to.plot$model <- roc.data.to.plot$model %>% as.character()
roc.data.to.plot$model[roc.data.to.plot$model == 'Initial'] <- 'CNN'
minimal.label <- paste(auc.label, auc.label.signalstrength, sep='\n')

ggplot(roc.data.to.plot, aes(x = fp / (fp + tn), 
                     y = tp / (tp + fn), 
                     color = model)) +
    geom_path(alpha=0.5, size=1.5) + 
    geom_abline(slope = 1, intercept = 0, linetype='dashed') +
    scale_color_manual(name='', values=rev(brewer.pal(2, 'Set1'))) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    labs(caption=minimal.label) +
    theme_cowplot() + background_grid() +
    theme(
        legend.position='top',
        legend.justification='center'
    )

ggsave('./fig/regression/cnn/minimal_roc_gradability_cutoff_4.png', height=8, width=8)

