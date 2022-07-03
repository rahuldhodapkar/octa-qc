#!/usr/bin/env Rscript
#
## CNNRegressionBenchmarks.R
# benchmark convolutional neural network regression against Gradable.sum
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

##############################################
## Generate plots
##############################################

# loss functions over time
training.history.plot <- melt(training.history, id="epoch")

ggplot(subset(training.history.plot, variable %in% c('loss', 'val_loss')), 
            aes(x=epoch, y=value, color=variable)) +
    geom_line() +
    scale_color_discrete(name='', labels=c('Train', 'Validation')) +
    xlab('Model Training Epoch') + ylab('Mean Absolute Error') +
    theme_half_open() + background_grid() +
    theme(
        legend.position = 'top'
    )

ggsave('./fig/regression/cnn/training_history_loss.png', height= 6, width=10)

cor.data <- cor.test(x = predictions$predictions, 
                     y = predictions$truth,
                     method = 'pearson')

cor.label.text <- sprintf(
    'Pearson Correlation = %.2f (95%% CI [%.2f - %.2f]) p = %.2e',
    cor.data$estimate, 
    cor.data$conf.int[1],
    cor.data$conf.int[2],
    cor.data$p.value)

# concordance of predictions with human-graded "truth"
ggplot(predictions, aes(x=truth, y=predictions)) +
    geom_point() +
    geom_smooth() + geom_abline(slope=1, intercept=0) +
    xlab('Gradable(Rahul) + Gradable(Jay)') + ylab('Model Prediction') +
    ggtitle(cor.label.text) +
    theme_half_open() + background_grid()
ggsave('./fig/regression/cnn/test_plot.png', height=8, width=7)

## Generate plots with all three models
#
# (1) Initial model
#
# (2) Increased number of neurons
#
# (3) Increased number of neurons + regularization (dropout)
#

training.history$model <- 'Initial'
training.history.incrneuron$model <- 'Incr. Neurons'
training.history.incrneuron.dropout$model <- 'Incr. Neurons + Dropout'

history.merged <- rbind(training.history,
                        training.history.incrneuron,
                        training.history.incrneuron.dropout)

plot.history.merged <- melt(history.merged, id=c('epoch', 'model'))

variable2printlabel <- hashmap(
    c('loss', 'val_loss'),
    c('Train', 'Validation'))

colors <- brewer.pal(3, 'Set1')
dulled.colors <-  brightness(colors, 0.5)
plot.colors <- c(rbind(colors, dulled.colors))

ggplot(subset(plot.history.merged, variable %in% c('loss', 'val_loss')),
        aes(x=epoch, y=value, 
            color=paste(model, variable2printlabel[[variable]], sep=" | "))) +
        geom_line() + 
        scale_color_manual(name='', values=plot.colors) +
        xlab('Model Training Epoch') + ylab('Mean Absolute Error') +
        theme_half_open() + background_grid() +
        theme(
            legend.position='bottom',
            legend.justification='center'
        )
ggsave('./fig/regression/cnn/multimodel_training_history.plot.png', width=10, height=6)


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

merged.predictions <- rbind(predictions,
                            predictions.incrneuron,
                            predictions.incrneuron.dropout)

merged.predictions$model <- factor(merged.predictions$model,
                                    levels=c(
                                        'Initial',
                                        'Incr. Neurons',
                                        'Incr. Neurons + Dropout'
                                    ))

ggplot(merged.predictions, aes(x=truth, y=predictions, color=model)) +
    geom_point() +
    geom_smooth()
ggsave('./fig/regression/cnn/multimodel_test_plot.png', height=8, width=7)

##################################################################
## Examine PPV with cutoff points.
##################################################################

full.cor.label <- paste(
        paste0("Initial: ", cor.label.text),
        paste0("Incr. Neurons: ", cor.label.incrneuron.text),
        paste0("Incr. Neurons + Dropout: ", cor.label.incrneuron.dropout.text),
        sep='\n'
    )

ggplot(merged.predictions, aes(x=predictions, color=truth, fill=truth, group=truth)) +
    geom_density(alpha = 0.5) +
    xlab('CNN Prediction') + ylab('Gaussian Kernel Density Estimate') +
    scale_color_continuous(name='Manual Grade') +
    scale_fill_continuous(name='Manual Grade') +
    facet_grid(model ~ .) +
    theme_cowplot() + background_grid() +
    labs(caption=full.cor.label) +
    theme(
        legend.position = 'top',
        legend.justification = 'left',
        legend.spacing.x = unit(1.5,"line"),
        legend.key.width = unit(2,"line")
    )
ggsave('./fig/regression/cnn/multimodel_gaussian_kde.png', height=8, width=12)

# now plot ROC curves for models in grid facet format as below:
#
#   |  TP  |  FN  |
#   |  FP  |  TN  |
#

# binarization cutoff from human classification data.
HUMAN.GRADABLE.CUTOFF <- 3
STEP.WIDTH <- 0.01

test.threshold.iter <- seq(
    min(merged.predictions$predictions), 
    max(merged.predictions$predictions),
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
    "Initial: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
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

auc.caption = paste(
    auc.label,
    auc.label.incrneuron,
    auc.label.incrneuron.dropout,
    sep = '\n'
)

# now make plots of ROC space.

ggplot(roc.data, aes(x = fp / (fp + tn), 
                     y = tp / (tp + fn), 
                     color = model)) +
    geom_path(alpha=0.5, size=1.5) + 
    geom_abline(slope = 1, intercept = 0, linetype='dashed') +
    scale_color_manual(name='', values=rev(brewer.pal(3, 'Set1'))) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    labs(caption=auc.caption) +
    theme_cowplot() + background_grid() +
    theme(
        legend.position='top',
        legend.justification='center'
    )

ggsave('./fig/regression/cnn/multimodel_roc.png', height=8, width=8)

