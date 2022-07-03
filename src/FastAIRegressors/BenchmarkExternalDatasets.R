#!/usr/bin/env Rscript
## BenchmarkExternalDatasets.R
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
ifelse(!dir.exists("./fig/fastai/resnet/validation"), dir.create("./fig/fastai/resnet/validation"), FALSE)

#####################################################
## Load data
#####################################################

predictions.hisens <- read.csv('./calc/fastai/resnet/validation/hisens_cnn_6x6_superficial.csv')
predictions.hisens$model <- 'High Sensitivity ResNet'
predictions.hispec <- read.csv('./calc/fastai/resnet/validation/hispec_cnn_6x6_superficial.csv')
predictions.hispec$model <- 'High Specificity ResNet'


merged.predictions <- rbind(predictions.hisens,
                            predictions.hispec)

merged.predictions$model <- factor(merged.predictions$model,
                                    levels=c(
                                        'High Sensitivity ResNet',
                                        'High Specificity ResNet'
                                    ))

GenerateROCData <- function(merged.predictions, human.gradable.cutoff = 2, step.width=0.01) {

    test.threshold.iter <- seq(
        min(merged.predictions$predictions),
        max(merged.predictions$predictions) + step.width*10,
        by=step.width)

    roc.data <- NULL
    for (t in test.threshold.iter) {
        x <- merged.predictions %>%
                group_by(model) %>%
                summarise(
                    tp = sum(truth >= human.gradable.cutoff
                             & predictions >= t), 
                    fn = sum(truth >= human.gradable.cutoff
                             & predictions < t),
                    fp = sum(truth < human.gradable.cutoff
                             & predictions >= t),
                    tn = sum(truth < human.gradable.cutoff
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

    roc.data$accuracy <- (roc.data$tp + roc.data$tn) / (roc.data$tp + roc.data$tn + roc.data$fp + roc.data$fn)
    roc.data$kappa <- (2 * (roc.data$tp * roc.data$tn - roc.data$fn * roc.data$fp) ) / (
                            (roc.data$tp + roc.data$fp) * (roc.data$fp + roc.data$tn) +
                            (roc.data$tp + roc.data$fn) * (roc.data$fn + roc.data$tn)
                        )

    return(roc.data)
}

CalculateAUC <- function(predictions, human.gradable.cutoff = 2) {
    m <- predictions
    m$labels <- as.numeric(m$truth >= human.gradable.cutoff)
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
    return(pROC_obj)
}

PrintLabelFromROC <- function(pROC_obj, model.name) {
    return(
        sprintf(
            "%s: AUC = %.2f (95%% DeLong CI [%.2f - %.2f])",
            model.name,
            pROC_obj$auc,
            pROC_obj$ci[1],
            pROC_obj$ci[3]
        )
    )
}

##############
## low quality cutoff
##############
low.quality.cutoff.roc <- GenerateROCData(
    merged.predictions,
    human.gradable.cutoff = 1)
hisens.low.quality.roc.data <- subset(low.quality.cutoff.roc, model == 'High Sensitivity ResNet')
hispec.low.quality.roc.data <- subset(low.quality.cutoff.roc, model == 'High Specificity ResNet')

hispec.low.quality.pROC <- CalculateAUC(predictions.hispec, human.gradable.cutoff = 2)
print(PrintLabelFromROC(hispec.low.quality.pROC, 'HiSpec Model'))
hisens.low.quality.pROC <- CalculateAUC(predictions.hisens, human.gradable.cutoff = 2)
print(PrintLabelFromROC(hisens.low.quality.pROC, 'HiSens Model'))

# acc estimates
hispec.low.quality.roc.data.opt.acc <- hispec.low.quality.roc.data[which.max(hispec.low.quality.roc.data$accuracy),]
hispec.low.quality.roc.data.opt.acc

hisens.low.quality.roc.data.opt.acc <- hisens.low.quality.roc.data[which.max(hisens.low.quality.roc.data$accuracy),]
hisens.low.quality.roc.data.opt.acc


hispec.lq.prev.opt.acc <- head(
    subset(hispec.low.quality.roc.data, threshold > 0.007982801),
    n=1
)
hispec.lq.prev.opt.acc

hisens.lq.prev.opt.acc <- head(
    subset(hisens.low.quality.roc.data, threshold > 0.5079828),
    n=1
)
hisens.lq.prev.opt.acc

# high quality cutoff
high.quality.cutoff.roc <- GenerateROCData(
    merged.predictions,
    human.gradable.cutoff = 2)
hisens.high.quality.roc.data <- subset(high.quality.cutoff.roc, model == 'High Sensitivity ResNet')
hispec.high.quality.roc.data <- subset(high.quality.cutoff.roc, model == 'High Specificity ResNet')

hispec.high.quality.pROC <- CalculateAUC(predictions.hispec, human.gradable.cutoff = 4)
print(PrintLabelFromROC(hispec.high.quality.pROC, 'HiSpec Model'))
hisens.high.quality.pROC <- CalculateAUC(predictions.hisens, human.gradable.cutoff = 4)
print(PrintLabelFromROC(hisens.high.quality.pROC, 'HiSens Model'))

hispec.high.quality.roc.data.opt.acc <- hispec.high.quality.roc.data[which.max(hispec.high.quality.roc.data$accuracy),]
hispec.high.quality.roc.data.opt.acc

hisens.high.quality.roc.data.opt.acc <- hisens.high.quality.roc.data[which.max(hisens.high.quality.roc.data$accuracy),]
hisens.high.quality.roc.data.opt.acc

# using previously determined cutoffs

hispec.hq.prev.opt.acc <- head(
    subset(hispec.high.quality.roc.data, threshold > 0.4979828),
    n=1
)
hispec.hq.prev.opt.acc

hisens.hq.prev.opt.acc <- head(
    subset(hisens.high.quality.roc.data, threshold > 0.9979821),
    n=1
)
hisens.hq.prev.opt.acc
