#!/usr/bin/env Rscript
#
## FastaiTrainingHistory.R
# Plot training history for manuscript figures
#
# Tested on | R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out"
#

library(dplyr)
library(ggplot2)
library(reshape)
library(cowplot)
library(RColorBrewer)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/fastai"), dir.create("./fig/fastai"), FALSE)
ifelse(!dir.exists("./fig/fastai/resnet"), dir.create("./fig/fastai/resnet"), FALSE)

#####################################################
## Load data
#####################################################

hisens.history <- read.csv('./calc/fastai/resnet/hisens_history.csv')
hisens.history$epoch <- 1:nrow(hisens.history)

hisens.history.cnn <- read.csv('./calc/fastai/resnet/alexnet_hisens_history.csv')
hisens.history.cnn$epoch <- 1:nrow(hisens.history.cnn)

hispec.history <- read.csv('./calc/fastai/resnet/hispec_history.csv')
hispec.history$epoch <- 1:nrow(hispec.history)

hispec.history.cnn <- read.csv('./calc/fastai/resnet/alexnet_hispec_history.csv')
hispec.history.cnn$epoch <- 1:nrow(hispec.history.cnn)

#####################################################
## Generate History Plots
#####################################################

hisens.df <- melt(hisens.history[,!colnames(hisens.history) 
                    %in% c('accuracy')], id='epoch')
hisens.df$variable <- paste0('resnet_', hisens.df$variable)
hisens.cnn.df <- melt(hisens.history.cnn[,!colnames(hisens.history.cnn) 
                                         %in% c('accuracy')], id='epoch')
hisens.cnn.df$variable <- paste0('cnn_', hisens.cnn.df$variable)
hisens.plot.df <- rbind(hisens.df, hisens.cnn.df)

ggplot(hisens.plot.df,
    aes(
        x = epoch, y = value, color = variable,
    )) +
    scale_color_manual(values=rev(brewer.pal(n=4, name='Paired'))) +
    geom_line(size=1.5, alpha=0.8) + theme_cowplot() + background_grid() +
    xlab('Epoch') + ylab('Cross Entropy') +
    theme(
        legend.position='top',
        legend.justification='left'
    )
ggsave('./fig/fastai/resnet/hisens_history.png', height=6, width=10)


hispec.df <- melt(hispec.history[,!colnames(hispec.history) 
                                 %in% c('accuracy')], id='epoch')
hispec.df$variable <- paste0('resnet_', hisens.df$variable)
hispec.cnn.df <- melt(hispec.history.cnn[,!colnames(hispec.history.cnn) 
                                         %in% c('accuracy')], id='epoch')
hispec.cnn.df$variable <- paste0('cnn_', hispec.cnn.df$variable)
hispec.plot.df <- rbind(hispec.df, hispec.cnn.df)

ggplot(hispec.plot.df,
    aes(
        x = epoch, y = value, color = variable
    )) +
    scale_color_manual(values=rev(brewer.pal(n=4, name='Paired'))) +
    geom_line(size=1.5, alpha=0.8)  + theme_cowplot() + background_grid() +
    xlab('Epoch') + ylab('Cross Entropy') +
    theme(
        legend.position='top',
        legend.justification='left'
    )
ggsave('./fig/fastai/resnet/hispec_history.png', height=6, width=10)

