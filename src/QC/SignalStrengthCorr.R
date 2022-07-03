#!/usr/bin/env Rscript
#
## SignalStrengthCorr.R
# Examine the correlation of signal strength with
#
# Tested on | R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out"
#

library(ggplot2)
library(cowplot)
library(dplyr)
library(reshape)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/SignalStrength"), dir.create("./fig/SignalStrength"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)

#####################################################
## Load data
#####################################################

merged.ratings <- read.csv('./calc/merged_ratings.csv')
metadata <- read.csv('./data/metadata.csv')

#####################################################
## Generate plots
#####################################################

merged.metadata <- merge(merged.ratings, metadata, by='ID')

merged.metadata$Gradable.sum <- merged.metadata$Gradable.r1 + merged.metadata$Gradable.r2

# heatmap
heatmap.data <- table(merged.metadata$SignalStrength,
                      merged.metadata$Gradable.sum) %>%
                melt()

ggplot(heatmap.data, aes(x=Var.1, y=Var.2, fill=value)) +
    geom_tile() +
    xlab('Signal Strength') +
    ylab('Manual Gradability Score') +
    scale_fill_continuous(name='Number of Images') +
    scale_x_continuous(breaks=0:10) +
    scale_y_continuous(breaks=0:4) +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='left',
        legend.spacing.x = unit(1.5,"line"),
        legend.key.width = unit(2,"line")
    )
ggsave('./fig/SignalStrength/SignalStrengthvGradableSumHeatmap.png',
        height=8.25, width=8)

# boxplot
merged.metadata$Gradable.sum.factor <- factor(
    as.character(merged.metadata$Gradable.sum),
    ordered=T,
    levels=0:4 %>% as.character())

ggplot(merged.metadata, aes(group = Gradable.sum, x = Gradable.sum, y = SignalStrength)) +
    geom_boxplot(outlier.shape=NA) +
    geom_jitter(alpha = 0.4, height=0) + 
    xlab('Manual Gradability Score') +
    ylab('Signal Strength') + 
    scale_y_continuous(breaks=0:10) +
    geom_hline(yintercept=6, color='red', linetype='dashed') +
    theme_cowplot() + background_grid()

ggsave('./fig/SignalStrength/SignalStrengthvGradableSumBoxplot.png',
        height=8.25, width=8)

ggplot(merged.metadata, aes(group = Gradable.sum, x = Gradable.sum, y = SignalStrength)) +
    geom_boxplot(outlier.shape=NA) +
    geom_jitter(alpha = 0.4, height=0) + 
    xlab('Manual Gradability Score') +
    ylab('Signal Strength') + 
    scale_y_continuous(breaks=0:10) +
    geom_hline(yintercept=6, color='red', linetype='dashed') +
    theme_cowplot() + background_grid()

ggsave('./fig/SignalStrength/SignalStrengthvGradableSumBoxplot.png',
        height=8.25, width=8)

ggplot(merged.metadata, aes(group = SignalStrength, x = SignalStrength, y = Gradable.sum)) +
    geom_boxplot(outlier.shape=NA) +
    geom_jitter(alpha = 0.4, height=0) + 
    ylab('Manual Gradability Score') +
    xlab('Signal Strength') + 
    scale_x_continuous(breaks=0:10) +
    scale_y_continuous(breaks=0:4) +
    geom_vline(xintercept=5.5, color='red', linetype='dashed') +
    theme_cowplot() + background_grid()

ggsave('./fig/SignalStrength/GradableSumvSignalStrengthBoxplot.png',
        height=6, width=9)