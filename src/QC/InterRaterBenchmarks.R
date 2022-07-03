#!/usr/bin/env Rscript
#
## InterRaterBenchmarks.R
# Perform standard benchmarking of inter-rater characteristics and quality control.
#
# Tested on | R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out"
#

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(reshape)
library(cowplot)
library(psych)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/interrater"), dir.create("./fig/interrater"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/interrater"), dir.create("./calc/interrater"), FALSE)

#####################################################
## Load data
#####################################################

r.1 <- read.csv('./data/quality_ratings/rater1.csv')
r.2 <- read.csv('./data/quality_ratings/rater2.csv')

r.1$Rater <- 'Jay'
r.2$Rater <- 'Rahul'

merged.ratings <- merge(r.1, r.2, by='ID', suffixes=c('.r1', '.r2'))
write.csv(merged.ratings, './calc/merged_ratings.csv', row.names = F)

##############################################
## Generate plots
##############################################

cohen.kappa(x=cbind(merged.ratings$Gradable.r1, merged.ratings$Gradable.r2))

# heatmap for "gradable"
gradable.tab <- table(merged.ratings$Gradable.r1, merged.ratings$Gradable.r2)
gradable.df <- melt(gradable.tab)
ggplot(gradable.df, aes(x=Var.1, y=Var.2)) + 
    geom_tile(aes(fill = value)) +
    geom_text(aes(label = round(value, 1))) +
    xlab('Jay "Gradable"') + ylab('Rahul "Gradable"') + 
    scale_fill_gradient(low = "white", high = "red") +
    scale_x_continuous(breaks=c(0,1,2)) + scale_y_continuous(breaks=c(0,1,2)) +
    theme_cowplot() +
    theme(
        legend.position='none'
    )
ggsave('./fig/interrater/gradable.png', width=5, height=5)

strict.gradable.tab <- table(merged.ratings$StrictGradable.r1, merged.ratings$StrictGradable.r2)
strict.gradable.df <- melt(strict.gradable.tab)
ggplot(strict.gradable.df, aes(x=Var.1, y=Var.2)) + 
    geom_tile(aes(fill = value)) +
    geom_text(aes(label = round(value, 1))) +
    xlab('Jay "StrictGradable"') + ylab('Rahul "StrictGradable"') + 
    scale_fill_gradient(low = "white", high = "red") +
    scale_x_continuous(breaks=c(0,1)) + scale_y_continuous(breaks=c(0,1)) +
    theme_cowplot() +
    theme(
        legend.position='none'
    )
ggsave('./fig/interrater/strict_gradable.png', width=5, height=5)

