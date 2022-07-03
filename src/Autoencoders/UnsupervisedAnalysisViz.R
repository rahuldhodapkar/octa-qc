#!/usr/bin/env Rscript
#
## UnsupervisedAnalysisViz.R
# Visualation scripts for unsupervised analysis of OCTA data
#
# @author rahuldhodapkar <rahul.dhodapkar@yale.edu>
# @version 2021.05.18
#

library(ggplot2)
library(cowplot)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/unsupervised"), dir.create("./fig/unsupervised"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/unsupervised"), dir.create("./calc/unsupervised"), FALSE)

#####################################################
## Load data
#####################################################

merged.ratings <- read.csv('./calc/merged_ratings.csv')
umap.embeddings <- read.csv('./calc/umap_embeddings.csv')

merged.ratings <- merge(merged.ratings, umap.embeddings, by='ID')

##############################################
## Generate plots
##############################################

merged.ratings$Gradable.sum <- merged.ratings$Gradable.r1 + merged.ratings$Gradable.r2

ggplot(merged.ratings, aes(x=UMAP1, y=UMAP2, color=Gradable.sum)) + 
    geom_point() +
    scale_color_continuous(name='"Gradable.sum" Value') +
    theme_cowplot() +
    theme(legend.position="top",
          legend.spacing.x = unit(1.5,"line"),
          legend.key.width = unit(2,"line"))
ggsave('./fig/unsupervised/umap_embeddings.png', height=8, width=8)
