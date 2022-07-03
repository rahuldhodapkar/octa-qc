#!/usr/bin/env Rscript
## OutlierDetection.R
#
# Script to detect outliers based on autoencoder error.
#
#

library(ggplot2)
library(magick)
library(hashmap)
library(cowplot)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/autoencoder"), dir.create("./fig/autoencoder"), FALSE)
ifelse(!dir.exists("./fig/autoencoder/outlier"), dir.create("./fig/autoencoder/outlier"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/autoencoder"), dir.create("./calc/autoencoder"), FALSE)
ifelse(!dir.exists("./calc/autoencoder/outlier"), dir.create("./calc/autoencoder/outlier"), FALSE)

#####################################################
## Load data
#####################################################

merged.ratings <- read.csv('./calc/merged_ratings.csv')
mae.df <- read.csv('./calc/autoencoder/mae_per_image.csv')
autoencoder.history <- read.csv('./calc/autoencoder/training_history_autoencoder.csv')

#####################################################
## Plot history
#####################################################

autoencoder.history$epoch <- 1:nrow(autoencoder.history)

ggplot(autoencoder.history, aes(x=epoch)) +
    geom_line(aes(y=loss)) +
    geom_line(aes(y=val_loss))

#####################################################
## Detect outliers
#####################################################

merged.mae <- merge(
    merged.ratings, mae.df, by='ID'
)

ggplot(merged.mae, aes(x=MAE, group=Group, fill=Group)) +
    geom_density(alpha = 0.5)

ggplot(merged.mae, aes(x=MAE, group=Group, fill=Group)) +
    geom_density(alpha = 0.5) +
    geom_vline(
        xintercept=quantile(merged.mae$MAE,.99),
        linetype='dashed') +
    ylab('Density') + xlab('Reconstruction Error (MAE)') +
    theme_cowplot() +
    theme(
        legend.position='top'
    )
ggsave('./fig/autoencoder/outlier/density_plot.png', height=8.25, width=8)

mae.outlier.cutoff <- quantile(
    merged.mae$MAE,
    .99)

outliers <- merged.mae[merged.mae$MAE > mae.outlier.cutoff,]

#####################################################################
## cowplot figure for outliers
#####################################################################

id2testtrain <- hashmap(
    merged.mae$ID, merged.mae$Group
)

row.plots <- list()
ix <- 1
for (i in outliers$ID) {
    p.l <- cowplot::ggdraw() +
        cowplot::draw_image(paste0('./data/images/superficial_',i,'.bmp'), scale = 0.9)
    p.r <- cowplot::ggdraw() +
        cowplot::draw_image(
           paste0('./calc/autoencoder/reconstructions/superficial_',i,'_', id2testtrain[[i]],'_recon.bmp'),
           scale = 0.9)
    row.plots[[ix]] <- plot_grid(p.l, p.r,
        labels=c(sprintf(
            'MAE = %.3f', merged.mae[merged.mae$ID == i,]$MAE
        ),''))
    ix <- ix + 1
}
outliers.plot <- plot_grid(plotlist=row.plots, ncol=1)

#####################################################################
## cowplot figure for images close to the mean
#####################################################################

mae.representative.cutoffs <- quantile(
    merged.mae$MAE,
    c(.49, .51))

representatives <- merged.mae[
    merged.mae$MAE > mae.representative.cutoffs[1] &
    merged.mae$MAE < mae.representative.cutoffs[2],]

row.plots <- list()
ix <- 1
for (i in representatives$ID[1:nrow(outliers)]) {
    p.l <- cowplot::ggdraw() +
        cowplot::draw_image(paste0('./data/images/superficial_',i,'.bmp'), scale = 0.9)
    p.r <- cowplot::ggdraw() +
        cowplot::draw_image(
           paste0('./calc/autoencoder/reconstructions/superficial_',i,'_', id2testtrain[[i]],'_recon.bmp'),
           scale = 0.9)
    row.plots[[ix]] <- plot_grid(p.l, p.r, 
        labels=c(sprintf(
            'MAE = %.3f', merged.mae[merged.mae$ID == i,]$MAE
        ),''))
    ix <- ix + 1
}
representatives.plot <- plot_grid(plotlist=row.plots, ncol=1)

#####################################################################
## cowplot figure for images close to the mean
#####################################################################

main.plot <- plot_grid(outliers.plot, representatives.plot, 
                labels=c('', ''), ncol=2)
ggsave('./fig/autoencoder/outlier/reconstructions_plot.png', height=12, width=8)

#####################################################################
## Reconstruction Error vs Gradable Sum
#####################################################################

merged.mae$MAE.percentiles <- ecdf(merged.mae$MAE)(merged.mae$MAE)

cor.data <- cor.test(
    x = merged.mae$MAE.percentiles, 
    y = merged.mae$Gradable.r1 + merged.mae$Gradable.r2,
    method='pearson')
text <- sprintf(
    'Pearson Correlation = %.2f (95%% CI [%.2f - %.2f]) p = %.2e',
    cor.data$estimate, 
    cor.data$conf.int[1],
    cor.data$conf.int[2],
    cor.data$p.value)

ggplot(merged.mae, aes(x=MAE.percentiles, y=Gradable.r1 + Gradable.r2)) +
    geom_point() + 
    xlab('Reconstruction Error Percentile') +
    ylab('Gradability Score') + 
    labs(caption=text) +
    theme_cowplot() + background_grid()

ggsave('./fig/autoencoder/outlier/recon_err_vs_gradability.png',
            height=8.25, width=8)