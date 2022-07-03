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
library(stringr)
library(umap)
library(hashmap)
library(assertr)

library(igraph)
library(cccd)
library(mclust)
library(aricode)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/autoencoder"), dir.create("./fig/autoencoder"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/autoencoder"), dir.create("./calc/autoencoder"), FALSE)

#####################################################
## Load data
#####################################################

internalrep.filenames <- list.files('./calc/autoencoder/internalrep/')

d <- str_match(internalrep.filenames, "superficial_([0-9]+)_([a-z]+)")

merged.internalrep <- NULL
for (i in 1:length(internalrep.filenames)) {
    v <- read.csv(paste(
            'calc', 'autoencoder', 'internalrep',
            internalrep.filenames[i], sep='/'), header=FALSE)

    tmp <- t(v)
    colnames(tmp) <- paste0('V', 1:nrow(v))
    rownames(tmp) <- as.integer(d[,2][i])

    if(is.null(merged.internalrep)) {
        merged.internalrep <- tmp
    } else {
        merged.internalrep <- rbind(merged.internalrep, tmp)
    }
}

merged.ratings <- read.csv('./calc/merged_ratings.csv')

metadata <- read.csv('./data/metadata.csv')

mae.df <- read.csv('./calc/autoencoder/mae_per_image.csv')

training.history <- read.csv('./calc/autoencoder/training_history_autoencoder.csv')

#####################################################
## Plot training history
#####################################################

# loss functions over time
training.history$epoch <- 1:nrow(training.history)
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

ggsave('./fig/autoencoder/autoencoder_training_history.png', height= 6, width=10)

#####################################################
## Calculate UMAP coordinates
#####################################################

umap.coords <- umap(merged.internalrep)
umap.coords.df <- data.frame(
    UMAP1=umap.coords$layout[,1],
    UMAP2=umap.coords$layout[,2],
    ID=as.integer(rownames(umap.coords$layout))
)

# merge with ratings data.

merged.ratings.umap <- merge(
    merged.ratings, umap.coords.df, by='ID'
)
merged.ratings.umap <- merge(
    merged.ratings.umap, mae.df, by='ID'
)

#####################################################
## Perform Louvain clustering
#####################################################

knng <- nng(merged.internalrep, k=15)
knng.undirected <- as.undirected(knng, mode='each')
louvain.clusters <- cluster_louvain(knng.undirected)
walktrap.clusters <- cluster_walktrap(knng)
kmeans.clusters <- kmeans(umap.coords$layout[,1:2], centers=2) #centers ~ k

merged.ratings.umap$cluster <- as.factor(
    louvain.clusters$membership[
        match(seq(0, nrow(merged.internalrep) - 1), rownames(merged.internalrep))
    ]
)

plot.df <- merge(
    merged.ratings.umap, metadata,
    by='ID'   
)

eye2eyename <- hashmap(c(0,1), c('OD', 'OS'))

plot.df$Eye <- factor(eye2eyename[[plot.df$Eye]])
plot.df$HBA1C[plot.df$HBA1C == '>14.0%'] <- 15
plot.df$HBA1C <- as.numeric(plot.df$HBA1C)

#####################################################
## Plot data
#####################################################

ggplot(plot.df, aes(x=UMAP1, y=UMAP2, 
                                color=Gradable.r1 + Gradable.r2 == 4)) +
    geom_point()

ggplot(plot.df, aes(x=UMAP1, y=UMAP2, 
                                color=cluster)) +
    geom_point()

ggplot(plot.df, aes(x=UMAP1, y=UMAP2, 
                                color=Eye)) +
    geom_point()

ggplot(plot.df, aes(x=UMAP1, y=UMAP2, 
                                color=log(MAE))) +
    geom_point()

ggplot(plot.df, aes(x=cluster, y=Gradable.r1 + Gradable.r2, 
                    color=cluster, fill=cluster)) +
    geom_violin(alpha=0.5) + geom_jitter()


#####################################################
## Generate polished plots
#####################################################

# Quality mapping
gradablesum2label <- hashmap(
    c(0,1,2,3,4),
    c('low', 'low', 'borderline', 'borderline', 'gradable')
)

plot.df$Quality <- factor(gradablesum2label[[
    plot.df$Gradable.r1 + plot.df$Gradable.r2
]], levels=c('gradable', 'borderline', 'low'))

# Ethnicity mapping
ethnicity2ethnicitylabel <- hashmap(
    c(0, 1, 2, 3, 4),
    c('Caucasian', 'Black', 'Hispanic', 'Asian', 'Other')
)

plot.df$EthnicityLabel <- factor(ethnicity2ethnicitylabel[[
    plot.df$Ethnicity
]], levels = c('Caucasian', 'Black', 'Hispanic', 'Asian', 'Other'))

# Generate plots
nclusters <- length(levels(plot.df$cluster))
ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=cluster)) +
    geom_point() +
    scale_color_manual(
        name='Cluster  ',
        values=brewer.pal(nclusters, 'Set1')) +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
ggsave('./fig/autoencoder/LouvainCluster.png', width=8, height=8.25)


###
# calculate adjusted rand index of clustering for high sensitivity
# and high specificity scenarios
#
h.spec.map <- hashmap(c('low', 'borderline', 'gradable'), c('lo', 'lo', 'hi'))
h.sens.map <- hashmap(c('low', 'borderline', 'gradable'), c('lo', 'hi', 'hi'))
cluster.map <- hashmap(c('1','2','3','4','5'), c('lo', 'hi', 'hi', 'lo', 'lo'))

adjustedRandIndex(cluster.map[[plot.df$cluster]],
                  h.spec.map[[plot.df$Quality]])

adjustedRandIndex(cluster.map[[plot.df$cluster]],
                  h.sens.map[[plot.df$Quality]])

AMI(plot.df$cluster, h.spec.map[[plot.df$Quality]])
AMI(plot.df$cluster, h.sens.map[[plot.df$Quality]])

###
# calculate bootstrapped confidence intervals
#
calc.sens.ami <- function(x, ixs) {
    AMI(x[ixs,]$cluster, h.sens.map[[x[ixs,]$Quality]])
}

bootstrap <- boot(plot.df, calc.sens.ami, R=1000)
boot.ci(boot.out = bootstrap, type=c('perc'))


ggplot(plot.df, aes(x=cluster, y=Gradable.r1 + Gradable.r2, 
                    color=cluster, fill=cluster)) +
    geom_violin(alpha=0.5) + geom_jitter() +
    scale_color_manual(
        name='Cluster  ',
        values=brewer.pal(nclusters, 'Set1')) +
    scale_fill_manual(
        name='Cluster  ',
        values=brewer.pal(nclusters, 'Set1')) +
    ylab('Gradability Score') + xlab('Cluster') +
    theme_cowplot() + background_grid(major='y', minor='y') +
    theme(
        legend.position='none',
        legend.justification='center'
    )
ggsave('./fig/autoencoder/ClusterGradabilityScores.png', width=8, height=8.25)

p.1 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=Quality)) +
    scale_color_manual(name='Image Quality  ',
        labels=c('gradable (4)', 'borderline (2-3)', 'low (0-1)'),
        values=c('#008450', '#EFB700', '#B6B6B6')) +
    geom_point() +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.1
ggsave('./fig/autoencoder/QualityUMAP.png', width=8, height=8.25)

p.2 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=Eye)) +
    geom_point() +
    scale_color_discrete(name='Eye  ') +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.2
ggsave('./fig/autoencoder/EyeUMAP.png', width=8, height=8.25)

p.3 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=EthnicityLabel)) +
    geom_point() +
    scale_color_discrete(name='Ethnicity  ') +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.3
ggsave('./fig/autoencoder/EthnictyUMAP.png', width=8, height=8.25)


## Now create binned plots with internal continuous scales
if(range(plot.df$Age, na.rm=T)[1] < 20) {
    stop("unexpected lower bound for age range. Did you add new data?")
}
if(range(plot.df$Age, na.rm=T)[2] > 90) {
    stop("unexpected upper bound for age range. Did you add new data?")
}

age.breaks <- c(20, 30, 40, 50, 60, 70, 80, 90)
age.labs <- c('20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89')

plot.df$Age.bin <- factor(cut(plot.df$Age,
                        breaks=age.breaks,
                        labels=age.labs,
                        include.lowest=TRUE,
                        right=TRUE, #interval form ~ [lo, hi)
                        ordered_result=TRUE),
                    levels=c(age.labs, 'Unknown'))
plot.df$Age.bin[is.na(plot.df$Age.bin)] <- 'Unknown'


p.4 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=Age.bin)) +
    geom_point(na.rm=FALSE) +
    scale_color_manual(
        name='Age  ',
        values=c(
            brewer.pal(length(age.labs), 'Reds'),
            '#C2C2C2')) +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.4
ggsave('./fig/autoencoder/AgeUMAP.png', width=8, height=8.25)

hba1c.breaks <- c(0, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 13, 14, 16)
hba1c.labs <- c('<5.5%', '5.5% - 5.9%', 
                '6.0% - 6.4%', '6.5% - 6.9%', 
                '7.0% - 7.4%', '7.5% - 7.9%', 
                '8.0% - 8.4%', '8.5% - 8.9%',
                '9.0% - 9.4%', '9.5% - 9.9%',
                '10.0% - 10.9%', '11.0% - 11.9%',
                '12.0% - 12.9%', '13.0% - 13.9%',
                '≥14%')

plot.df$HBA1C.bin <- factor(cut(plot.df$HBA1C,
                        breaks=hba1c.breaks,
                        labels=hba1c.labs,
                        include.lowest=TRUE,
                        right=TRUE, #interval form ~ [lo, hi)
                        ordered_result=TRUE),
                    levels=c(hba1c.labs, 'Unknown'))
plot.df$HBA1C.bin[is.na(plot.df$HBA1C.bin)] <- 'Unknown'


p.5 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=HBA1C.bin)) +
    geom_point() +
    scale_color_manual(
        name='HBA1C  ',
        values=c(
            colorRampPalette(c("#FFCCCB", "#FF0000"))(length(hba1c.labs)),
            '#C2C2C2')) +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.5
ggsave('./fig/autoencoder/HBA1CFineGrainUMAP.png', width=8, height=8.25)


hba1c.breaks.coarse <- c(0, 6, 7, 9, 16)
hba1c.labs.coarse <- c('<6%', 
                '6.0% - 6.9%', 
                '7.0% - 8.9%',
                '≥9%')

plot.df$HBA1C.bin.coarse <- factor(cut(plot.df$HBA1C,
                        breaks=hba1c.breaks.coarse,
                        labels=hba1c.labs.coarse,
                        include.lowest=TRUE,
                        right=TRUE, #interval form ~ [lo, hi)
                        ordered_result=TRUE),
                    levels=c(hba1c.labs.coarse, 'Unknown'))
plot.df$HBA1C.bin.coarse[is.na(plot.df$HBA1C.bin.coarse)] <- 'Unknown'


p.5 <- ggplot(plot.df, aes(x=UMAP1, y=UMAP2, color=HBA1C.bin.coarse)) +
    geom_point() +
    scale_color_manual(
        name='HBA1C  ',
        values=c(
            brewer.pal(length(hba1c.labs.coarse), 'Reds'),
            '#C2C2C2')) +
    theme_cowplot() +
    theme(
        legend.position='top',
        legend.justification='center'
    )
p.5
ggsave('./fig/autoencoder/HBA1CCoarseGrainUMAP.png', width=8, height=8.25)


