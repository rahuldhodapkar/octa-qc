#!/usr/bin/env Rscript
#
## DemographicsTable.R
# Generate data for "Table 1" in manuscript
#
# Tested on | R version 4.0.3 (2020-10-10) -- "Bunny-Wunnies Freak Out"
#

library(hashmap)

#####################################################
## Create output scaffolding
#####################################################

ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
ifelse(!dir.exists("./fig/demographics"), dir.create("./fig/demographics"), FALSE)

ifelse(!dir.exists("./calc"), dir.create("./calc"), FALSE)
ifelse(!dir.exists("./calc/demographics"), dir.create("./calc/demographics"), FALSE)

#####################################################
## Load data
#####################################################

r.1 <- read.csv('./data/quality_ratings/rater1.csv')
r.2 <- read.csv('./data/quality_ratings/rater2.csv')

r.1$Rater <- 'Jay'
r.2$Rater <- 'Rahul'

merged.ratings <- merge(r.1, r.2, by='ID', suffixes=c('.r1', '.r2'))
metadata <- read.csv('./data/metadata.csv')
merged.metadata <- merge(merged.ratings, metadata, by='ID')

#####################################################
## Generate plots
#####################################################

dmstage.map <- hashmap(c('NONE', 'MILD', 'MOD', 'S', 'E', 'HR', 'N/A'),
        c('NONE', 'NPDR', 'NPDR', 'NPDR', 'PDR', 'PDR', 'N/A'))

merged.metadata$Gradable.sum <- merged.metadata$Gradable.r1 + merged.metadata$Gradable.r2
merged.metadata$DMStage.grouped <- dmstage.map[[merged.metadata$DMStage]]

all <- merged.metadata
lq <- subset(merged.metadata, Gradable.sum <= 1) 
hq <- subset(merged.metadata, Gradable.sum == 4)

# Age
print("=====AGE=====")
print("# all")
all.age <- table(cut(all$Age, c(18, 45, 65, 120)))
all.age.pct <- all.age / sum(all.age)
sum(all.age)
all.age
all.age.pct

print("# low quality (Gradable.sum <= 1)")
lq.age <- table(cut(lq$Age, c(18, 45, 65, 120)))
lq.age.pct <- lq.age / sum(lq.age)
sum(lq.age)
lq.age
lq.age.pct

print("# high quality (Gradable.sum == 4)")
hq.age <- table(cut(hq$Age, c(18, 45, 65, 120)))
hq.age.pct <- hq.age / sum(hq.age)
sum(hq.age)
hq.age
hq.age.pct

print("# statistics - chisq.test")
chisq.test(cbind(all.age, lq.age, hq.age))

print("=====SEX=====")
print("# all")
all.sex <- table(all$Sex)
all.sex.pct <- all.sex / sum(all.sex)
sum(all.sex)
all.sex
all.sex.pct

print("# low quality (Gradable.sum <= 1)")
lq.sex <- table(lq$Sex)
lq.sex.pct <- lq.sex / sum(lq.sex)
sum(lq.sex)
lq.sex
lq.sex.pct

print("# high quality (Gradable.sum == 4)")
hq.sex <- table(hq$Sex)
hq.sex.pct <- hq.sex / sum(hq.sex)
sum(hq.sex)
hq.sex
hq.sex.pct

print("# statistics - chisq.test")
chisq.test(cbind(all.sex, lq.sex, hq.sex))

print("=====ETHNICITY=====")
print("# all")
all.ethnicity <- table(all$Ethnicity)
all.ethnicity.pct <- all.ethnicity / sum(all.ethnicity)
sum(all.ethnicity)
all.ethnicity
all.ethnicity.pct

print("# low quality (Gradable.sum <= 1)")
lq.ethnicity <- table(lq$Ethnicity)
lq.ethnicity.pct <- lq.ethnicity / sum(lq.ethnicity)
sum(lq.ethnicity)
lq.ethnicity
lq.ethnicity.pct

print("# high quality (Gradable.sum == 4)")
hq.ethnicity <- table(hq$Ethnicity)
hq.ethnicity.pct <- hq.ethnicity / sum(hq.ethnicity)
sum(hq.ethnicity)
hq.ethnicity
hq.ethnicity.pct

print("# statistics - chisq.test")
chisq.test(cbind(all.ethnicity, lq.ethnicity, hq.ethnicity))

print("=====DR Stage=====")
print("# all")
all.dmstage <- table(all$DMStage.grouped[all$DMStage.grouped != 'N/A'])
all.dmstage.pct <- all.dmstage / sum(all.dmstage)
sum(all.dmstage)
all.dmstage
all.dmstage.pct

print("# low quality (Gradable.sum <= 1)")
lq.dmstage <- table(lq$DMStage.grouped[lq$DMStage.grouped != 'N/A'])
lq.dmstage.pct <- lq.dmstage / sum(lq.dmstage)
sum(lq.dmstage)
lq.dmstage
lq.dmstage.pct

print("# high quality (Gradable.sum == 4)")
hq.dmstage <- table(hq$DMStage.grouped[hq$DMStage.grouped != 'N/A'])
hq.dmstage.pct <- hq.dmstage / sum(hq.dmstage)
sum(hq.dmstage)
hq.dmstage
hq.dmstage.pct

print("# statistics - chisq.test")
chisq.test(cbind(all.dmstage, lq.dmstage, hq.dmstage))

print("All done!")