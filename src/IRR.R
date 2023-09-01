
#Cohen's Kappa using "psych" and irr pacakge 

install.packages("irr")
library("irr")

setwd("C:/0_RData/1")
dir()

a <- read.csv("All_adjudicated_ELL_data_1022.csv", header = TRUE)
colnames(a)

overall <- a[, c(5, 14)]
cohesion <- a[, c(6, 15)]
syntax <- a[, c(7, 16)]
vocabulary <- a[, c(8, 17)]
phraseology <- a[, c(9, 18)]
grammar <- a[, c(10, 19)]
conventions <- a[, c(11, 20)]


kappa2(overall, weight = "squared")
kappa2(cohesion, weight = "squared")
kappa2(syntax, weight = "squared")
kappa2(vocabulary, weight = "squared")
kappa2(phraseology, weight = "squared")
kappa2(grammar, weight = "squared")
kappa2(conventions, weight = "squared")

