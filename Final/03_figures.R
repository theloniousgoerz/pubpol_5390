## Create Figures ## 
# Thelonious Goerz 
# 11/18/2025
## ---------------- ##
### Packages ###
rm(list = ls())
library(readr)
library(tidyverse)
library(fixest)
library(modelsummary)
library(haven)
library(magrittr)
library(splines)
library(glmnet)
library(KernSmooth)
library(boot)
library(tidymodels)
library(here)
library(KernSmooth)
## ----------------
## Source Model Functions 
source(here("_Functions","ols_function.R"))
source(here("_Functions","regularize_function.R"))
source(here("_Functions","random_forest_function.R"))
source(here("_Functions","xgboost_function.R"))
## --------------------------------------------------------------------------------------------------------------------------------
