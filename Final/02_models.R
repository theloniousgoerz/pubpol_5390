## Create Tables ## 
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
library(tictoc)
## ----------------
setwd(here())
## Source Model Functions 
source(here("Final","_Functions","ols_function.R"))
source(here("Final","_Functions","regularize_function.R"))
source(here("Final","_Functions","random_forest_function.R"))
source(here("Final","_Functions","xgboost_function.R"))
## ----------------
## Read in Data 
data = read_csv(here("Data","project_data.csv"))

tic()
# reshape data  ----------------------------------------------------------------------
wide_data = data %>%
  mutate(death_age = dyear -byear) %>%
pivot_wider(
  names_from  = c(year),
  values_from = -c(HHID,PN,year,died,less75,death_age)
) %>% mutate(die_bf_75 = case_when(
  less75 == 1 ~ "1",
  less75 == 0 ~ "0" 
),
die_bf_75 = as_factor(die_bf_75))
#  ----------------------------------------------------------------------

## Prep split analysis ----------------------------------------------------------------------
set.seed(999)
## diers (continuous) ----------------------------------------------------------------------
deceased = wide_data %>% 
  filter(died == 1) 
  # remove problematic features 
# test size 
split_d = initial_split(deceased)
train_d = training(split_d,prop = .8)
test_d = testing(split_d)
# ----------------------------------------------------------------------
# Feature Set 1
predictor_vars_base <- deceased %>% 
select(
  -death_age,
  -die_bf_75, 
  -less75,
  -contains("p_live_80_LE"),
  -contains("p_live_80"),
  -contains("p_live_75p"),
  -contains("r_chg_live75p"),
  -contains("able_to_give_10"),
  -contains("able_to_give_100"),
  -contains("HHID"),
  -contains("PN"),
  -contains("dyear"),
  -contains("byear"),
  -contains("tot"),
  -contains("beq"),
  -contains("age75_plus")
) %>% 
  colnames()
# ----------------------------------------------------------------------
# Feature set 2 
predictor_vars_full <- deceased %>% 
  select(
    -death_age,
    -die_bf_75, 
    -less75,
    -contains("r_chg_live75p"),
    -contains("able_to_give_10"),
    -contains("able_to_give_100"),
    -contains("HHID"),
    -contains("PN"),
    -contains("dyear"),
    -contains("byear"),
    -contains("HHID"),
    -contains("PN"),
    -contains("dyear"),
    -contains("byear"),
    -contains("tot"),
    -contains("beq"),
    -contains("age75_plus")
  ) %>% 
  colnames()
# Feature set 0
predictor_vars_prob <- deceased %>% 
  select(
    contains("p_live_80_LE"),
    contains("p_live_80"),
    contains("p_live_75p"),
  ) %>% 
  colnames()
# ----------------------------------------------------------------------
# 2. Build recipe 
# 2.1 Recipe 1 
FS_cont_base_recipe <- 
  recipe(
    reformulate(termlabels = predictor_vars_base, response = "death_age"),
    data = train_d) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())
# 2.2 Recipe 2 
FS_cont_full_recipe <- 
  recipe(
    reformulate(termlabels = predictor_vars_full, response = "death_age"),
    data = train_d
  ) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())
# 2.3 Recipe 3
FS_cont_prob_recipe <- 
  recipe(
    reformulate(termlabels = predictor_vars_prob, response = "death_age"),
    data = train_d
  ) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# Live Longer Than 75 ------------------------------------------------------------

# ----------------------------------------------------------------------
# Feature Set 1
predictor_vars_base_b <- deceased %>% 
  select(
    contains("srh"),
    contains("wealth_q"),
    contains("mixed_family"),
    contains("employed"),
    contains("race"),
    contains("hispanic"),
    contains("sum_chronic"),
    contains("n_children"),
    contains("highest_degree")

    
  ) %>% 
  colnames()
# ----------------------------------------------------------------------
# Feature set 2 
predictor_vars_full_b <- deceased %>% 
  select(
    contains("p_live_80_LE"),
    contains("p_live_80"),
    contains("p_live_75p"),
    contains("srh"),
    contains("wealth_q"),
    contains("mixed_family"),
    contains("employed"),
    contains("race"),
    contains("hispanic"),
    contains("sum_chronic"),
    contains("n_children"),
    contains("highest_degree")
    
  ) %>% 
  colnames()

FS_cont_prob_recipe_mort <- 
  recipe(
    reformulate(termlabels = predictor_vars_prob, response = "die_bf_75"),
    data = train_d
  ) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

FS_cont_base_recipe_mort <- 
  recipe(
    reformulate(termlabels = predictor_vars_base_b, response = "die_bf_75"),
    data = train_d
  ) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

FS_cont_full_recipe_mort <- 
  recipe(
    reformulate(termlabels = predictor_vars_full_b, response = "die_bf_75"),
    data = train_d
  ) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())
# ----------------------------------------------------------------------

### Table 2 - Continuous Outcome Comparison Table (Death Age)----------------------------------------------------------------------
# OLS ----------------------------------------------------------------------
ols_0 = ols_function(
  outcome = "death_age",
  FS_recipe = FS_cont_prob_recipe,
  training = train_d,
  testing = test_d
)
ols_1 = ols_function(
  outcome = "death_age",
  FS_recipe = FS_cont_base_recipe,
  training = train_d,
  testing = test_d
)
ols_2 = ols_function(
  outcome = "death_age",
  FS_recipe = FS_cont_full_recipe,
  training = train_d,
  testing = test_d
)
# ----------------------------------------------------------------------
# LASSO
set.seed(999)
lasso_1 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_base_recipe,
  training = train_d,
  testing = test_d,
  type = "LASSO",
  n_folds = 5, 
  grid_max = 5
)
lasso_2 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_full_recipe,
  training = train_d,
  testing = test_d,
  type = "LASSO",
  n_folds = 5, 
  grid_max = 5
)
# ----------------------------------------------------------------------
# Ridge
ridge_1 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_base_recipe,
  training = train_d,
  testing = test_d,
  type = "Ridge",
  n_folds = 5, 
  grid_max = 5
)
ridge_2 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_full_recipe,
  training = train_d,
  testing = test_d,
  type = "Ridge",
  n_folds = 5, 
  grid_max = 5
)
# ----------------------------------------------------------------------
# Elastic Net
en_1 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_base_recipe,
  training = train_d,
  testing = test_d,
  type = "Elastic",
  n_folds = 5, 
  grid_max = 5
)
en_2 = regularize_function(
  outcome = "death_age",
  FS_recipe = FS_cont_full_recipe,
  training = train_d,
  testing = test_d,
  type = "Elastic",
  n_folds = 5, 
  grid_max = 5
)
# ----------------------------------------------------------------------
# Tree
set.seed(999)
tree_1 = random_forest(outcome = "death_age",
                   training = train_d,
                   testing = test_d,
                   FS_recipe = FS_cont_base_recipe,
                   n_trees = 1, 
                   n_folds = 5,
                   grid_max = 5,
                   label = "tree1")
tree_2 = random_forest(outcome = "death_age",
                       training = train_d,
                       testing = test_d,
                       FS_recipe = FS_cont_full_recipe,
                       n_trees = 1, 
                       n_folds = 5,
                       grid_max = 5,
                       label = "tree2")

# ----------------------------------------------------------------------
# Random Forest 
set.seed(999)
forest_1 = random_forest(outcome = "death_age",
                       training = train_d,
                       testing = test_d,
                       FS_recipe = FS_cont_base_recipe,
                       n_trees = 100, 
                       n_folds = 5,
                       grid_max = 5,
                       label = "forest1")
forest_2 = random_forest(outcome = "death_age",
                       training = train_d,
                       testing = test_d,
                       FS_recipe = FS_cont_full_recipe,
                       n_trees = 100, 
                       n_folds = 5,
                       grid_max = 5,
                       label = "forest2")

# ----------------------------------------------------------------------
# XGBoost 
set.seed(999)
boost_1 = xgboost_function(
  outcome = "death_age",
  training = train_d,
  testing = test_d, 
  FS_recipe = FS_cont_base_recipe,
  n_trees = 100,
  n_folds = 5, 
  grid_max = 5,
  label = "1"
)
boost_2 = xgboost_function(
  outcome = "death_age",
  training = train_d,
  testing = test_d, 
  FS_recipe = FS_cont_full_recipe,
  n_trees = 100,
  n_folds = 5, 
  grid_max = 5,
  label = "2"
)
# )------------------------------------ (Save model info for table making)----------------------------------------------------------------------
# -- ols -- #
#write_rds(ols_0,here("Final","Output","Performance","ols_sum_0.rds"))
#write_rds(ols_1,here("Final","Output","Performance","ols_sum_1.rds"))
#write_rds(ols_2,here("Final","Output","Performance","ols_sum_2.rds"))
## -- lasso -- #
#write_rds(lasso_1,here("Final","Output","Performance","lasso_sum_1.rds"))
#write_rds(lasso_2,here("Final","Output","Performance","lasso_sum_2.rds"))
## -- ridge -- #
#write_rds(ridge_1,here("Final","Output","Performance","ridge_sum_1.rds"))
#write_rds(ridge_2,here("Final","Output","Performance","ridge_sum_2.rds"))
## -- elastic -- # 
#write_rds(en_1,here("Final","Output","Performance","elastic_sum_1.rds"))
#write_rds(en_2,here("Final","Output","Performance","elastic_sum_2.rds"))
##-- tree -- # 
#write_rds(tree_1,here("Final","Output","Performance","tree_sum_1.rds"))
#write_rds(tree_2,here("Final","Output","Performance","tree_sum_2.rds"))
# -- forest -- # 
write_rds(forest_1,here("Final","Output","Performance","forest_sum_1.rds"))
write_rds(forest_2,here("Final","Output","Performance","forest_sum_2.rds"))
# -- boost -- # 
write_rds(boost_1,here("Final","Output","Performance","boost_sum_1.rds"))
write_rds(boost_2,here("Final","Output","Performance","boost_sum_2.rds"))
beepr::beep()

### Table 2 - Continuous Outcome Comparison Table (Mortality)----------------------------------------------------------------------
# ----------------------------------------------------------------------
## Logit
tic()
logit_0 = binary_function(outcome = "die_bf_75",
                training = train_d,
                testing = test_d,
                FS_recipe = FS_cont_prob_recipe_mort,
                label = "1"
)

logit_1 = binary_function(outcome = "die_bf_75",
                          training = train_d,
                          testing = test_d,
                          FS_recipe = FS_cont_base_recipe_mort,
                          label = "2"
)
logit_2 = binary_function(outcome = "die_bf_75",
                          training = train_d,
                          testing = test_d,
                          FS_recipe = FS_cont_full_recipe_mort,
                          label = "3"
)

## Logit Lasso
logit_lasso_1 = regularize_function_binary(outcome = "die_bf_75",
                          training = train_d,
                          testing = test_d,
                          FS_recipe = FS_cont_base_recipe_mort,
                          label = "1",
                          type = "LASSO",
                          n_folds = 5,
                          grid_max = 5
)
toc()
tic()
logit_lasso_2 = regularize_function_binary(outcome = "die_bf_75",
                                           training = train_d,
                                           testing = test_d,
                                           FS_recipe = FS_cont_full_recipe_mort,
                                           label = "2",
                                           type = "LASSO",
                                           n_folds = 5,
                                           grid_max = 5
)
## Logit Ridge 
logit_ridge_1 = regularize_function_binary(outcome = "die_bf_75",
                                           training = train_d,
                                           testing = test_d,
                                           FS_recipe = FS_cont_base_recipe_mort,
                                           label = "1",
                                           type = "Ridge",
                                           n_folds = 5,
                                           grid_max = 5
)
logit_ridge_2 = regularize_function_binary(outcome = "die_bf_75",
                                           training = train_d,
                                           testing = test_d,
                                           FS_recipe = FS_cont_full_recipe_mort,
                                           label = "2",
                                           type = "Ridge",
                                           n_folds = 5,
                                           grid_max = 5
)
## Logit Elastic 
logit_en_1 = regularize_function_binary(outcome = "die_bf_75",
                                           training = train_d,
                                           testing = test_d,
                                           FS_recipe = FS_cont_base_recipe_mort,
                                           label = "1",
                                           type = "Elastic",
                                           n_folds = 5,
                                           grid_max = 5
)
logit_en_2 = regularize_function_binary(outcome = "die_bf_75",
                                           training = train_d,
                                           testing = test_d,
                                           FS_recipe = FS_cont_full_recipe_mort,
                                           label = "2",
                                           type = "Elastic",
                                           n_folds = 5,
                                           grid_max = 5
)
# ----------------------------------------------------------------------
## Tree
tree_binary_1 = random_forest_binary(outcome = "die_bf_75",
                                       training = train_d,
                                       testing = test_d,
                                       FS_recipe = FS_cont_base_recipe_mort,
                                       n_trees = 1, 
                                       n_folds = 5,
                                       grid_max = 5,
                                       label = "tree_binary1")

tree_binary_2 = random_forest_binary(outcome = "die_bf_75",
                                       training = train_d,
                                       testing = test_d,
                                       FS_recipe = FS_cont_full_recipe_mort,
                                       n_trees = 1, 
                                       n_folds = 5,
                                       grid_max = 5,
                                       label = "tree_binary2")
# ----------------------------------------------------------------------
## Random forest
forest_binary_1 = random_forest_binary(outcome = "die_bf_75",
                         training = train_d,
                         testing = test_d,
                         FS_recipe = FS_cont_base_recipe_mort,
                         n_trees = 100, 
                         n_folds = 5,
                         grid_max = 5,
                         label = "forest_binary1")

forest_binary_2 = random_forest_binary(outcome = "die_bf_75",
                                       training = train_d,
                                       testing = test_d,
                                       FS_recipe = FS_cont_full_recipe_mort,
                                       n_trees = 100, 
                                       n_folds = 5,
                                       grid_max = 5,
                                       label = "forest_binary2")

# ----------------------------------------------------------------------
## XGBoost 
boost_b_1 = xgboost_binary_function(outcome = "die_bf_75",
                     training = train_d,
                     testing = test_d,
                     FS_recipe = FS_cont_base_recipe_mort,
                     n_trees = 100, 
                     n_folds = 5,
                     grid_max = 5,
                     label = "boost_binary1")
boost_b_2 = xgboost_binary_function(outcome = "die_bf_75",
                                    training = train_d,
                                    testing = test_d,
                                    FS_recipe = FS_cont_full_recipe_mort,
                                    n_trees = 100, 
                                    n_folds = 5,
                                    grid_max = 5,
                                    label = "boost_binary2")
toc()
# )------------------------------------ (Save model info for table making)----------------------------------------------------------------------
# -- logit -- #
write_rds(logit_0,here("Final","Output","Performance","logit_sum_0.rds"))
write_rds(logit_1,here("Final","Output","Performance","logit_sum_1.rds"))
write_rds(logit_2,here("Final","Output","Performance","logit_sum_2.rds"))
# -- lasso -- #
write_rds(logit_lasso_1,here("Final","Output","Performance","logit_lasso_sum_1.rds"))
write_rds(logit_lasso_2,here("Final","Output","Performance","logit_lasso_sum_2.rds"))
# -- ridge -- #
write_rds(logit_ridge_1,here("Final","Output","Performance","logit_ridge_sum_1.rds"))
write_rds(logit_ridge_2,here("Final","Output","Performance","logit_ridge_sum_2.rds"))
# -- elastic -- # 
write_rds(logit_en_1,here("Final","Output","Performance","logit_elastic_sum_1.rds"))
write_rds(logit_en_2,here("Final","Output","Performance","logit_elastic_sum_2.rds"))
# -- tree -- # 
write_rds(tree_binary_1,here("Final","Output","Performance","binary_tree_sum_1.rds"))
write_rds(tree_binary_2,here("Final","Output","Performance","binary_tree_sum_2.rds"))
# -- forest -- # 
write_rds(forest_binary_1,here("Final","Output","Performance","binary_forest_sum_1.rds"))
write_rds(forest_binary_2,here("Final","Output","Performance","binary_forest_sum_2.rds"))
# -- boost -- # 
write_rds(boost_b_1,here("Final","Output","Performance","binary_boost_sum_1.rds"))
write_rds(boost_b_2,here("Final","Output","Performance","binary_boost_sum_2.rds"))
beep()
# ########################################################################
# Make GOF table 

