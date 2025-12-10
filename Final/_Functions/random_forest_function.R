## Random Forest Function ## 
# Thelonious Goerz 
# 11/18/2025
## ---------------- ##
### Packages ###
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

random_forest = function(outcome,
                               training,
                               testing,
                               FS_recipe,
                               n_trees, 
                               n_folds,
                               grid_max,
                               label
){
# Descriptives
  ## SD Y ------------------------------
  is_sd_y = sd(training[[outcome]])
  os_sd_y = sd(testing[[outcome]])
  ## Tune Parameters ------------------------------
 
  
    tune_spec <- rand_forest(
      mtry = tune(),
      trees = n_trees,
      min_n = tune()
      ) %>%
      set_mode("regression") %>%
      set_engine("ranger", importance = "permutation")
  
  tune_wf <- workflow() %>%
    add_recipe(FS_recipe) %>%
    add_model(tune_spec)
  
## Define folds 
folds = vfold_cv(training,v = n_folds) 

# parallelize
doParallel::registerDoParallel()
# tune parameters
tune_res <- tune_grid(
  tune_wf,
  resamples = folds,
  grid = grid_max
)

best_rmse <- select_best(tune_res, metric ="rmse")
  
  ## Fit Model -------------------------------------
final_rf <- finalize_model(
  tune_spec,
  best_rmse
)
final_wf <- workflow() %>%
  add_recipe(FS_recipe) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  fit(data = training)

  ## Obtain Predictions -----------------------------
  fit_os = predict(final_res,testing)
  
  # Calculate GOF ---------------------------------
  ## Combine Train-Test predictions and Outcomes into a DF 
  res_gof_train_df = data.frame(
    estimate = as.numeric(predict(final_res,training)$.pred),
    truth = unlist(final_res$pre$mold$outcomes[[outcome]]))
  
  res_gof_test_df = data.frame(
    estimate = as.numeric(fit_os$.pred),
    truth = as.numeric(testing[[outcome]]))
  
  ## Calculate GOF ------------------------------
  res_train_rmse = rmse(
    data = res_gof_train_df,
    truth,
    estimate)
  
  res_test_rmse = rmse(
    data = res_gof_test_df,
    truth,
    estimate)
  
  oos_r2 = rsq(data = res_gof_test_df,
               truth = truth,
               estimate = estimate)$.estimate %>% as.numeric()
  is_r2 = rsq(data = res_gof_train_df,
              truth = truth,
              estimate = estimate)$.estimate %>% as.numeric()
  
  # Quintiles ------------------------------
  quintiles = res_gof_test_df %>%
    mutate(quintile = cut(
      truth,
      breaks = quantile(truth, 
      probs = seq(0, 1, 0.20), na.rm = TRUE),
      include.lowest = TRUE,
      labels = paste0("P", 1:5)
    )) %>% 
    # summarise by metric 
    group_by(quintile) %>% 
    summarise(mu_lv = mean(truth),
              mu_p = mean(estimate),
              diff = (mu_lv - mu_p)
    ) %>%
    select(quintile,diff) %>% 
    pivot_wider(names_from = "quintile", values_from = "diff")
  # Return Items ------------------------------
  # Write model
 write_rds(final_res,
  file = here("Final","Output","Ests", paste0("forest_", label, ".rds")))
  # Write Params
  write_rds(best_rmse,
            file = here("Final","Output","Ests", paste0("forest_", label,"_tune_result", ".rds")))
  return(
    cbind(data.frame(
      Model = "Random Forest",
      in_sample_sd_y = is_sd_y,
      oo_sample_sd_y = os_sd_y,
      in_sample_rmse = res_train_rmse$.estimate,
      oo_sample_rmse = res_test_rmse$.estimate,
      n_trees = n_trees,
      is_r2 = is_r2,
      oos_r2 = oos_r2
    ),
    quintiles
    )
  ) 
}

# ------------ Binary ---------------

random_forest_binary = function(outcome,
                         training,
                         testing,
                         FS_recipe,
                         n_trees, 
                         n_folds,
                         grid_max,
                         label
){
  
  tune_spec <- rand_forest(
    mtry = tune(),
    trees = n_trees,
    min_n = tune()
  ) %>%
    set_mode("classification") %>%
    set_engine("ranger", importance = "permutation")
  
  tune_wf <- workflow() %>%
    add_recipe(FS_recipe) %>%
    add_model(tune_spec)
  
  ## Define folds 
  folds = vfold_cv(training,v = n_folds) 
  
  # parallelize
  doParallel::registerDoParallel()
  # tune parameters
  tune_res <- tune_grid(
    tune_wf,
    resamples = folds,
    grid = grid_max
  )
  
  best_auc <- select_best(tune_res, metric ="roc_auc")
  
  ## Fit Model -------------------------------------
  final_rf <- finalize_model(
    tune_spec,
    best_auc
  )
  final_wf <- workflow() %>%
    add_recipe(FS_recipe) %>%
    add_model(final_rf)
  
  final_res <- final_wf %>%
    fit(data = training)
  
  
  # Calculate GOF ---------------------------------
  rf_oos_accuracy = augment(final_res, new_data = testing) %>% 
    accuracy(truth = outcome, estimate = .pred_class)
  # Calculate sensitivity ---------------------------------
  rf_oos_sensitivity = augment(final_res, new_data = testing) %>% 
    sens(truth = outcome, estimate = .pred_class)
  # Calculate precision ---------------------------------
  rf_oos_precision = augment(final_res, new_data = testing) %>% 
    precision(truth = outcome, estimate = .pred_class)
  # AUC 
  rf_oos_auc = augment(final_res, new_data = testing) %>% 
    roc_auc(truth = outcome, .pred_1)
  # Return Items ------------------------------
  # Write model
  write_rds(final_res,
            file = here("Final","Output","Ests", paste0("forest_binary_", label, ".rds")))
  # Write tuning parameters 
  write_rds(best_auc,
            file = here("Final","Output","Ests", paste0("forest_binary_", label,"_tune_result", ".rds")))
  return(
     data.frame(
      Model = "Random Forest Classification",
      oos_accuracy =    rf_oos_accuracy$.estimate,
      oos_sensitivity = rf_oos_sensitivity$.estimate,
      oos_precision =   rf_oos_precision$.estimate,
      oos_auc = rf_oos_auc$.estimate,
      n_trees = n_trees
    )
  ) 
}




