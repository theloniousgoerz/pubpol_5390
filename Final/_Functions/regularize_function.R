## Regularize Function ## 
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
regularize_function = function(outcome,
                               training,
                               testing,
                               FS_recipe,
                               type,
                               n_folds,
                               grid_max
){
  ## SD Y ------------------------------
  is_sd_y = sd(training[[outcome]])
  os_sd_y = sd(testing[[outcome]])
  ## Tune Parameters ------------------------------
  ## Pick Model
  
  ## Define folds 
  folds = vfold_cv(training,v = n_folds) 
  
  if (type == "LASSO"){
    model_tune = linear_reg(
      mode = "regression",
      engine = "glmnet",
      mixture = 1, 
      penalty = tune()
    ) 
    # Cross-Validate
    cv_lambda = tune_grid(model_tune, 
                          FS_recipe,
                          resamples = folds, 
                          tune = penalty,
                          grid = expand.grid(penalty = 0:grid_max))
    
    # Pick lambda # 
    best_penalty = as.numeric(show_best(cv_lambda)[1,"penalty"])
    best_mixture = 1
    
  } else if (type == "Ridge"){
    model_tune = linear_reg(
      mode = "regression",
      engine = "glmnet",
      mixture = 0, 
      penalty = tune()
    ) 
    # Cross-Validate
    cv_lambda = tune_grid(model_tune, 
                          FS_recipe,
                          resamples = folds, 
                          tune = penalty,
                          grid = expand.grid(penalty = 0:grid_max))
    
    # Pick lambda # 
    best_penalty = as.numeric(show_best(cv_lambda)[1,"penalty"])
    best_mixture = 0
    
    
  } else if (type == "Elastic"){
    model_tune = linear_reg(
      mode = "regression",
      engine = "glmnet",
      mixture = tune(), 
      penalty = tune()
    ) 
    # Cross-Validate 
    
    cv_pick = tune_grid(model_tune, 
                        FS_recipe,
                        resamples = folds, 
                        tune = c(penalty,mixture),
                        grid = expand.grid(penalty = 0:grid_max,
                                           mixture = c(.25,.5,.75)))
    
    # Pick lambda # 
    best_penalty = as.numeric(show_best(cv_pick)[1,"penalty"])
    best_mixture = as.numeric(show_best(cv_pick)[1,"mixture"])
  }
  
  # ------------------------------------------------------------
  # Declare Model # 
  regularized_model = linear_reg(
    mode = "regression",
    engine = "glmnet",
    mixture = best_mixture, 
    penalty = best_penalty
  ) 
  ## Define Workflow ------------------------------
  rglr_workflow = 
    workflow() %>% 
    add_model(regularized_model) %>% 
    add_recipe(FS_recipe)
  
  ## Fit Model -------------------------------------
  fit_rglr =
    rglr_workflow %>% 
    fit(data = training)
  
  ## Obtain Predictions -----------------------------
  rglr_oos = predict(fit_rglr,testing)
  
  tidy_coefficients <- fit_rglr %>%
    extract_fit_parsnip() %>%
    tidy()
  # Filter for non-zero coefficients (selected predictors)
  n_terms <- tidy_coefficients %>%
    filter(estimate != 0) %>%
    pull(term) %>% 
    length() %>% 
    as.numeric()
  
  # Calculate GOF ---------------------------------
  ## Combine Train-Test predictions and Outcomes into a DF 
  rglr_gof_train_df = data.frame(
    estimate = as.numeric(predict(fit_rglr,training)$.pred),
    truth = unlist(fit_rglr$pre$mold$outcomes[[outcome]]))
  
  rglr_gof_test_df = data.frame(
    estimate = as.numeric(rglr_oos$.pred),
    truth = as.numeric(testing[[outcome]]))
  
  ## Calculate GOF ------------------------------
  rglr_train_rmse = rmse(
    data = rglr_gof_train_df,
    truth,
    estimate)
  
  rglr_test_rmse = rmse(
    data = rglr_gof_test_df,
    truth,
    estimate)
  
  oos_r2 = rsq(data = rglr_gof_test_df,
               truth = truth,
               estimate = estimate)$.estimate %>% as.numeric()
  is_r2 = rsq(data = rglr_gof_train_df,
              truth = truth,
              estimate = estimate)$.estimate %>% as.numeric()
  
  # Quintiles ------------------------------
  quintiles = rglr_gof_test_df %>%
    mutate(quintile = cut(
      truth,
      breaks = quantile(truth, probs = seq(0, 1, 0.20), na.rm = TRUE),
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
  return(
    cbind(data.frame(
      Model = {{type}},
      in_sample_sd_y = is_sd_y,
      oo_sample_sd_y = os_sd_y,
      in_sample_rmse = rglr_train_rmse$.estimate,
      oo_sample_rmse = rglr_test_rmse$.estimate,
      n_predictors = n_terms,
      mixture = best_mixture,
      penalty = best_penalty,
      is_r2 = is_r2,
      oos_r2 = oos_r2
    ),
    quintiles
    )
  ) 
}
# ####################################################################################
# Binary 

#########################################################################################################

regularize_function_binary = function(outcome,
                               training,
                               testing,
                               FS_recipe,
                               label,
                               type,
                               n_folds,
                               grid_max
){
  
  ## Tune Parameters ------------------------------
  ## Pick Model
  ## Define folds 
  folds <- vfold_cv(training, v = n_folds)
  
  ## Penalty grid
  lambda_grid <- tibble(
    penalty = 10^seq(-4, 1, length.out = 50)
  )
  
  if (type == "LASSO") {
    
    model_tune <- logistic_reg(
      mode = "classification",
      engine = "glmnet",
      mixture = 1,
      penalty = tune()
    )
    
    wf <- workflow() %>%
      add_model(model_tune) %>%
      add_recipe(FS_recipe)
    
    cv_lambda <- tune_grid(
      wf,
      resamples = folds,
      grid = lambda_grid
    )
    
    best_term <- select_best(cv_lambda, metric = "roc_auc")
    final_lg <- finalize_model(model_tune, best_term)
    b_mixture = 1
    b_lambda = show_best(cv_lambda, metric = "roc_auc")[1,"penalty"]
    
  } else if (type == "Ridge") {
    
    model_tune <- logistic_reg(
      mode = "classification",
      engine = "glmnet",
      mixture = 0,
      penalty = tune()
    )
    
    wf <- workflow() %>%
      add_model(model_tune) %>%
      add_recipe(FS_recipe)
    
    cv_lambda <- tune_grid(
      wf,
      resamples = folds,
      grid = lambda_grid
    )
    
    best_term <- select_best(cv_lambda, metric = "roc_auc")
    final_lg <- finalize_model(model_tune, best_term)
    b_mixture = 0
    b_lambda = show_best(cv_lambda, metric = "roc_auc")[1,"penalty"]
    
  } else if (type == "Elastic") {
    
    elastic_grid <- expand.grid(
      penalty = lambda_grid$penalty,
      mixture = c(.25, .50, .75)
    )
    
    model_tune <- logistic_reg(
      mode = "classification",
      engine = "glmnet",
      mixture = tune(),
      penalty = tune()
    )
    
    wf <- workflow() %>%
      add_model(model_tune) %>%
      add_recipe(FS_recipe)
    
    cv_lambda <- tune_grid(
      wf,
      resamples = folds,
      grid = elastic_grid
    )
    
    best_term <- select_best(cv_lambda, metric = "roc_auc")
    final_lg <- finalize_model(model_tune, best_term)
    b_mixture = show_best(cv_lambda, metric = "roc_auc")[1,"mixture"]
    b_lambda =  show_best(cv_lambda, metric = "roc_auc")[1,"penalty"]
  }
  
  # ------------------------------------------------------------
  # Declare Model # 
  final_wf <- workflow() %>%
    add_recipe(FS_recipe) %>%
    add_model(final_lg)
  ## Fit Model -------------------------------------
  final_res <- final_wf %>%
    fit(data = training)

  ## Obtain Predictions -----------------------------
  logit_oos = predict(final_res,testing, type = "prob")
  
  tidy_coefficients <- final_res %>%
    extract_fit_parsnip() %>%
    tidy()
  # Filter for non-zero coefficients (selected predictors)
  n_terms <- tidy_coefficients %>%
    filter(estimate != 0) %>%
    pull(term) %>% 
    length() %>% 
    as.numeric()
  
  # Calculate accuracy --------------------------------
  logit_oos_accuracy = augment(final_res, new_data = testing) %>% 
    accuracy(truth = outcome, estimate = .pred_class)
  # Calculate sensitivity ---------------------------------
  logit_oos_sensitivity = augment(final_res, new_data = testing) %>% 
    sens(truth = outcome, estimate = .pred_class)
  # Calculate precision ---------------------------------
  logit_oos_precision = augment(final_res, new_data = testing) %>% 
    precision(truth = outcome, estimate = .pred_class)
  # AUC 
  logit_oos_auc = augment(final_res, new_data = testing) %>% 
    roc_auc(truth = outcome, .pred_1)
  # Return Items ------------------------------
  # Write model
  write_rds(logit_oos,
            file = here("Final","Output","Ests", paste0(type,"logit_os_y", label, ".rds")))
  return(
    data.frame(
      Model = {{type}},
      oos_accuracy =    logit_oos_accuracy$.estimate,
      oos_sensitivity = logit_oos_sensitivity$.estimate,
      oos_precision =   logit_oos_precision$.estimate,
      oos_auc = logit_oos_auc$.estimate,
      n_predictors = n_terms,
     mixture = b_mixture,
     penalty = b_lambda
    ))
}
