4## OLS Function ## 
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

# ------------ Cont ---------------------------# 
ols_function = function(outcome,
                        training,
                        testing,
                        FS_recipe
){
  ## SD Y ------------------------------
  is_sd_y = sd(training[[outcome]])
  os_sd_y = sd(testing[[outcome]])
  ## Define Model ------------------------------
  ols = linear_reg(
    mode = "regression",
    engine = "lm"
  ) 
  ## Define Workflow ------------------------------
  ols_workflow = 
    workflow() %>% 
    add_model(ols) %>% 
    add_recipe(FS_recipe)
  
  ## Fit Model -------------------------------------
  fit_ols =
    ols_workflow %>% 
    fit(data = training)
  
  ## Obtain Predictions -----------------------------
  ols_oos = predict(fit_ols,testing)
  
  # Calculate RMSE ---------------------------------
  
  ## Combine Train-Test predictions and Outcomes into a DF 
  ols_gof_train_df = data.frame(
    estimate = as.numeric(predict(fit_ols,training)$.pred),
    truth = unlist(fit_ols$pre$mold$outcomes[[outcome]]))
  
  ols_gof_test_df = data.frame(
    estimate = as.numeric(ols_oos$.pred),
    truth = as.numeric(testing[[outcome]]))
  
  ## Calculate RMSE ------------------------------
  ols_train_rmse = rmse(
    data = ols_gof_train_df,
    truth =  truth,
    estimate =  estimate)
  
  ols_test_rmse = rmse(
    data = ols_gof_test_df,
    truth = truth,
    estimate = estimate)
  
  oos_r2 = rsq(data = ols_gof_test_df,
               truth = truth,
               estimate = estimate)$.estimate %>% as.numeric()
  is_r2 = rsq(data = ols_gof_train_df,
              truth = truth,
              estimate = estimate)$.estimate %>% as.numeric()
  # Quitiles of price  ------------------------------
  quintiles = ols_gof_test_df %>%
    mutate(quintile = cut(
      truth,
      breaks = quantile(truth, probs = seq(0, 1, 0.20), na.rm = TRUE),
      include.lowest = TRUE,
      labels = paste0("P", 1:5)
    )) %>% 
    # summarise by metric 
    group_by(quintile) %>% 
    summarise(mu_lv_ols = mean(truth),
              mu_p_ols = mean(estimate),
              ols_diff = (mu_lv_ols - mu_p_ols)
    ) %>%
    select(quintile,ols_diff) %>% 
    pivot_wider(names_from = "quintile", values_from = "ols_diff")
  
  # Return Items ------------------------------
  return(
    cbind(data.frame(
      Model = "OLS",
      in_sample_sd_y = is_sd_y,
      oo_sample_sd_y = os_sd_y,
      in_sample_rmse = ols_train_rmse$.estimate,
      oo_sample_rmse = ols_test_rmse$.estimate,
      n_predictors = ncol(fit_ols$pre$mold$predictors),
      is_r2 = is_r2,
      oos_r2 = oos_r2
    ),
    # quintiles
    quintiles
    )
  )
}

# -------------------- Binary ---------------- # 
binary_function = function(outcome,
                        training,
                        testing,
                        FS_recipe,
                        label
){
  ## N Y ------------------------------


  ## Define Model ------------------------------
  logit = logistic_reg() %>% 
    set_engine("glm", family = binomial(link = "logit")) 
  ## Define Workflow ------------------------------
  logit_workflow = 
    workflow() %>% 
    add_model(logit) %>% 
    add_recipe(FS_recipe)
  
  ## Fit Model -------------------------------------
  fit_logit =
    logit_workflow %>% 
    fit(data = training)
  
  ## Obtain Predictions -----------------------------
  logit_oos = predict(fit_logit,testing, type = "prob")
  
  # Calculate accuracy --------------------------------
  logit_oos_accuracy = augment(fit_logit, new_data = testing) %>% 
    accuracy(truth = outcome, estimate = .pred_class)
  # Calculate sensitivity ---------------------------------
  logit_oos_sensitivity = augment(fit_logit, new_data = testing) %>% 
    sens(truth = outcome, estimate = .pred_class)
  # Calculate precision ---------------------------------
  logit_oos_precision = augment(fit_logit, new_data = testing) %>% 
    precision(truth = outcome, estimate = .pred_class)
  # AUC 
  logit_oos_auc = augment(fit_logit, new_data = testing) %>% 
    roc_auc(truth = outcome, .pred_1)
  # Return Items ------------------------------
  # Write model
  write_rds(logit_oos,
            file = here("Final","Output","Ests", paste0("logit_os_y", label, ".rds")))
  return(
    data.frame(
      Model = "Logit",
      oos_accuracy =    logit_oos_accuracy$.estimate,
      oos_sensitivity = logit_oos_sensitivity$.estimate,
      oos_precision =   logit_oos_precision$.estimate,
      oos_auc = logit_oos_auc$.estimate,
      n_predictors = ncol(fit_logit$pre$mold$predictors)
    ))
}

