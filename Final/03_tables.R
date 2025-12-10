## Create Tables ## 
# Thelonious Goerz 
# 11/22/25
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
library(tinytable)
library(cowplot)
library(vip)
## ----------------
# Model information
# ------------------ Regression --------- #
ols_sum_0 = read_rds(here("Final","Output","Performance","ols_sum_0.rds"))
ols_sum_1 = read_rds(here("Final","Output","Performance","ols_sum_1.rds"))
ols_sum_2 = read_rds(here("Final","Output","Performance","ols_sum_2.rds"))
lasso_sum_1 = read_rds(here("Final","Output","Performance","lasso_sum_1.rds"))
lasso_sum_2 = read_rds(here("Final","Output","Performance","lasso_sum_2.rds"))
ridge_sum_1 =read_rds(here("Final","Output","Performance","ridge_sum_1.rds"))
ridge_sum_2 =read_rds(here("Final","Output","Performance","ridge_sum_2.rds"))
elastic_sum_1 = read_rds(here("Final","Output","Performance","elastic_sum_1.rds"))
elastic_sum_2 = read_rds(here("Final","Output","Performance","elastic_sum_2.rds"))
tree_sum_1 = read_rds(here("Final","Output","Performance","tree_sum_1.rds"))
tree_sum_2 = read_rds(here("Final","Output","Performance","tree_sum_2.rds"))
forest_sum_1 = read_rds(here("Final","Output","Performance","forest_sum_1.rds"))
forest_sum_2 = read_rds(here("Final","Output","Performance","forest_sum_2.rds"))
boost_sum_1 = read_rds(here("Final","Output","Performance","boost_sum_1.rds"))
boost_sum_2 = read_rds(here("Final","Output","Performance","boost_sum_2.rds"))
# ------------------ Classification --------- #
logit_0 = read_rds(here("Final","Output","Performance","logit_sum_0.rds"))
logit_1 = read_rds(here("Final","Output","Performance","logit_sum_1.rds"))
logit_2 = read_rds(here("Final","Output","Performance","logit_sum_2.rds"))
logit_lasso_1 = read_rds(here("Final","Output","Performance","logit_lasso_sum_1.rds"))
logit_lasso_2 = read_rds(here("Final","Output","Performance","logit_lasso_sum_2.rds"))
logit_ridge_1 = read_rds(here("Final","Output","Performance","logit_ridge_sum_1.rds"))
logit_ridge_2 = read_rds(here("Final","Output","Performance","logit_ridge_sum_2.rds"))
logit_en_1 = read_rds(here("Final","Output","Performance","logit_elastic_sum_1.rds"))
logit_en_2 = read_rds(here("Final","Output","Performance","logit_elastic_sum_2.rds"))
tree_binary_1 = read_rds(here("Final","Output","Performance","binary_tree_sum_1.rds"))
tree_binary_2 = read_rds(here("Final","Output","Performance","binary_tree_sum_2.rds"))
forest_binary_1 = read_rds(here("Final","Output","Performance","binary_forest_sum_1.rds"))
forest_binary_2 = read_rds(here("Final","Output","Performance","binary_forest_sum_2.rds"))
boost_b_1 = read_rds(here("Final","Output","Performance","binary_boost_sum_1.rds"))
boost_b_2 = read_rds(here("Final","Output","Performance","binary_boost_sum_2.rds"))
# Data 
## Read in Data 
data = read_csv(here("Data","project_data.csv"))


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
# test size 
split_d = initial_split(deceased)
train_d = training(split_d,prop = .8)
test_d = testing(split_d)

# ########################################################################
# Make GOF table 

# ########################################################################
### Table 2 - Cont Outcome Comparison Table (Longevity)----------------------------------------------------------------------
# Rename Tree 
tree_sum_1 %<>% mutate(Model = "Tree")
tree_sum_2 %<>% mutate(Model = "Tree")

# Make Table
table_long = rbind(
  ols_sum_0 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)     %>% mutate(`Feature Set` = "Probs"),
  ols_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)     %>% mutate(`Feature Set` = "Covariates"),
  ols_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)     %>% mutate(`Feature Set` = "Covariates + Probs"),
  lasso_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates"),
  lasso_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates + Probs"),
  ridge_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates"),
  ridge_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates + Probs"),
  elastic_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5) %>% mutate(`Feature Set` = "Covariates"),
  elastic_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5) %>% mutate(`Feature Set` = "Covariates + Probs"),
  tree_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)    %>% mutate(`Feature Set` = "Covariates"),
  tree_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)    %>% mutate(`Feature Set` = "Covariates + Probs"),
  forest_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)  %>% mutate(`Feature Set` = "Covariates"),
  forest_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)  %>% mutate(`Feature Set` = "Covariates + Probs"),
  boost_sum_1 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates"),
  boost_sum_2 %>% select(Model,oo_sample_sd_y,oo_sample_rmse,oos_r2,P1,P2,P3,P4,P5)   %>% mutate(`Feature Set` = "Covariates + Probs")
) %>% 
  mutate(
    # calculate percent improvement 
    overall_improvement_rmse = ((oo_sample_rmse - ols_sum_0$oo_sample_rmse) / ols_sum_0$oo_sample_rmse)*100,
    improvement_1 = ((P1 - ols_sum_0$P1) / ols_sum_0$P1)*100,
    improvement_2 = ((P2 - ols_sum_0$P2) / ols_sum_0$P2)*100,
    improvement_3 = ((P3 - ols_sum_0$P3) / ols_sum_0$P3)*100,
    improvement_4 = ((P4 - ols_sum_0$P4) / ols_sum_0$P4)*100,
    improvement_5 = ((P5 - ols_sum_0$P5) / ols_sum_0$P5)*100,
  ) 

table_long %>% 
  select(
    Model,
    `Feature Set`,
    -starts_with("P"),
    `OOS SD(Y)` = oo_sample_sd_y,
    `OOS RMSE` = oo_sample_rmse,
    `OOS $R^2$` = oos_r2,
    `Overall $\\Delta$ RMSE` = overall_improvement_rmse,
    `Q1 $\\Delta$` = improvement_1,
    `Q2 $\\Delta$` = improvement_2,
    `Q3 $\\Delta$` = improvement_3,
    `Q4 $\\Delta$` = improvement_4,
    `Q5 $\\Delta$` = improvement_5
  )  %>%
  datasummary_df(
    title = "Summary of Model Performance (Longevity)",
    output = "tinytable",
    align = "llccccccccc",
    notes = "Q columns refer to quantiles of improvement in RMSE over base OLS.",
    fmt = 3
  ) %>% save_tt(here("Final","Output","03_performance_long.tex"),overwrite = T)

## ------ Visualize --------- ## 
# Absolute 
long_performance = 
  table_long %>%
  mutate(Model = factor(Model, levels = unique(table_long$Model))) %>%
  ggplot(aes(Model, oo_sample_rmse, 
             color = Model,
             shape = `Feature Set`)) +
  geom_point(
    position = position_dodge(width = 0.4),
    size = 3
  ) +
  geom_hline(yintercept = 9.28,
             linetype = "dashed",
             lwd = 1, 
             color = "grey50") +
  theme_cowplot() + 
  theme(legend.position = "bottom") + 
  labs(y = "Improvement in RMSE") + 
    annotate("text", 
             x = 4.5, 
             y = 9.1, 
             label = "OOS SD(Death Age)", 
             size = 4, color = "grey50")
ggsave(long_performance,
       filename = here("Final","Images","long_improvement.jpeg"),
       width = 12,
       height = 5)

# Relative
long_rel = table_long %>%
  filter(`Feature Set` != "Probs") %>%
  mutate(Model = factor(Model, levels = unique(table_long$Model))) %>%
  ggplot(aes(Model, overall_improvement_rmse, 
             color = Model,
             shape = `Feature Set`)) +
  geom_point(
    position = position_dodge(width = 0.4),
    size = 3
  ) + 
  theme_cowplot() + 
  theme(legend.position = "bottom") + 
  geom_hline(yintercept = 0,
             color = "grey50",
             lwd = 1, 
             linetype = "dashed") + 
  annotate("text", 
           x = 4.5, 
           y = -1, 
           label = "Baseline OLS", 
           size = 4, color = "grey50") +
  labs(
    y = "% Improvement in RMSE Over Baseline")
ggsave(long_rel,
       filename = here("Final","Images","long_improvement_rel.jpeg"),
       width = 12,
       height = 5)
# quintile
improvement_q = 
table_long %>%
  filter(`Feature Set` != "Probs" &
           Model %in% c("Random Forest","XGBoost")) %>%
  pivot_longer(cols = starts_with("improvement"),
                      names_to = "q_y",
                      values_to = "value") %>%
  mutate(`Quintile(Y)` = 
  case_when(
    q_y == "improvement_1" ~ 1,
    q_y == "improvement_2" ~ 2,
    q_y == "improvement_3" ~ 3,
    q_y == "improvement_4" ~ 4,
    q_y == "improvement_5" ~ 5
    ),
  Model = factor(Model,
                 levels = c(unique(Model)))
                 ) %>%
  ggplot(aes(`Quintile(Y)`,
             value,
             color = Model,
             shape = `Feature Set`,
             group = `Feature Set`)) +
  geom_point(
    position = position_dodge(width = 0.6),
    size = 3) +
  theme_cowplot() + 
  labs(
    y = "% Improvement in RMSE Over Baseline"
  ) + 
  facet_grid(~Model) + 
  theme(legend.position = "bottom") + 
  geom_hline(yintercept = 0,
             lwd = 1, 
             linetype = "dashed")

ggsave(improvement_q,
       filename = here("Final","Images","long_improvement_q.jpeg"),
       width = 12, 
       height = 5)


### Table 3 - Binary Outcome Comparison Table (Live to 75)----------------------------------------------------------------------
tree_binary_1 %<>% mutate(Model = "Tree")
tree_binary_2 %<>% mutate(Model = "Tree")
forest_binary_1 %<>% mutate(Model = "Random Forest")
forest_binary_2 %<>% mutate(Model = "Random Forest")
boost_b_1 %<>% mutate(Model = "XGBoost")
boost_b_2 %<>% mutate(Model = "XGBoost")

survive_table = 
rbind(
  logit_0 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Probs"), 
  logit_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  logit_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  logit_lasso_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  logit_lasso_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  logit_ridge_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  logit_ridge_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  logit_en_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  logit_en_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  tree_binary_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  tree_binary_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  forest_binary_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  forest_binary_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs"), 
  boost_b_1 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates"), 
  boost_b_2 %>% select(Model,oos_accuracy,oos_sensitivity,oos_precision,oos_auc) %>% mutate(`Feature Set` = "Covariates + Probs") 
) %>% 
  mutate(#oos_auc = 1-oos_auc,
         accuracy_improve =     ((oos_accuracy - logit_0$oos_accuracy) / logit_0$oos_accuracy)*100 ,
         sensitivity_improve =  ((oos_sensitivity - logit_0$oos_sensitivity) / logit_0$oos_sensitivity)*100 ,
         precision_improve =    ((oos_precision - logit_0$oos_precision) / logit_0$oos_precision)*100 ,
         auc_improve =          ((oos_auc - (logit_0$oos_auc)) / (logit_0$oos_auc))*100
        )

# --------- Create Table -----------
survive_table %>% 
  select(
    Model,
    `Feature Set`,
    `AUC`  = oos_auc,
    `Accuracy`  = oos_accuracy,
    `Sensitivity`  = oos_sensitivity,
    `Precision`  = oos_precision,
    `$\\Delta$ AUC`  = auc_improve,
    `$\\Delta$ Accuracy`  = accuracy_improve,
    `$\\Delta$ Sensitivity`  = sensitivity_improve,
    `$\\Delta$ Precision`  = precision_improve
  )  %>%
  datasummary_df(
    title = "Summary of Model Performance (Survival to age 75+)",
    output = "tinytable",
    align = "llcccccccc",
    notes = "Improvement is relative to a logistic regression with subjective probabilities as features",
    fmt = 3
  ) %>% 
  save_tt(here("Final","Output","04_performance_long.tex"),overwrite = T)

## ------ Visualize --------- ## 

# Absolute 
survive_performance = 
  survive_table %>%
  mutate(Model = factor(Model, levels = unique(survive_table$Model))) %>%
  ggplot(aes(Model, oos_auc, 
             color = Model,
             shape = `Feature Set`)) +
  geom_point(
    position = position_dodge(width = 0.4),
    size = 5
  ) +
  theme_cowplot() + 
  labs(
    y = "Absolute Improvement in AUC"
  ) + 
   theme(legend.position = "bottom") + 
    annotate("text", 
             x = 4.5, 
             y = .48, 
             label = "Random Classification", 
             size = 4, color = "grey50") + 
    geom_hline(yintercept = .5,
               lwd = 1, 
               color = "grey50",
               linetype = "dashed")
## all 
survive_improvement = 
  survive_table %>%
  mutate(Model = factor(Model, levels = unique(survive_table$Model))) %>%
  ggplot(aes(Model, auc_improve, 
             color = Model,
             shape = `Feature Set`)) +
  geom_point(
    position = position_dodge(width = 0.4),
    size = 5
  ) +
  geom_hline(yintercept = 0,
             color = "grey50",
             lwd = 1, 
             linetype = "dashed") + 
  annotate("text", 
           x = 4.5, 
           y = -1, 
           label = "Baseline Logit", 
           size = 4, color = "grey50") +
  theme_cowplot() + 
    theme(legend.position = "bottom") + 
  labs(
    y = "Relative Improvement in AUC"
  )


# write figures 
ggsave(long_performance,
       filename = here("Final","Images","long_improvement.jpeg"),
       width = 12,
       height = 4)

ggsave(long_rel,
       filename = here("Final","Images","long_improvement_rel.jpeg"),
       width = 12,
       height = 4)

ggsave(improvement_q,
       filename = here("Final","Images","long_improvement_q.jpeg"),
       width = 12, 
       height = 4)


ggsave(survive_performance,
       filename = here("Final","Images","survive_improvement.jpeg"),
       width = 12, 
       height = 4)

ggsave(survive_improvement,
       filename = here("Final","Images","survive_improvement_rel.jpeg"),
       width = 12, 
       height = 4)

## Claibration Plot 

xgb_bin_mod = read_rds(here("Final","Output","Ests","xgboost_binary_boost1.rds"))

bins <- seq(0, 1, by = 0.1)

df_plot = augment(xgb_bin_mod,new_data = test_d) %>% 
  mutate(
    y_pred_bins =cut(.pred_0, breaks = bins, include.lowest = TRUE),
    live_75 = ifelse(less75 == 1,0,1)
  ) %>% 
  group_by(y_pred_bins) %>%
  summarise(m = mean(live_75,na.rm = T)) %>% 
  mutate(rn = row_number()) 

calib_plot_mort = df_plot %>%
  ggplot(
    aes(x = rn,
        y = m,
        group = 1) 
  ) + 
  geom_line(lwd = 1,
            color = "navy") + 
  geom_segment(aes(
    x    = min(rn), 
    y    = min(m),
    xend = max(rn), 
    yend = max(m)
  ), linetype = "dashed",
  lwd = 1) + 
  scale_x_continuous(
    breaks = df_plot$rn,
    labels = df_plot$y_pred_bins
  ) + 
  theme_cowplot() + 
  labs(x = "Predicted Survival",
       y = "Actual Survival")
  
ggsave(calib_plot_mort,
       filename = here("Final","Images","survive_calib.jpeg"),
       width = 10, 
       height = 5)

### Load models 
linear_forest_2 = read_rds(here("Final","Output","Ests","forest_forest2.rds"))
class_forest_2 = read_rds(here("Final","Output","Ests","xgboost_binary_boost_binary2.rds"))

# Create VIP Plots 

vip_forest = linear_forest_2 %>% 
  vip(n =10) +
  theme_cowplot() + 
  labs(y = "Importance",
       x = "Variable",
       caption = str_wrap("This plot describes the top 10 most relevant features using the permutation importance metric 
       for my preferred random forest model with subjective probabilities and covariates. 
                          Higher values equal greater importance.",100))
vip_boost = class_forest_2 %>% 
  vip(n =10) +
  theme_cowplot() + 
  labs(y = "Importance",
       x = "Variable",
       caption = str_wrap("This plot describes the top 10 most relevant features using the permutation importance metric 
       for my preferred XGBoost model with subjective probabilities and covariates. 
                          Higher values equal greater importance.",100))

# Save VIP plots 
ggsave(vip_forest,
       filename = here("Final","Images","vip_longevity.jpeg"),
       width = 7, 
       height = 5)
ggsave(vip_boost,
       filename = here("Final","Images","vip_survive.jpeg"),
       width = 7, 
       height = 5)
