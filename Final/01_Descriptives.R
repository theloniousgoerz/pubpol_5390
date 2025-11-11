# Final Project Tables 
# Thelonious Goerz 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rm(list = ls())
library(tidyverse)
library(haven)
library(magrittr)
library(foreign)
library(fixest)
library(modelsummary)
library(cowplot)
library(marginaleffects)
library(broom)
library(readxl)
library(tinytable)
library(here)
library(lme4)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Data 
data_c <- read_csv("~/Academic/Projects/RA/HRS/bequest_beliefs/Data/Cleaned/regression_data.csv")
setwd(here())
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# Make Descriptives 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create Summary table -------------------------------------------------
data_c %>% 
  mutate(Black = ifelse(race == "Black",1,0),
         White = ifelse(race == "White",1,0),
         Other = ifelse(race == "Other",1,0),
         Hispanic = ifelse(hispanic == "Hispanic",1,0),
         Female = ifelse(gender == "Female",1,0),
         `Less than HS` = ifelse(highest_degree == "Less HS",1,0),
         `HS` = ifelse(highest_degree == "HS",1,0),
         `Some College` = ifelse(highest_degree == "Some College",1,0),
         `College` = ifelse(highest_degree == "College",1,0),
         `Advanced Degree` = ifelse(highest_degree == "Advanced Degree",1,0),
         `Wealth (Q1)` = ifelse(wealth_q == 1,1,0),
         `Wealth (Q2)` = ifelse(wealth_q == 2,1,0),
         `Wealth (Q3)` = ifelse(wealth_q == 3,1,0),
         `Wealth (Q4)` = ifelse(wealth_q == 4,1,0),
         `Wealth (Q5)` = ifelse(wealth_q == 5,1,0),
         tot_wealth = tot_wealth/1000,
         tot_nonHousWealth = tot_nonHousWealth/1000,
         tot_stockVal = tot_stockVal/1000,
         Died = died
  ) %>%
  select(Died,
         Age = r_age,
         `Live to Age 75`= age75_plus,
         Black,
         White,
         Other,
         Hispanic,
         Female,
         `Subjective Life Expectancy (75+)` = p_live_75p,
         `Subjective Life Expectancy (80+)` = p_live_80,
         `Less than HS`, 
         `HS`, 
         `Some College`, 
         `College`, 
         `Advanced Degree`, 
         `Wealth (Q1)`,
         `Wealth (Q2)`,
         `Wealth (Q3)`,
         `Wealth (Q4)`,
         `Wealth (Q5)`,
         `Total Wealth ($\\$$1,000s)` = tot_wealth,
         `Total non-Housing Wealth ($\\$$1,000s)` = tot_nonHousWealth,
         `Total Stock Value ($\\$$1,000s)` = tot_stockVal,
         `N Children` = n_children,
         `Mean Days of Contact with Children` = c_avg_contact,
         `Mean Age of Children` = c_avg_age,
         `Sum of Functional Limitations` = sum_fl,
         `Sum of Chronic Conditions` = sum_chronic_cond
  ) %>%
  datasummary(
    All(.)~N + Mean + Min + Max,
    data = . ,
    align = "lcccc",
    notes = "All dollar values are inflation adjusted to 2022 Dollars per the CPI from the Bureau of Labor Statistics.
  Number of contact days with child is the number days over a two-year period.
  Number of observations varies by variable because certain values (e.g., subjective life expectancy) that are only observed for those over a certain age.
  Numbers of observations refer to person-observations over several HRS waves.",
    title = "Descriptive Statistics of Health and Retirement Study Sample",
    threeparttable = T,
    output = "tinytable") %>%
  save_tt(output = here("Final","Output","01_descriptive_table.tex"),
          overwrite = T)

# Create Outcome Table -------------------------------------------------

data.frame(
  Outcome = c("Longevity",
              "Survive to 75"),
  Definition = c("Death age in years",
                 "Whether the individual lives past 75 year of age"),
  Encoding = c("Numeric (continuous)",
               "Numeric (binary)")
) %>% 
  datasummary_df(title = "Outcome Descriptions",
                 output = "tinytable"
                 ) %>% 
  save_tt(output = here("Final","Output","02_outcome_table.tex"),
          overwrite = T)























