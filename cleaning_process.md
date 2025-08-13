# Further Cleaning of the #TinyTuesday-Preprocessed SO2024 Survey
https://github.com/rfordatascience/tidytuesday/blob/main/data/2024/2024-09-03/readme.md

## Introduction
TinyTuesday, a "weekly community of practice" organized by the Data Science Learning Community, codified the results of StackOverflow's 2024 developer survey to find relationships using the R language.
Using their preprocessed dataset, I set out to use lasso regression to predict [self-reported] salary from survey variables and find positive and negative correlations.
I propose the following requirements for reasonable source records:
1. Developer must give an age and be older than 18
2. Developer must be an active professional (currently employed)
3. Developer must give values for salary, organization size, and years coding (both private and professional)

## Cleaning Operations
### Current Status (main_branch)
Only values with `1`: "I am a developer by profession" are accepted.

### Age (age)
Values 7 and above are dropped: no 'prefer not to say' nor 'age under 18'.

### Remote work (remote_work)
Remote work is re-ordinalized to measure approximate "distance from office", where 0 is in-person, 1 is hybrid, and 2 is remote.
No answer given is assumed to mean in-person, as most jobs used to be.

### Educational level (ed_level)
An ordinal relationship is clarified by remapping original values to approximate levels of education,
ranging from 0 (primary school) to 7 (Ph.D).
```python
edu_mapping = {
    4: 0, # 'Primary/elementary school'
    6: 1, # 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)'
    8: 2, # 'Something else': We can consider this to mean self-taught(?)
    7: 3, # 'Some college/university study without earning a degree'
    1: 4, # 'Associate degree (A.A., A.S., etc.)'
    2: 5, # 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)'
    3: 6, # 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)'
    5: 7, # 'Professional degree (JD, MD, Ph.D, Ed.D, etc.)'
}
```

### Job Descriptions (dev_type)
TinyTuesday's dataset codifies job descriptions using integer indices pointing to labels, but for the purposes of lasso regression, this could be improperly processed as continuous values.
I propose reclassifying roles into eight larger categories used with One-Hot Encoding for improved predictions. As an outlier, "blockchain" retains its own category.
```python
devtypes = [
    "NONE",     # No values at zero
    "Academia", # Academic researcher
    "Blockchain",
    "BackEnd",  # Cloud infrastructure engineer
    "DataScienceAndAI",	# Data or business analyst
    "DataScienceAndAI",	# Data engineer
    "DataScienceAndAI",	# Data scientist or machine learning specialist
    "BackEnd",	# Database administrator
    "FrontEnd",	# Designer
    "DevSupport",	# Developer Experience
    "DevSupport",	# Developer Advocate
    "DataScienceAndAI",	# Developer, AI
    "Backend",	# Developer, back-end
    "Native",	# Developer, desktop or enterprise applications
    "Native",	# Developer, embedded applications or devices
    "FullStack",	# Developer, full-stack
    "FrontEnd",	# Developer, front-end
    "Native",	# Developer, game or graphics
    "Native",	# Developer, mobile
    "DevSupport",	# Developer, QA or test
    "DevSupport",	# DevOps specialist
    "Academia",	# Educator
    "DevSupport",	# Engineer, site reliability
    "Management",	# Engineering manager
    "Native",	# Hardware Engineer
    "DevSupport",	# Marketing or sales professional
    "Management",	# Project manager
    "Management",	# Product manager
    "ResearchAndDevelopment",	# Research & Development role
    "ResearchAndDevelopment",	# Scientist
    "BackEnd",	# Security professional
    "ExecutiveManagement",	# Senior Executive (C-Suite, VP, etc.)
    "BackEnd", # System administrator
]
```

### Organization Headcout (org_size)
Here again we remap the range to make more sense as a continuous set of values.
Perhaps the methods used by TinyTuesday expand all these classifications into one-hot encoding...?
```python
org_mapping = {
    10: 0,# Freelancer
    5: 1, # 2 to 9
    2: 2, # 10 to 19
    6: 3, # 20 to 99
    4: 4, # 100 to 499
    8: 5, # 500 to 999
    1: 6, # 1000 to 4999
    7: 7, # 5000 to 9999
    3: 8  # 10000 or more
}         # The missing "9" was "I don't know."
```

### Years Coding (years_code + years_code_pro)
These values are highly correlated with the age of the developer. A more interesting value: how many years did the developer spend coding BEFORE going professional? Thus, these columns are combined into a new variable, 'pre_pro_years_code', calculated via `years_code_pro - years_code`, defaulting to zero if ONLY `years_code` is provided. If neither `years_code` and `years_code_pro` are provided, the record is dropped.

### AI Sentiment (ai_sent)
Remapped as a function of respondant favorability, such that a positive outlook is positive, and negative is negative. Our interest here is not necessarily whether AI sentiment can predict salary, but rather if highly paid developers are more or less positive towards generative AI as a tool available to them.
```python
ai_sentiment_remapping = {
    6: -2, # Very unfavorable
    3: -1, # Unfavorable
    2: 0,  # Indifferent
    1: 1,  # Favorable
    2: 2,  # Very favorable
}
```

### Is AI a threat to my job? (ai_threat)
For fun, I preserved answers to this question, remapping it as a function of respondant confidence.
```python
ai_threat_remapping = {
    2: -1, # No
    1: 0,  # Not sure
    3: 1.  # Yes
}
```

## Target Variable: Salary as percentage of GNI per capita (comp_total + currency)
It is a well-known fact that developer jobs pay best in the United States of America; the correlation between one's nationality and their salary is not interesting for the purposes of this regression analysis. To identify "highly paid" developers as opposed to "poorly paid" developers, I propose that the target variable be five classifiers identifying developer salary as a percentage of their country or currency's GNI (gross national income) per capita, basically comparing their salary to the average salary in the same currency or country.

This requires first that their currency be converted into USD using the exchange rate in 2024. 
Libraries such as `pyxrate` and `forex-python` provided access to public, free APIs that can perform currency conversions, even historical conversions-- but such API-accessing libraries tend to fail when the API is broken, nor do they work on my company computer. Thus, I had to gather exchange rates into USD, specifically 2024 averages, from exchange-rates.org and save them in a CSV file.

Salaries converted into USD are combined with national GNI per capita to produce a scalar indicating the level of income one receives as opposed to the average income people of their country recieve.

The final range doubles for each integer, starting at 0.25 or less = -3, doubling for each integer upwards:
-3: <= 0.25, -2: <= 0.5, -1: <= 1, 0: <= 2, etc.
```python
from math import ceil, log2
def get_percent_gni_category(p_gni) -> int:
    # Clamp the value to avoid math domain errors
    p_gni = max(p_gni, 0.25)
    value = ceil(log2(p_gni / 0.25)) - 3
    # Clamp the result to the range [-3, 3]
    return max(-3, min(3, value))
```

Can career category, organization size, and education level be used to predict how well a developer's job pays? Let's find out!