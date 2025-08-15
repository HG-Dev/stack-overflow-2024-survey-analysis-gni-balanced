# Introduction
A lasso regression analysis was conducted upon survey results from the 2024 StackOverflow developer survey of developer salaries, history, and many other traits.
The goal of this analysis was to find a subset amongst 19 "features" that could be used to predict the scalar of a developer's salary in comparison to their home country's GNI per capita.

## Why try to predict `salary / GNI per capita`?
An initial conceit of this analysis is that features might be able to predict how "well paid" a developer is in US dollars.
However, because the survey responses come from developers from all over the world, salary as a simple consistent currency would be heavily influenced by the currency's exchange rate into USD. On that subject, we conceive GNI per capita, or the average salary of individuals living in a particular country, to form a "fairly paid" amount for developers in a country. We will refer to this concept as **"pay level"** from here on.

# [Pre-Cleaning Process](cleaning_process.md)
The process by which values from the developer survey were organized into features, and discussion of the features themselves.

# Final Cleaning Steps
It turned out that with too much variation in GNI per capita, the final "percentage of GNI per capita" pay level could not be reliably predicted. A discussion of why this is the case is outside the scope of this analysis, but we may assume that the level of pay developers recieve in any given country can be swayed from everything from local culture to business demands.

The final step taken to get *some* insights out of the dataset was to filter out records from countries with GNI per capita beneath 80000 USD. This left behind repsonses from four countries. It should be noted that most survey responses came from the USA.
```
Countries used: ['united states' 'switzerland' 'luxembourg' 'norway']
```

This left behind 4803 usable records. For processing with LASSO, to remove any bias amongst features, a pipeline with StandardScalar normalization was used.

# Results

Records were randomly split into 70%/30% training / prediction sets. The "LASSO" least angle regression algorithm with k=10 fold validation was used to estimate the model for the training set, and the model was validated using the test set. The change in the cross validation average squared error at each setp was used to identify the best subset of predictor features.

Figure 1. Change in the validation mean square error at each step
![](images/lasso-mean-squared-error.png)
![](images/lasso-regression.png)
Of the 18 predictor features, 16 were shown to have some level of influence on the predicted salary vs. GNI per capita scalar.

## Negative Correlations
Some features were shown to have a negative association with pay level. It comes as no surprise that careers in academia had the strongest negative association (-0.046). Sentiment regarding generative AI as a threat to job safety also had a negative association (-0.046), perhaps due to a recognition on the part of developers that their lower-paying job was in danger of being automated.

## Positive Correlations
Organization size had the largest positive pay level (0.21) followed by age (0.18). In other words, the larger the organization, the greater the pay-- not a big surprise for the large corporations of high GNI per capita countries. Surprisingly, the ability to work remotely also saw higher pay amounts (0.16), although there may be correlation between remote work and organization size. The highest correlation between developer career type and pay was management (0.09), even moreso than executive management (negligable). Perhaps C-suite executives that can call themselves developers aren't so specialized as to lead major corporations.


```
ai_threat coefficient:                       -0.04660440958999441
DEVTYPE_Academia coefficient:                -0.04650608511561227
DEVTYPE_FrontEnd coefficient:                -0.03753334662462475
DEVTYPE_FullStack coefficient:               -0.004959473714949047
DEVTYPE_ExecutiveManagement coefficient:     -0.002626863216234048
DEVTYPE_Native coefficient:                  0.0
DEVTYPE_ResearchAndDevelopment coefficient:  0.0
ai_sent coefficient:                         0.0017450440415792208
years_code_pre_pro coefficient:              0.015822258660326733
DEVTYPE_DataScienceAndAI coefficient:        0.02485905839726543
DEVTYPE_DevSupport coefficient:              0.05075454827595284
DEVTYPE_BackEnd coefficient:                 0.08955136831731532
ed_level coefficient:                        0.09006423713935299
DEVTYPE_Management coefficient:              0.09296773877707698
remote_work coefficient:                     0.16573055142363452
age coefficient:                             0.18583497099144833
org_size coefficient:                        0.2131686797413075
training data MSE: 0.9330953296330019
test data MSE: 1.0792330059806485
training data R-square: 0.16425114803147134
test data R-square: 0.15103237120463808
```