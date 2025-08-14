from dataclasses import dataclass
import re
from typing import Any, Callable, List, Optional, Tuple
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

@dataclass
class TrainTestSet:
    for_training: Series
    for_testing: Series

    def __str__(self):
        return "TrainTestSet shapes\n--- Training ---\n{}\n----  Testing ----\n{}".format(
            str(self.for_training), 
            str(self.for_testing)
            )

class App:
    DTYPE_ERROR = "Columns of {} type were found in the DataFrame. Preprocess or remove them before analysis: {}"

    def __init__(self, clean_dataframe: DataFrame, target_var_name: str, test_size=0.3):
        self.data: DataFrame = clean_dataframe.dropna()
        assert(len(self.data) > 1000) # A good number of records is necessary to run analysis
        assert target_var_name in self.data.columns, str(self.data.columns) # Target name should be found amongst columns

        string_dtypes = self.data.select_dtypes(include=['string', 'object'])
        assert string_dtypes.empty, App.DTYPE_ERROR.format("string", string_dtypes.columns)

        self.predictor_names: List[str] = [col for col in self.data.columns if col != target_var_name]
        self.target_var_name: str = target_var_name
        print("CSV loaded. {} predictors of {} found.".format(len(self.predictor_names), target_var_name))

        #pred_train, pred_test, tar_train, tar_test
        tt_series: List[Series]  = train_test_split(self.data[self.predictor_names], self.data[self.target_var_name], test_size=test_size)
        self.prediction_set = TrainTestSet(tt_series[0], tt_series[1])
        self.target_set = TrainTestSet(tt_series[2], tt_series[3])
        print("Prediction " + str(self.prediction_set))
        print("Target " + str(self.target_set))

    def create_random_forest_classifier(self, n_estimators: int = 25):
        return RandomForestClassifier(n_estimators=n_estimators).fit(self.prediction_set.for_training, self.target_set.for_training)

    def create_extra_trees_classifier(self):
        return ExtraTreesClassifier().fit(self.prediction_set.for_training, self.target_set.for_training)
    
    def create_lasso_regression_pipeline(self, cv=10) -> Pipeline:
        pipeline = make_pipeline(StandardScaler(), LassoLarsCV(cv=cv, max_iter=5000, precompute=False))
        pipeline.fit(
                self.prediction_set.for_training, 
                self.target_set.for_training)
        return pipeline

# https://github.com/ousstrk/Stack-Overflow-2024-Survey-Analysis/
df = pd.read_csv("so_2024survey_response_processed.csv")
df = df[~df["country"].isin(['ukraine'])] # Ukraine records are generally outliers in their low salary; possibly due to wartime economic effects
df = df[df['home_gni'].apply(lambda amt: amt > 80000)]
print("Countries used: " + str(df['country'].unique()))
print("Total blockchain devs out of {}: {}".format(len(df), df['DEVTYPE_Blockchain'].sum()))
df = df[~df['DEVTYPE_Blockchain']]
print("Final count: {}".format(len(df)))
df = df.drop(columns=["country", "DEVTYPE_Blockchain", "home_gni"])
app = App(df, "per_gni")

pipeline = app.create_lasso_regression_pipeline()
# For curiosity's sake, we can peek into the machinery here by creating a new dataframe with the scaler
scaler: StandardScaler = pipeline.named_steps['standardscaler']
predictors_scaled = pd.DataFrame(scaler.transform(app.prediction_set.for_training), columns=app.predictor_names)
print(predictors_scaled.describe())

model_ = pipeline.named_steps['lassolarscv']
results = dict(zip(app.predictor_names, model_.coef_))
for name, coef in sorted(results.items(), key=lambda key: key[1]):
    print("{} coefficient:{}{}".format(name, (32-len(name))*" ", coef))

# plot coefficient progression
m_log_alphas = -np.log10(model_.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model_.coef_path_.T)
plt.axvline(-np.log10(model_.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.savefig("regression.png")

# plot mean square error for each fold
m_log_alphascv = -np.log10(model_.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model_.mse_path_, ':')
plt.plot(m_log_alphascv, model_.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model_.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.savefig("mean-squared-error.png")

# MSE from training and test data
train_error = mean_squared_error(app.target_set.for_training, pipeline.predict(app.prediction_set.for_training))
test_error = mean_squared_error(app.target_set.for_testing, pipeline.predict(app.prediction_set.for_testing))
print ('training data MSE: {}'.format(train_error))
print ('test data MSE: {}'.format(test_error))

# R-square from training and test data
rsquared_train = pipeline.score(app.prediction_set.for_training,app.target_set.for_training)
rsquared_test = pipeline.score(app.prediction_set.for_testing,app.target_set.for_testing)
print ('training data R-square: {}'.format(rsquared_train))
print ('test data R-square: {}'.format(rsquared_test))
