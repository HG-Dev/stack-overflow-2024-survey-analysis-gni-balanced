from dataclasses import dataclass
import re
from typing import Any, Callable, List, Optional, Tuple
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
from scipy.spatial.distance import cdist

@dataclass
class TrainTestSet:
    for_training: np.ndarray | DataFrame
    for_testing: np.ndarray | DataFrame

    def __str__(self):
        return "TrainTestSet -- {} rows for training {}, {} rows for testing {}".format(
            len(self.for_training), self.for_training.mean(),
            len(self.for_testing), self.for_testing.mean()
            )

class App:
    DTYPE_ERROR = "Columns of {} type were found in the DataFrame. Preprocess or remove them before analysis: {}"

    def __init__(self, clean_dataframe: DataFrame, target_var_names: Optional[List[str]], test_size=0.3, seed: Optional[int] = None):
        self.data: DataFrame = clean_dataframe.dropna()
        self.original_data: DataFrame = self.data.copy()
        self.seed = seed
        assert(len(self.data) > 1000) # A good number of records is necessary to run analysis
        assert target_var_names is None or all(t in self.data.columns for t in target_var_names), str(self.data.columns) # Target name should be found amongst columns

        string_dtypes = self.data.select_dtypes(include=['string', 'object'])
        assert string_dtypes.empty, App.DTYPE_ERROR.format("string", string_dtypes.columns)

        self.feature_names: List[str] = [col for col in self.data.columns if target_var_names is None or not col in target_var_names]
        self.target_names: Optional[List[str]] = target_var_names
        print("CSV loaded. {} features of {} found.".format(len(self.feature_names), target_var_names or "dataset"))

        #pred_train, pred_test, tar_train, tar_test
        # Features should be be scaled for standard deviation = 1 and mean = 0
        datasets = [StandardScaler().fit_transform(self.data[self.feature_names])]
        if target_var_names:
            datasets.append(self.data[target_var_names])

        tt_series: List[Series]  = train_test_split(*datasets, test_size=test_size, random_state=seed)
        for thing in tt_series:
            print("train_test_split created a " + str(type(thing)))
        self.model_set = TrainTestSet(tt_series[0], tt_series[1])
        print("Model " + str(self.model_set))
        if len(tt_series) >= 4:
            self.target_set = TrainTestSet(tt_series[2], tt_series[3])
            print("Target " + str(self.target_set))

    def create_random_forest_classifier(self, n_estimators: int = 25):
        return RandomForestClassifier(n_estimators=n_estimators).fit(self.model_set.for_training, self.target_set.for_training)

    def create_extra_trees_classifier(self):
        return ExtraTreesClassifier().fit(self.model_set.for_training, self.target_set.for_training)

    # This should be redone so as to not return a pipeline    
    # def create_lasso_regression_pipeline(self, cv=10) -> Pipeline:
    #     pipeline = make_pipeline(StandardScaler(), LassoLarsCV(cv=cv, max_iter=5000, precompute=False))
    #     pipeline.fit(
    #             self.model_set.for_training, 
    #             self.target_set.for_training)
    #     return pipeline
    
    def create_optimal_kmeans_cluster_model(self, min_clusters=3, max_clusters=10, view_plt=False) -> KMeans:
        max_clusters = max(max_clusters, 3) # Need at least three to perform second derivative
        models: List[KMeans] = []
        mean_distances: List[float] = []
        k_range = range(min_clusters, max_clusters)
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.seed)
            model.fit(self.model_set.for_training)
            models.append(model)
            distances = cdist(self.model_set.for_training, model.cluster_centers_, 'euclidean')
            min_distances = np.min(distances, axis=1)
            mean_distances.append(np.mean(min_distances))

        second_derivative = np.diff(mean_distances, 2)
        elbow_k: int = int(np.argmax(second_derivative) + 2) # Two added to align with original indices

        if view_plt:
            plt.figure(figsize=(8,5))
            plt.plot(k_range, mean_distances, marker='o')
            plt.scatter(elbow_k, mean_distances[elbow_k-min_clusters], color='red', zorder=5, label='Elbow Point')
            plt.annotate(
                f'Elbow at k={elbow_k}', xy=(float(elbow_k), mean_distances[elbow_k-min_clusters]),
                xytext=(elbow_k + 0.5, mean_distances[elbow_k-min_clusters] + 0.02),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10, color='red')
            plt.title('Elbow Method for Optimal k')
            plt.xlabel('Cluster Count (k)')
            plt.ylabel('Mean Distance to Closest Centroid')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return models[elbow_k - min_clusters] # Find k with model.n_clusters
    
    def get_cluster_canonical_variables(self, model: KMeans, view_plt: bool=False) -> Tuple[plt.ndarray, plt.ndarray]:
        cluster_assignments = model.predict(app.model_set.for_training)
        pca = PCA(n_components=2)
        plot_columns = pca.fit_transform(app.model_set.for_training)
        #[:, n] means all rows in column 'n'
        principal_component_1 = plot_columns[:, 0]
        principal_component_2 = plot_columns[:, 1]

        if view_plt:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                principal_component_1, 
                principal_component_2, 
                c=cluster_assignments, 
                cmap='viridis',
                s=50)
            plt.legend(*scatter.legend_elements(), loc="lower left", title="Clusters")
            plt.xlabel('Canonical Variable 1')
            plt.ylabel('Canonical Variable 2')
            plt.title(f'PCA Scatterplot for {model.n_clusters} Clusters')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return (principal_component_1, principal_component_2)


# https://github.com/ousstrk/Stack-Overflow-2024-Survey-Analysis/
df = pd.read_csv("so_2024survey_response_processed.csv")
df = df[~df["country"].isin(['ukraine'])] # Ukraine records are generally outliers in their low salary; possibly due to wartime economic effects
df = df[df['home_gni'].apply(lambda amt: amt > 80000)]
df = df[~df['DEVTYPE_Blockchain']]
print("Record count: {}".format(len(df)))
for col in df.columns:
    if 'DEVTYPE' in col:
        selection = df[df[col]]
        print("# of {}: {} or {} (min={} max={})".format(col, df[col].sum(), len(selection), selection['per_gni'].min(), selection['per_gni'].max()))
ai_sentiment = df['ai_sent']
df = df.drop(columns=["country", "DEVTYPE_Blockchain", "home_gni"])
test_size = 0.01
seed = 1001
app = App(df, ["ai_sent", "comp_usd"], test_size=0.01, seed=seed)

cluster_model = app.create_optimal_kmeans_cluster_model(3, 12)
canx, cany = app.get_cluster_canonical_variables(cluster_model, True)
results = pd.DataFrame(app.model_set.for_training, columns=app.feature_names)
# Validate using ai_sent
print("Adding target columns:\n" + str(app.target_set.for_training))
if app.target_names:
    for col in app.target_names:
        print(f"Adding {col} column to results: {app.target_set.for_training[col]}")
        results[col] = app.target_set.for_training[col]

results['cluster'] = cluster_model.labels_
# Creates a dataframe with MANY columns
print(str(results['comp_usd'].describe()))
grouped_results = results.groupby('cluster').describe()
useful_cols = pd.MultiIndex.from_tuples([col for col in grouped_results.columns if '%' not in col[1] and col[1] != 'count'])
grouped_results = grouped_results[useful_cols]
print(str(grouped_results))
#excess_cols = pd.MultiIndex.from_tuples([col for col in grouped_results.columns if '%' in col[1] or col[1] == 'count'])
# for col in excess_cols:
#     grouped_results.drop(columns=col)
#print(str(grouped_results[excess_cols[0]].describe()))
#grouped_results.drop(columns=excess_cols)
#print("{} excess columns; {} final column count".format(len(excess_cols), len(grouped_results.columns)))
grouped_results.to_csv('cluster-results.csv')

print("--- Cluster Info exported to cluster-results.csv --- ")