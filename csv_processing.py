from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Callable, Iterator, List, Optional, Tuple
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

EDU_MAPPING = {
    4: 0, # 'Primary/elementary school'
    6: 1, # 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)'
    8: 2, # 'Something else': We can consider this to mean self-taught(?)
    7: 3, # 'Some college/university study without earning a degree'
    1: 4, # 'Associate degree (A.A., A.S., etc.)'
    2: 5, # 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)'
    3: 6, # 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)'
    5: 7, # 'Professional degree (JD, MD, Ph.D, Ed.D, etc.)'
}

DEVTYPES = [
    None,     # No values at zero
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
    None, # Student
    "BackEnd", # System administrator
]

ORG_MAPPING = {
    10: 0,# Freelancer
    5: 1, # 2 to 9
    2: 2, # 10 to 19
    6: 3, # 20 to 99
    4: 4, # 100 to 499
    8: 5, # 500 to 999
    1: 6, # 1000 to 4999
    7: 7, # 5000 to 9999
    3: 8  # 10000 or more
    # 9 was "I don't know"
}

AI_THREAT_MAPPING = {
    2: -1, # No
    1: 0,  # Not sure
    3: 1.  # Yes
}

AI_SENTIMENT_MAPPING = {
    6: -2, # Very unfavorable
    3: -1, # Unfavorable
    4: 0,  # Unsure
    2: 0,  # Indifferent
    1: 1,  # Favorable
    5: 2,  # Very favorable
}

COUNTRY_MAPPING = {
    "Democratic People's Republic of Korea": "North Korea",
    "Republic of Korea": "South Korea",
    "United States of America": "United States"
}

def SimplifyCountryName(fullname: str):
    if isinstance(fullname, str) is False:
        return None
    name = fullname.lower()
    name = re.split(r'[,(]', name)[0].strip()
    return name

usd_df = pd.read_csv("exchange-rates-dot-org_usd.csv")
USD_EXCHANGE_RATES = dict(zip(usd_df['currency'], usd_df['usd_rate']))
del usd_df

gni_df = pd.read_csv("world-bank_gni_per_capita.csv")
GNI_PER_CAPITA = dict(zip(gni_df['CountryName'].apply(SimplifyCountryName), gni_df['GniPerCapita']))
COUNTRY_INCOME_LEVELS = [
    GNI_PER_CAPITA['low income'],
    GNI_PER_CAPITA['lower middle income'],
    GNI_PER_CAPITA['low & middle income'],
    GNI_PER_CAPITA['middle income'],
    GNI_PER_CAPITA['upper middle income'],
    GNI_PER_CAPITA['high income']
]
del gni_df

def get_income_category(country):
    gni_per_capita = GNI_PER_CAPITA[country]
    for idx in range(len(COUNTRY_INCOME_LEVELS)):
        if gni_per_capita > COUNTRY_INCOME_LEVELS[idx]:
            return idx - 1

class CsvDataFramePreprocessor:
    def __init__(self, csv_filepath: str):
        self._filepath = csv_filepath
        self._df = pd.read_csv(csv_filepath)
        self._touched_columns: set[str] = set()
        print("Opened CSV with {} records. Available columns:\n{}".format(
            len(self._df), self._df.columns))

    def filter_include(self, column_name: str, good_values: List, ignore_na=False):
        self._touched_columns.add(column_name)
        prev_size = len(self._df)
        print("Removing records for col={} outside acceptable values: {}".format(
            column_name, good_values
        ))
        print("Records with the following values will be removed: {}".format(
            self._df[~self._df[column_name].isin(good_values)][column_name].unique()
        ))

        if ignore_na:
            self._df = self._df[self._df[column_name].isna() | self._df[column_name].isin(good_values)]
        else:
            self._df = self._df[self._df[column_name].isin(good_values)]
        print("{} records removed.".format(prev_size - len(self._df)))

    def filter_exclude(self, column_name: str, bad_values: List, ignore_na=False):
        self._touched_columns.add(column_name)
        prev_size = len(self._df)
        msg = "" if ignore_na else "NA or "
        print("Removing records for col={} which have {}values: {}".format(
            column_name, msg, bad_values
        ))
        if ignore_na is not True:
            self._df[column_name].dropna()
        self._df = self._df[~self._df[column_name].isin(bad_values)]
        print("{} records removed.".format(prev_size - len(self._df)))

    def filter_predicate(self, column_name: str, predicate: Callable[[Any], bool], ignore_na=False):
        self._touched_columns.add(column_name)
        prev_size = len(self._df)
        print("Removing records for col={} using predicate filter".format(column_name))
        if ignore_na:
            self._df = self._df[self._df[column_name].apply(predicate) | self._df[column_name].isna()]
        else:
            self._df = self._df[self._df[column_name].apply(predicate)]
        print("{} records removed.".format(prev_size - len(self._df)))

    def create_series_from_max(self, columns: List[str]) -> Series:
        sizediff = len(self._df)
        self._df.dropna(subset=columns, how='all')

        sizediff -= len(self._df)
        if sizediff > 0:
            print("Creating a series using max from {} columns removed {} records.".format(
                columns, sizediff))
        return self._df[columns].max(axis=1, skipna=True)

    def create_series_from_min(self, columns: List[str]) -> Series:
        sizediff = len(self._df)
        self._df.dropna(subset=columns, how='all')
        
        sizediff -= len(self._df)
        if sizediff > 0:
            print("Creating a series using min from {} columns removed {} records.".format(
                columns, sizediff))
        return self._df[columns].min(axis=1, skipna=True)

    def drop_columns(self, columns: List[str]):
        for column in columns:
            if column in self._touched_columns:
                self._touched_columns.remove(column)
        print("Removing columns {} from dataframe.".format(columns))
        self._df.drop(columns=columns)

    def add_column(self, col_name: str, series: Series):
        print("Adding new column {} with {} unique values to dataframe.".format(
            col_name, len(series.unique())))
        self._touched_columns.add(col_name)
        self._df[col_name] = series

    def drop_untouched_columns(self):
        prev_count = len(self._df.columns)
        cols = list(self._touched_columns)
        print("Removing all columns except {} from dataframe.".format(cols))
        self._df = self._df[cols]
        print("Removed {} untouched columns from dataframe.".format(
            prev_count - len(self._df.columns)))

    def remap(self, column_name: str, remap_dict: dict, na_value: Optional[Any] = None, strict: bool = True):
        self._touched_columns.add(column_name)
        print("Preparing to remap values for col={} with {} unique values.".format(
            column_name, len(self._df[column_name].unique())
        ))

        if strict is False:
            self._df[column_name] = self._df[column_name].apply(lambda value: value if value not in remap_dict else remap_dict[value])
            if na_value:
                self._df[column_name] = self._df[column_name].fillna(na_value)
            return

        if na_value:
            self.filter_include(column_name, [key for key in remap_dict.keys()], ignore_na=True)
            self._df[column_name] = self._df[column_name].map(remap_dict, na_action="ignore")
            self._df[column_name] = self._df[column_name].fillna(na_value)
        else:
            self.filter_include(column_name, [key for key in remap_dict.keys()], ignore_na=False)
            self._df[column_name] = self._df[column_name].map(remap_dict)

    def one_hot_encode(self, column_name: str, column_prefix: str):
        print("Converting string entries within {} into one-hot columns.".format(column_name))
        self._df[column_name] = self._df[column_name].apply(CsvDataFramePreprocessor.clean_pascal)
        self._df.dropna(subset=[column_name])
        self._df = pd.get_dummies(self._df, columns=[column_name], prefix=column_prefix)
        if column_name in self._touched_columns:
            self._touched_columns.remove(column_name)

        dummies = [col for col in self._df.columns if col.startswith(column_prefix + '_')]
        for dummy_col in dummies:
            self._touched_columns.add(dummy_col)
        print("Added {} to dataframe in place of {}.".format(dummies, column_name))

    def create_series_from_selection(self, columns: List[str] | str, delegate: Callable[[Series], Any] | Callable[[Any], Any]):
        if isinstance(columns, str):
            return self._df[columns].apply(delegate)
        else:
            return self._df[columns].apply(delegate, axis=1)
    
    def overwrite_series_from_selection(self, columns: List[str] | str, delegate: Callable[[Series], Any] | Callable[[Any], Any]):
        if isinstance(columns, str):
            self._df[columns] = self._df[columns].apply(delegate)
        else:
            self._df[columns] = self._df[columns].apply(delegate, axis=1)
        
    def export_unique_values(self, col_name: str, modifier: Callable[[str], str] = lambda x: x):
        self._df[col_name].dropna().apply(modifier).unique().tofile(col_name + ".txt", "\n")

    @staticmethod
    def clean_pascal(value):
        if not value or pd.isna(value):
            return None
        # Remove special characters and convert to PascalCase
        cleaned = re.sub(r'[^a-zA-Z0-9 ]', ' ', value)
        return ''.join(word[:1].upper() + word[1:] for word in cleaned.split())

    def save(self):
        print("Saving CSV with {} records.".format(len(self._df)))
        self._df = self._df[sorted(self._df.columns)]
        self._df.to_csv(self._filepath.replace('.csv', '_processed.csv'), index=False)

sal_database = pd.read_csv("exchange-rates-dot-org_usd.csv")
gni_database = pd.read_csv("world-bank_gni_per_capita.csv")
preprocessor = CsvDataFramePreprocessor("so_2024survey_response_ttraw.csv")
preprocessor.filter_include("main_branch", [1])
preprocessor.filter_exclude("age", [7, 8])
preprocessor.filter_predicate("comp_total", lambda amt: amt is not None and isinstance(amt, float) and amt > 0) # mostly just get rid of na values
preprocessor.filter_predicate("country", lambda name: name is not None and isinstance(name, str))
preprocessor.remap("country", COUNTRY_MAPPING, strict=False)
preprocessor.overwrite_series_from_selection("country", SimplifyCountryName)
preprocessor.filter_include("country", list(GNI_PER_CAPITA.keys()))
preprocessor.filter_predicate("currency", lambda currency: isinstance(currency, str) is True and currency[:3] in USD_EXCHANGE_RATES.keys())
preprocessor.overwrite_series_from_selection("currency", lambda currency: currency[:3])
preprocessor.remap("remote_work", {2: 0, 1: 1, 3: 2}, na_value=0)
preprocessor.remap("ed_level", EDU_MAPPING)
years_code_pro = preprocessor.create_series_from_min(["years_code", "years_code_pro"])
years_code_amateur = preprocessor.create_series_from_max(["years_code", "years_code_pro"])
years_code_amateur = years_code_amateur.sub(years_code_pro)
assert(years_code_amateur.min() == 0)
preprocessor.add_column("years_code_pre_pro", years_code_amateur)
preprocessor.drop_columns(["main_branch", "years_code", "years_code_pro"])
preprocessor.remap("dev_type", dict([pair for pair in enumerate(DEVTYPES) if pair[1]]))
preprocessor.one_hot_encode("dev_type", "DEVTYPE")
preprocessor.remap("org_size", ORG_MAPPING)
preprocessor.remap("ai_sent", AI_SENTIMENT_MAPPING, na_value=0)
preprocessor.remap("ai_threat", AI_THREAT_MAPPING, na_value=0)

def convert_to_usd(amt, currency):
    return round(amt * USD_EXCHANGE_RATES[currency], 2)

comp_usd = preprocessor.create_series_from_selection(
    ["comp_total", "currency"], 
    lambda row: convert_to_usd(row["comp_total"], row["currency"]))
preprocessor.add_column("comp_usd", comp_usd)

def get_percent_gni(country: str, comp_usd: float):
    assert(country is not None)
    assert(comp_usd is not None)
    result = round(comp_usd / GNI_PER_CAPITA[country], 2)
    assert(result is not None)
    return result

percent_gni = preprocessor.create_series_from_selection(
    ["country", "comp_usd"],
    lambda row: get_percent_gni(row["country"], row["comp_usd"]))

from math import ceil, log2
def get_percent_gni_category(p_gni) -> int:
    # Clamp the value to avoid math domain errors
    p_gni = max(p_gni, 0.25)
    value = ceil(log2(p_gni / 0.25)) - 3
    # Clamp the result to the range [-3, 3]
    return max(-3, min(3, value))

assert(percent_gni.notna().all())

preprocessor.add_column("per_gni", percent_gni)
preprocessor.add_column("per_gni_category", percent_gni.apply(get_percent_gni_category))

preprocessor.add_column(
    "gni_class", 
    preprocessor.create_series_from_selection("country", get_income_category))

preprocessor.drop_columns(["currency", "comp_total", "comp_usd"])
preprocessor.drop_untouched_columns()

preprocessor.save()
