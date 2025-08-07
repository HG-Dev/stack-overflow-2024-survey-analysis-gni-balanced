from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Callable, Iterator, List, Optional, Tuple
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
from forex_python.converter import CurrencyRates
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from forex_python.converter import CurrencyRates

c = CurrencyRates()
df = pd.read_csv("download_exchange_rates.csv")

df["usd_rate"] = df["currency"].apply(lambda ccode: c.convert(ccode, "usd", 1, datetime(2025, 1, 1)))
