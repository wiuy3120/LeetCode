# Test line length
a_supa_long_name_variable_with_detailed_description = [
    "test",
    "line",
    "length",
    "79",
]
a_short_var = ["test", "line", "length", "79"]


import pandas as pd

import numpy as np


# Task 1

# a)

df = pd.read_csv("data.csv")
df = df.drop_duplicates()


# b)
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()))


def handle_outliers(df: pd.DataFrame, column: str):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.25)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    return df[df[column] >= lower_bound & df[column] <= upper_bound]


for col in num_cols:
    df = handle_outliers(df)

summary = df.describe().loc[["mean", "50%", "std"]]

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.c
