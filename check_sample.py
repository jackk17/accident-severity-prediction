# check_sample.py
import pandas as pd

df = pd.read_csv("data/df_with_features_sample.csv")
print("Colonnes:", df.columns.tolist())
print("\nTypes des colonnes:")
print(df.dtypes)
print("\nPremières lignes:")
print(df.head())