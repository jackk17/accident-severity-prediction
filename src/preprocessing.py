"""
Data preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Preprocesses accident data for model training and prediction"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, filepath):
        """Load data from CSV"""
        return pd.read_csv(filepath)

    def handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        df = df.dropna()
        return df

    def encode_categorical(self, df, categorical_cols, fit=False):
        """Encode categorical variables"""
        df_copy = df.copy()

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
            else:
                if col in self.label_encoders:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))

        return df_copy

    def scale_features(self, df, numerical_cols, fit=False):
        """Scale numerical features"""
        df_copy = df.copy()

        if fit:
            df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
        else:
            df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])

        return df_copy

    def preprocess(self, df, categorical_cols=None, numerical_cols=None, fit=False):
        """Full preprocessing pipeline"""
        if categorical_cols is None:
            categorical_cols = []
        if numerical_cols is None:
            numerical_cols = []

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical
        if categorical_cols:
            df = self.encode_categorical(df, categorical_cols, fit=fit)

        # Scale numerical
        if numerical_cols:
            df = self.scale_features(df, numerical_cols, fit=fit)

        return df
