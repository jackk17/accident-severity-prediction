"""
ML Model module for accident severity prediction
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AccidentSeverityModel:
    """Model for predicting accident severity"""

    def __init__(self):
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, X, y, test_size=0.2):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)

    def save(self, filepath):
        """Save model to file"""
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True

    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
