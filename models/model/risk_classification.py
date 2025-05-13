import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class RiskClassificationModel:
    """Base class for risk classification models"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              param_grid: Dict = None, cv: int = 5) -> Dict[str, Any]:
        """
        Train the classification model with optional hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target (ROI_category)
            param_grid: Dictionary of hyperparameters for grid search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing training metrics
        """
        if param_grid:
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(self.model, param_grid, 
                                      cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters for {self.model_name}: {best_params}")
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
            best_params = "Default parameters used"
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Training metrics
        y_pred_train = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        training_metrics = {
            'model_name': self.model_name,
            'train_accuracy': train_accuracy,
            'best_params': best_params
        }
        
        return training_metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test target (ROI_category)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        evaluation_metrics = {
            'model_name': self.model_name,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        return evaluation_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted risk categories
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for each class
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of class probabilities
        """
        return self.model.predict_proba(X)
    
    def save_model(self, model_dir: str = 'models/klasifikasi'):
        """
        Save the trained model and feature importance to disk
        
        Args:
            model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_path = os.path.join(model_dir, f"{self.model_name}_feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=self.feature_importance.head(10))
            plt.title(f'Top 10 Feature Importance - {self.model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{self.model_name}_feature_importance.png"))
            plt.close()
        
        print(f"{self.model_name} model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model instance
        """
        instance = cls.__new__(cls)
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)
        
        # Extract model name from file path
        instance.model_name = os.path.basename(model_path).replace('.pkl', '')
        
        return instance


class RandomForestRiskClassifier(RiskClassificationModel):
    """Random Forest model for risk classification"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__("random_forest_classifier")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )


class XGBoostRiskClassifier(RiskClassificationModel):
    """XGBoost model for risk classification"""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        super().__init__("xgboost_classifier")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )


if __name__ == "__main__":
    from model.preprocessing import preprocess_data, split_features_target
    
    # Load and preprocess data
    df = pd.read_csv("processed/processed_data.csv")
    processed_df = preprocess_data(df)
    
    # Split features and target
    X, _, y_classification = split_features_target(processed_df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )
    
    # Train and evaluate Random Forest model
    rf_model = RandomForestRiskClassifier()
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15]
    }
    rf_train_metrics = rf_model.train(X_train, y_train, param_grid=rf_params)
    rf_eval_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save_model()
    
    # Train and evaluate XGBoost model
    xgb_model = XGBoostRiskClassifier()
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_train_metrics = xgb_model.train(X_train, y_train, param_grid=xgb_params)
    xgb_eval_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save_model()
    
    # Print results
    print("\nRandom Forest Classifier Results:")
    print(f"Training Accuracy: {rf_train_metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {rf_eval_metrics['test_accuracy']:.4f}")
    
    print("\nXGBoost Classifier Results:")
    print(f"Training Accuracy: {xgb_train_metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {xgb_eval_metrics['test_accuracy']:.4f}") 