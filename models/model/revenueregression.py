import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class RevenueRegressionModel:
    """Base class for revenue regression models"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              param_grid: Dict = None, cv: int = 5) -> Dict[str, Any]:
        """
        Train the regression model with optional hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target (revenue)
            param_grid: Dictionary of hyperparameters for grid search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing training metrics
        """
        if param_grid:
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(self.model, param_grid, 
                                      cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
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
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        training_metrics = {
            'model_name': self.model_name,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'best_params': best_params
        }
        
        return training_metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test target (revenue)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # Create scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Revenue')
        plt.ylabel('Predicted Revenue')
        plt.title(f'{self.model_name} - Actual vs Predicted Revenue')
        
        # Save plot
        os.makedirs('models/regresi', exist_ok=True)
        plt.savefig(f'models/regresi/{self.model_name}_actual_vs_predicted.png')
        plt.close()
        
        evaluation_metrics = {
            'model_name': self.model_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        
        return evaluation_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted revenue values
        """
        return self.model.predict(X)
    
    def save_model(self, model_dir: str = 'models/regresi'):
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


class RandomForestRevenueRegressor(RevenueRegressionModel):
    """Random Forest model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__("random_forest_regressor")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )


class XGBoostRevenueRegressor(RevenueRegressionModel):
    """XGBoost model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        super().__init__("xgboost_regressor")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )


if __name__ == "__main__":
    from preprocessing import preprocess_data, split_features_target
    
    # Load and preprocess data
    df = pd.read_csv("processed/processed_data.csv")
    processed_df = preprocess_data(df)
    
    # Split features and target
    X, y_regression, _ = split_features_target(processed_df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Train and evaluate Random Forest model
    rf_model = RandomForestRevenueRegressor()
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15]
    }
    rf_train_metrics = rf_model.train(X_train, y_train, param_grid=rf_params)
    rf_eval_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save_model()
    
    # Train and evaluate XGBoost model
    xgb_model = XGBoostRevenueRegressor()
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_train_metrics = xgb_model.train(X_train, y_train, param_grid=xgb_params)
    xgb_eval_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save_model()
    
    # Print results
    print("\nRandom Forest Regressor Results:")
    print(f"Training RMSE: {rf_train_metrics['train_rmse']:.2f}")
    print(f"Training R²: {rf_train_metrics['train_r2']:.4f}")
    print(f"Test RMSE: {rf_eval_metrics['test_rmse']:.2f}")
    print(f"Test R²: {rf_eval_metrics['test_r2']:.4f}")
    
    print("\nXGBoost Regressor Results:")
    print(f"Training RMSE: {xgb_train_metrics['train_rmse']:.2f}")
    print(f"Training R²: {xgb_train_metrics['train_r2']:.4f}")
    print(f"Test RMSE: {xgb_eval_metrics['test_rmse']:.2f}")
    print(f"Test R²: {xgb_eval_metrics['test_r2']:.4f}")