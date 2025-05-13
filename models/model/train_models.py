import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessing import preprocess_data, split_features_target
from revenueregression import RandomForestRevenueRegressor, XGBoostRevenueRegressor
from risk_classification import RandomForestRiskClassifier, XGBoostRiskClassifier

def create_directories():
    """Create necessary directories for models and results"""
    directories = [
        'models/klasifikasi',
        'models/regresi',
        'results'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_models(data_path):
    """Train all models and save results"""
    # Load and preprocess data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv("processed_data.csv", index=False)
    
    # Split features and targets
    X, y_regression, y_classification = split_features_target(processed_df)
    
    # Split into train and test sets
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    _, _, y_cls_train, y_cls_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )
    
    # Sample distribution visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(data=processed_df, x='ROI_category')
    plt.title('Distribution of Risk Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/risk_category_distribution.png')
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=processed_df, x='revenue', bins=20, kde=True)
    plt.title('Distribution of Revenue')
    plt.tight_layout()
    plt.savefig('results/revenue_distribution.png')
    
    results = {
        'classification': {},
        'regression': {}
    }
    
    # Train classification models
    print("\n===== Training Classification Models =====")
    
    # Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_classifier = RandomForestRiskClassifier()
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15]
    }
    rf_cls_train_metrics = rf_classifier.train(X_train, y_cls_train, param_grid=rf_params)
    rf_cls_eval_metrics = rf_classifier.evaluate(X_test, y_cls_test)
    rf_classifier.save_model()
    
    results['classification']['random_forest'] = {
        'training': rf_cls_train_metrics,
        'evaluation': rf_cls_eval_metrics
    }
    
    # XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    xgb_classifier = XGBoostRiskClassifier()
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_cls_train_metrics = xgb_classifier.train(X_train, y_cls_train, param_grid=xgb_params)
    xgb_cls_eval_metrics = xgb_classifier.evaluate(X_test, y_cls_test)
    xgb_classifier.save_model()
    
    results['classification']['xgboost'] = {
        'training': xgb_cls_train_metrics,
        'evaluation': xgb_cls_eval_metrics
    }
    
    # Train regression models
    print("\n===== Training Regression Models =====")
    
    # Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf_regressor = RandomForestRevenueRegressor()
    rf_reg_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15]
    }
    rf_reg_train_metrics = rf_regressor.train(X_train, y_reg_train, param_grid=rf_reg_params)
    rf_reg_eval_metrics = rf_regressor.evaluate(X_test, y_reg_test)
    rf_regressor.save_model()
    
    results['regression']['random_forest'] = {
        'training': rf_reg_train_metrics,
        'evaluation': rf_reg_eval_metrics
    }
    
    # XGBoost Regressor
    print("\nTraining XGBoost Regressor...")
    xgb_regressor = XGBoostRevenueRegressor()
    xgb_reg_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_reg_train_metrics = xgb_regressor.train(X_train, y_reg_train, param_grid=xgb_reg_params)
    xgb_reg_eval_metrics = xgb_regressor.evaluate(X_test, y_reg_test)
    xgb_regressor.save_model()
    
    results['regression']['xgboost'] = {
        'training': xgb_reg_train_metrics,
        'evaluation': xgb_reg_eval_metrics
    }
    
    # Save results as JSON
    with open('results/model_metrics.json', 'w') as f:
        # Convert numpy values to regular Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Clean up non-serializable objects
        cleaned_results = json.loads(json.dumps(results, default=convert_to_serializable))
        json.dump(cleaned_results, f, indent=4)
    
    print("\nModel training complete. Results saved to results/model_metrics.json")
    
    # Compare models and create summary
    cls_models = ["Random Forest", "XGBoost"]
    cls_accuracy = [
        rf_cls_eval_metrics['test_accuracy'],
        xgb_cls_eval_metrics['test_accuracy']
    ]
    
    reg_models = ["Random Forest", "XGBoost"]
    reg_rmse = [
        rf_reg_eval_metrics['test_rmse'],
        xgb_reg_eval_metrics['test_rmse']
    ]
    reg_r2 = [
        rf_reg_eval_metrics['test_r2'],
        xgb_reg_eval_metrics['test_r2']
    ]
    
    # Create comparison plots
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cls_models, y=cls_accuracy)
    plt.title('Classification Models - Test Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('results/classification_accuracy_comparison.png')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=reg_models, y=reg_rmse)
    plt.title('Regression Models - Test RMSE')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig('results/regression_rmse_comparison.png')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=reg_models, y=reg_r2)
    plt.title('Regression Models - Test R²')
    plt.ylim(0, 1)
    plt.ylabel('R²')
    plt.tight_layout()
    plt.savefig('results/regression_r2_comparison.png')
    
    return results

if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Train models using the raw data
    data_path = "processed/processed_data.csv"
    results = train_models(data_path)
    
    print("\nResults Summary:")
    print("Classification Models:")
    print(f"  Random Forest Accuracy: {results['classification']['random_forest']['evaluation']['test_accuracy']:.4f}")
    print(f"  XGBoost Accuracy: {results['classification']['xgboost']['evaluation']['test_accuracy']:.4f}")
    
    print("\nRegression Models:")
    print(f"  Random Forest RMSE: {results['regression']['random_forest']['evaluation']['test_rmse']:.2f}")
    print(f"  Random Forest R²: {results['regression']['random_forest']['evaluation']['test_r2']:.4f}")
    print(f"  XGBoost RMSE: {results['regression']['xgboost']['evaluation']['test_rmse']:.2f}")
    print(f"  XGBoost R²: {results['regression']['xgboost']['evaluation']['test_r2']:.4f}")