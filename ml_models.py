"""Machine Learning Models for Telco Customer Churn Prediction.

This module implements 7 classification algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Naive Bayes

Author: Abhinav Rana
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


class ChurnModelSuite:
    """Suite of 7 ML algorithms for churn prediction."""
    
    def __init__(self):
        """Initialize all 7 models."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.trained_models = {}
        self.results = {}
    
    def train_all_models(self, X_train, y_train):
        """Train all 7 models.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("\n" + "="*60)
        print("TRAINING 7 MACHINE LEARNING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"✓ {name} trained successfully")
        
        print("\n" + "="*60)
        print("All models trained!")
        print("="*60)
    
    def evaluate_model(self, name, model, X_test, y_test):
        """Evaluate a single model.
        
        Args:
            name (str): Model name
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Comparison of all models
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        all_results = []
        
        for name, model in self.trained_models.items():
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate_model(name, model, X_test, y_test)
            metrics['Model'] = name
            all_results.append(metrics)
            self.results[name] = metrics
            
            # Print metrics
            print(f"Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall:    {metrics['Recall']:.4f}")
            print(f"F1-Score:  {metrics['F1-Score']:.4f}")
            print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
        # Create comparison dataframe
        results_df = pd.DataFrame(all_results)
        results_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        
        return results_df
    
    def print_comparison_table(self, results_df):
        """Print formatted comparison table.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 80)
        
        for idx, row in results_df.iterrows():
            print(f"{row['Model']:<20} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} {row['Recall']:<12.4f} {row['F1-Score']:<12.4f} {row['ROC-AUC']:<12.4f}")
        
        print("\n" + "="*80)
        best_model = results_df.iloc[0]['Model']
        best_score = results_df.iloc[0]['ROC-AUC']
        print(f"BEST MODEL: {best_model} (ROC-AUC: {best_score:.4f})")
        print("="*80)
    
    def get_best_model(self, results_df):
        """Get the best performing model.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            tuple: (model_name, model, metrics)
        """
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.trained_models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        return best_model_name, best_model, best_metrics
    
    def save_model(self, model_name, filename):
        """Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filename (str): Output filename
        """
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filename)
            print(f"\n✓ Model '{model_name}' saved to {filename}")
        else:
            print(f"\nError: Model '{model_name}' not found.")
    
    def load_model(self, filename):
        """Load a trained model from disk.
        
        Args:
            filename (str): Model filename
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filename)
            print(f"\n✓ Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            return None
    
    def get_detailed_report(self, model_name, X_test, y_test):
        """Get detailed classification report for a model.
        
        Args:
            model_name (str): Name of the model
            X_test: Test features
            y_test: Test target
        """
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            y_pred = model.predict(X_test)
            
            print(f"\n{'='*60}")
            print(f"DETAILED REPORT: {model_name}")
            print(f"{'='*60}\n")
            print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(f"\nTrue Negatives:  {cm[0][0]}")
            print(f"False Positives: {cm[0][1]}")
            print(f"False Negatives: {cm[1][0]}")
            print(f"True Positives:  {cm[1][1]}")
        else:
            print(f"Model '{model_name}' not found.")


def main():
    """Main function to demonstrate model training."""
    from data_loader import DataLoader
    from data_preprocessing import TelcoPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_data()
    
    if data is not None:
        preprocessor = TelcoPreprocessor(data)
        X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
        
        # Initialize model suite
        model_suite = ChurnModelSuite()
        
        # Train all models
        model_suite.train_all_models(X_train, y_train)
        
        # Evaluate all models
        results_df = model_suite.evaluate_all_models(X_test, y_test)
        
        # Print comparison
        model_suite.print_comparison_table(results_df)
        
        # Get best model
        best_name, best_model, best_metrics = model_suite.get_best_model(results_df)
        
        # Save best model
        model_suite.save_model(best_name, f'best_model_{best_name.replace(" ", "_").lower()}.pkl')
        
        # Detailed report
        model_suite.get_detailed_report(best_name, X_test, y_test)


if __name__ == "__main__":
    main()
