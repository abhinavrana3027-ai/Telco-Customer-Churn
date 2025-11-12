"""Main Telco Customer Churn Analysis Pipeline.

This is the main script that orchestrates the entire analysis:
1. Data loading
2. Preprocessing and feature engineering
3. Visualization generation
4. Model training and evaluation
5. Results reporting

Author: Abhinav Rana
Date: 2024
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from data_preprocessing import TelcoPreprocessor
from visualizations import TelcoVisualizer
from ml_models import ChurnModelSuite


def main():
    """Execute the complete churn analysis pipeline."""
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "TELCO CUSTOMER CHURN ANALYSIS" + " "*19 + "#")
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "Comprehensive ML-Based Prediction System" + " "*13 + "#")
    print("#"*70 + "\n")
    
    # ========== STEP 1: DATA LOADING ==========
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    loader = DataLoader()
    data = loader.load_data()
    
    if data is None:
        print("\nError: Failed to load data. Exiting...")
        return
    
    print(f"\n✓ Successfully loaded {data.shape[0]:,} customer records")
    print(f"✓ Dataset contains {data.shape[1]} features")
    
    # ========== STEP 2: EXPLORATORY DATA ANALYSIS ==========
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Prepare data for visualization
    viz_data = data.copy()
    viz_data['TotalCharges'] = pd.to_numeric(viz_data['TotalCharges'], errors='coerce')
    viz_data['TotalCharges'].fillna(viz_data['TotalCharges'].median(), inplace=True)
    viz_data['Churn'] = (viz_data['Churn'] == 'Yes').astype(int)
    
    # Generate visualizations
    visualizer = TelcoVisualizer(viz_data)
    visualizer.create_all_plots()
    
    print("\n✓ Exploratory analysis complete")
    print("✓ All visualizations saved to current directory")
    
    # ========== STEP 3: DATA PREPROCESSING ==========
    print("\n" + "="*70)
    print("STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*70)
    
    preprocessor = TelcoPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
    
    print("\n✓ Data preprocessing complete")
    print(f"✓ Training samples: {X_train.shape[0]:,}")
    print(f"✓ Test samples: {X_test.shape[0]:,}")
    print(f"✓ Total features after engineering: {X_train.shape[1]}")
    
    # ========== STEP 4: MODEL TRAINING ==========
    print("\n" + "="*70)
    print("STEP 4: MACHINE LEARNING MODEL TRAINING")
    print("="*70)
    
    model_suite = ChurnModelSuite()
    model_suite.train_all_models(X_train, y_train)
    
    print("\n✓ All 7 models trained successfully")
    
    # ========== STEP 5: MODEL EVALUATION ==========
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    results_df = model_suite.evaluate_all_models(X_test, y_test)
    model_suite.print_comparison_table(results_df)
    
    # ========== STEP 6: BEST MODEL ANALYSIS ==========
    print("\n" + "="*70)
    print("STEP 6: BEST MODEL ANALYSIS")
    print("="*70)
    
    best_name, best_model, best_metrics = model_suite.get_best_model(results_df)
    
    print(f"\nBest Performing Model: {best_name}")
    print("\nPerformance Metrics:")
    print("-" * 40)
    for metric, value in best_metrics.items():
        if metric != 'Model':
            print(f"{metric:<15}: {value:.4f}")
    
    # Get detailed classification report
    model_suite.get_detailed_report(best_name, X_test, y_test)
    
    # Save the best model
    model_filename = f'best_model_{best_name.replace(" ", "_").lower()}.pkl'
    model_suite.save_model(best_name, model_filename)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "#"*70)
    print("#" + " "*25 + "ANALYSIS COMPLETE" + " "*27 + "#")
    print("#"*70)
    
    print("\n✓ Pipeline Execution Summary:")
    print("  " + "-" * 60)
    print(f"  ✓ Data Loaded: {data.shape[0]:,} customers")
    print(f"  ✓ Features Engineered: {X_train.shape[1]} total features")
    print(f"  ✓ Models Trained: 7 algorithms")
    print(f"  ✓ Best Model: {best_name}")
    print(f"  ✓ Best ROC-AUC Score: {best_metrics['ROC-AUC']:.4f}")
    print(f"  ✓ Model Saved: {model_filename}")
    print("  " + "-" * 60)
    
    print("\n✓ Deliverables Generated:")
    print("  ✓ churn_distribution.png")
    print("  ✓ tenure_analysis.png")
    print("  ✓ charges_analysis.png")
    print("  ✓ categorical_analysis.png")
    print("  ✓ correlation_matrix.png")
    print(f"  ✓ {model_filename}")
    
    print("\n" + "#"*70)
    print("#" + " "*15 + "Thank you for using Telco Churn Analyzer!" + " "*12 + "#")
    print("#"*70 + "\n")
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("✓ Model comparison results saved to 'model_comparison_results.csv'\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\u2717 Error during execution: {str(e)}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt\n")
