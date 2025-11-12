"""Data preprocessing and feature engineering for Telco Customer Churn.

This module handles data cleaning, feature engineering, encoding,
and preparation for machine learning models.

Author: Abhinav Rana
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class TelcoPreprocessor:
    """Comprehensive preprocessing for Telco churn dataset."""
    
    def __init__(self, data):
        """Initialize preprocessor with data.
        
        Args:
            data (pd.DataFrame): Raw Telco customer data
        """
        self.data = data.copy()
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        print("\n" + "="*50)
        print("HANDLING MISSING VALUES")
        print("="*50)
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"\nFound {missing.sum()} missing values")
            print(missing[missing > 0])
            
            # Handle TotalCharges - convert to numeric and fill missing
            if 'TotalCharges' in self.data.columns:
                self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
                self.data['TotalCharges'].fillna(self.data['TotalCharges'].median(), inplace=True)
                print("\nTotalCharges missing values filled with median")
        else:
            print("\nNo missing values found")
    
    def remove_customer_id(self):
        """Remove customerID as it's not useful for prediction."""
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)
            print("\nRemoved customerID column")
    
    def engineer_features(self):
        """Create new features based on domain knowledge."""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # 1. Tenure Groups
        self.data['TenureGroup'] = pd.cut(self.data['tenure'], 
                                           bins=[0, 12, 24, 48, 72],
                                           labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'])
        
        # 2. Average Monthly Charges per Tenure
        self.data['AvgChargesPerTenure'] = self.data['MonthlyCharges'] / (self.data['tenure'] + 1)
        
        # 3. Total Services Subscribed
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        self.data['TotalServices'] = 0
        for col in service_cols:
            if col in self.data.columns:
                self.data['TotalServices'] += (self.data[col] == 'Yes').astype(int)
        
        # 4. Has Multiple Services
        self.data['HasMultipleServices'] = (self.data['TotalServices'] > 1).astype(int)
        
        # 5. Revenue per Service
        self.data['RevenuePerService'] = self.data['MonthlyCharges'] / (self.data['TotalServices'] + 1)
        
        # 6. Senior Citizen with Partner
        self.data['SeniorWithPartner'] = ((self.data['SeniorCitizen'] == 1) & 
                                          (self.data['Partner'] == 'Yes')).astype(int)
        
        # 7. High Value Customer (top 25% total charges)
        self.data['HighValueCustomer'] = (self.data['TotalCharges'] > 
                                          self.data['TotalCharges'].quantile(0.75)).astype(int)
        
        # 8. Contract Quality Score
        contract_scores = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
        if 'Contract' in self.data.columns:
            self.data['ContractQualityScore'] = self.data['Contract'].map(contract_scores)
        
        # 9. Payment Method Security (Electronic = less secure)
        if 'PaymentMethod' in self.data.columns:
            self.data['AutoPayment'] = self.data['PaymentMethod'].isin(
                ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
        
        # 10. Loyalty Score (combination of tenure and contract)
        self.data['LoyaltyScore'] = (self.data['tenure'] / 72) * self.data['ContractQualityScore']
        
        print(f"\nCreated {10} new engineered features:")
        print("  - TenureGroup")
        print("  - AvgChargesPerTenure")
        print("  - TotalServices")
        print("  - HasMultipleServices")
        print("  - RevenuePerService")
        print("  - SeniorWithPartner")
        print("  - HighValueCustomer")
        print("  - ContractQualityScore")
        print("  - AutoPayment")
        print("  - LoyaltyScore")
    
    def encode_categorical_features(self):
        """Encode categorical variables."""
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*50)
        
        # Binary categorical variables
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in self.data.columns:
                self.data[col] = (self.data[col] == 'Yes').astype(int) if col != 'gender' else (self.data[col] == 'Male').astype(int)
        
        # Multi-category variables - use one-hot encoding
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in self.data.columns:
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data.drop(col, axis=1, inplace=True)
        
        # Handle TenureGroup if it exists
        if 'TenureGroup' in self.data.columns:
            dummies = pd.get_dummies(self.data['TenureGroup'], prefix='Tenure', drop_first=True)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data.drop('TenureGroup', axis=1, inplace=True)
        
        print(f"\nEncoded categorical features")
        print(f"Total features after encoding: {len(self.data.columns)}")
    
    def encode_target(self):
        """Encode target variable (Churn)."""
        if 'Churn' in self.data.columns:
            self.data['Churn'] = (self.data['Churn'] == 'Yes').astype(int)
            print("\nTarget variable 'Churn' encoded (Yes=1, No=0)")
    
    def scale_features(self, feature_columns):
        """Scale numerical features using StandardScaler.
        
        Args:
            feature_columns (list): List of feature column names to scale
        """
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        
        self.data[feature_columns] = self.scaler.fit_transform(self.data[feature_columns])
        print(f"\nScaled {len(feature_columns)} numerical features")
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """Prepare final dataset for machine learning.
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\n" + "="*50)
        print("PREPARING FOR MODELING")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\nClass distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
        return X_train, X_test, y_train, y_test
    
    def run_full_pipeline(self):
        """Execute complete preprocessing pipeline.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\n" + "#"*60)
        print("#" + " "*15 + "PREPROCESSING PIPELINE" + " "*21 + "#")
        print("#"*60)
        
        # Step 1: Handle missing values
        self.handle_missing_values()
        
        # Step 2: Remove customer ID
        self.remove_customer_id()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Encode categorical features
        self.encode_categorical_features()
        
        # Step 5: Encode target
        self.encode_target()
        
        # Step 6: Prepare for modeling
        X_train, X_test, y_train, y_test = self.prepare_for_modeling()
        
        print("\n" + "#"*60)
        print("#" + " "*15 + "PREPROCESSING COMPLETE" + " "*21 + "#")
        print("#"*60)
        
        return X_train, X_test, y_train, y_test


def main():
    """Main function to demonstrate preprocessing."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_data()
    
    if data is not None:
        # Initialize preprocessor
        preprocessor = TelcoPreprocessor(data)
        
        # Run full pipeline
        X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
        
        print("\n" + "="*50)
        print("Data ready for machine learning!")
        print("="*50)


if __name__ == "__main__":
    main()
