"""Data loading module for Telco Customer Churn dataset.

This module handles loading the Telco customer churn dataset from CSV
and provides data validation and initial inspection capabilities.

Author: Abhinav Rana
Date: 2024
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Class for loading and validating the Telco customer churn dataset."""
    
    def __init__(self, filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        """Initialize DataLoader with file path.
        
        Args:
            filepath (str): Path to the CSV file containing customer data
        """
        self.filepath = filepath
        self.data = None
        
    def load_data(self):
        """Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded customer data
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Data loaded successfully!")
            print(f"Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def get_data_info(self):
        """Display basic information about the dataset."""
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Customers: {self.data.shape[0]:,}")
        print(f"Number of Features: {self.data.shape[1]}")
        
        print("\n" + "-"*50)
        print("Column Information:")
        print("-"*50)
        print(self.data.info())
        
        print("\n" + "-"*50)
        print("First 5 Rows:")
        print("-"*50)
        print(self.data.head())
        
        print("\n" + "-"*50)
        print("Statistical Summary:")
        print("-"*50)
        print(self.data.describe())
        
        print("\n" + "-"*50)
        print("Missing Values:")
        print("-"*50)
        missing_vals = self.data.isnull().sum()
        if missing_vals.sum() == 0:
            print("No missing values found.")
        else:
            print(missing_vals[missing_vals > 0])
        
        print("\n" + "-"*50)
        print("Target Variable Distribution:")
        print("-"*50)
        if 'Churn' in self.data.columns:
            print(self.data['Churn'].value_counts())
            print(f"\nChurn Rate: {(self.data['Churn'].value_counts(normalize=True)['Yes']*100):.2f}%")
    
    def validate_data(self):
        """Validate data quality and identify potential issues.
        
        Returns:
            dict: Dictionary containing validation results
        """
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return None
        
        validation_results = {
            'total_records': len(self.data),
            'duplicate_records': self.data.duplicated().sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict()
        }
        
        print("\n" + "="*50)
        print("DATA VALIDATION RESULTS")
        print("="*50)
        print(f"Total Records: {validation_results['total_records']:,}")
        print(f"Duplicate Records: {validation_results['duplicate_records']}")
        print(f"Total Missing Values: {sum(validation_results['missing_values'].values())}")
        
        return validation_results


def main():
    """Main function to demonstrate data loading capabilities."""
    # Initialize data loader
    loader = DataLoader()
    
    # Load the data
    data = loader.load_data()
    
    if data is not None:
        # Display data information
        loader.get_data_info()
        
        # Validate data
        validation_results = loader.validate_data()
        
        print("\n" + "="*50)
        print("Data loading and validation complete!")
        print("="*50)


if __name__ == "__main__":
    main()
