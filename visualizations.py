"""Visualization module for Telco Customer Churn Analysis.

This module provides comprehensive visualization capabilities including
EDA charts, correlation matrices, and model performance visualizations.

Author: Abhinav Rana
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class TelcoVisualizer:
    """Comprehensive visualization suite for Telco churn analysis."""
    
    def __init__(self, data):
        """Initialize visualizer with data.
        
        Args:
            data (pd.DataFrame): Customer data for visualization
        """
        self.data = data.copy()
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    def plot_churn_distribution(self):
        """Plot churn distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        churn_counts = self.data['Churn'].value_counts()
        ax1.bar(['No Churn', 'Churn'], churn_counts.values, color=self.colors[:2])
        ax1.set_title('Churn Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        
        # Percentage
        churn_pct = self.data['Churn'].value_counts(normalize=True) * 100
        ax2.pie(churn_pct.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                colors=self.colors[:2], startangle=90)
        ax2.set_title('Churn Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Churn distribution plot saved")
    
    def plot_tenure_analysis(self):
        """Analyze tenure vs churn."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Tenure distribution by churn
        self.data[self.data['Churn'] == 0]['tenure'].hist(bins=30, ax=ax1, alpha=0.7, label='No Churn', color=self.colors[0])
        self.data[self.data['Churn'] == 1]['tenure'].hist(bins=30, ax=ax1, alpha=0.7, label='Churn', color=self.colors[1])
        ax1.set_xlabel('Tenure (months)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Tenure Distribution by Churn', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Boxplot
        self.data.boxplot(column='tenure', by='Churn', ax=ax2)
        ax2.set_title('Tenure Comparison', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Churn')
        ax2.set_ylabel('Tenure (months)')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('tenure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Tenure analysis plot saved")
    
    def plot_charges_analysis(self):
        """Analyze charges vs churn."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Monthly charges
        self.data.boxplot(column='MonthlyCharges', by='Churn', ax=ax1)
        ax1.set_title('Monthly Charges by Churn')
        ax1.set_ylabel('Monthly Charges ($)')
        
        # Total charges
        self.data.boxplot(column='TotalCharges', by='Churn', ax=ax2)
        ax2.set_title('Total Charges by Churn')
        ax2.set_ylabel('Total Charges ($)')
        
        # Scatter plot
        churn_no = self.data[self.data['Churn'] == 0]
        churn_yes = self.data[self.data['Churn'] == 1]
        ax3.scatter(churn_no['tenure'], churn_no['MonthlyCharges'], alpha=0.5, label='No Churn', color=self.colors[0])
        ax3.scatter(churn_yes['tenure'], churn_yes['MonthlyCharges'], alpha=0.5, label='Churn', color=self.colors[1])
        ax3.set_xlabel('Tenure')
        ax3.set_ylabel('Monthly Charges')
        ax3.set_title('Tenure vs Monthly Charges')
        ax3.legend()
        
        # Histogram
        churn_no['MonthlyCharges'].hist(bins=30, ax=ax4, alpha=0.7, label='No Churn', color=self.colors[0])
        churn_yes['MonthlyCharges'].hist(bins=30, ax=ax4, alpha=0.7, label='Churn', color=self.colors[1])
        ax4.set_xlabel('Monthly Charges')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Monthly Charges Distribution')
        ax4.legend()
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('charges_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Charges analysis plot saved")
    
    def plot_categorical_analysis(self):
        """Analyze categorical variables."""
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(18, 5))
        
        for idx, col in enumerate(categorical_cols):
            if col in self.data.columns:
                churn_data = pd.crosstab(self.data[col], self.data['Churn'], normalize='index') * 100
                churn_data.plot(kind='bar', ax=axes[idx], color=self.colors[:2])
                axes[idx].set_title(f'{col} vs Churn Rate')
                axes[idx].set_ylabel('Percentage')
                axes[idx].set_xlabel(col)
                axes[idx].legend(['No Churn', 'Churn'])
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Categorical analysis plot saved")
    
    def plot_correlation_matrix(self, numerical_cols):
        """Plot correlation matrix."""
        plt.figure(figsize=(12, 10))
        correlation = self.data[numerical_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation matrix saved")
    
    def create_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50 + "\n")
        
        self.plot_churn_distribution()
        self.plot_tenure_analysis()
        self.plot_charges_analysis()
        self.plot_categorical_analysis()
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        if available_cols:
            self.plot_correlation_matrix(available_cols)
        
        print("\n" + "="*50)
        print("All visualizations generated successfully!")
        print("="*50)


def main():
    """Main function to demonstrate visualization."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_data()
    
    if data is not None:
        # Handle TotalCharges
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
        
        # Encode Churn
        data['Churn'] = (data['Churn'] == 'Yes').astype(int)
        
        # Create visualizations
        visualizer = TelcoVisualizer(data)
        visualizer.create_all_plots()


if __name__ == "__main__":
    main()
