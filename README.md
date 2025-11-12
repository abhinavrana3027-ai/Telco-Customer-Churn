# 📊 Telco Customer Churn Analysis

[![Python Package CI](https://github.com/abhinavrana3027-ai/Telco-Customer-Churn/actions/workflows/python-package.yml/badge.svg)](https://github.com/abhinavrana3027-ai/Telco-Customer-Churn/actions/workflows/python-package.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready machine learning pipeline for predicting customer churn in telecommunications industry**

A comprehensive data science project implementing 7 machine learning algorithms to predict customer churn for a telecommunications company. Features advanced feature engineering, automated EDA, model comparison, and complete CI/CD pipeline.

---

## 🎯 Project Overview

This project analyzes 7,044 customer records with 21 behavioral and demographic attributes to identify churn drivers and develop actionable retention strategies. The solution implements a complete end-to-end ML pipeline with professional production standards.

### Key Features

✅ **7 Machine Learning Algorithms** - Comprehensive model comparison  
✅ **10 Engineered Features** - Domain-driven feature engineering  
✅ **Automated Visualizations** - Professional EDA charts and reports  
✅ **CI/CD Pipeline** - GitHub Actions for automated testing  
✅ **Modular Architecture** - Clean, maintainable, extensible code  
✅ **Complete Documentation** - Professional docstrings and comments

---

## 📁 Project Structure

```
Telco-Customer-Churn/
│
├── data_loader.py              # Data loading and validation
├── data_preprocessing.py       # Feature engineering and encoding
├── visualizations.py           # Automated EDA and plotting
├── ml_models.py                # 7 ML algorithms implementation
├── churn_analysis.py           # Main pipeline orchestration
├── requirements.txt            # Project dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── .github/
│   └── workflows/
│       └── python-package.yml  # CI/CD pipeline
│
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/abhinavrana3027-ai/Telco-Customer-Churn.git
cd Telco-Customer-Churn

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Execute complete pipeline
python churn_analysis.py
```

This will:
1. Load and validate the dataset
2. Generate exploratory visualizations
3. Engineer 10 new features
4. Train 7 machine learning models
5. Compare model performance
6. Save the best model and results

---

## 🤖 Machine Learning Models

The project implements and compares **7 classification algorithms**:

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Interpretable tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Advanced boosting algorithm
5. **Support Vector Machine (SVM)** - Kernel-based classifier
6. **K-Nearest Neighbors (KNN)** - Instance-based learning
7. **Naive Bayes** - Probabilistic classifier

### Model Evaluation Metrics

- **Accuracy** - Overall prediction correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the ROC curve

---

## 🔧 Feature Engineering

The preprocessing module creates **10 engineered features**:

| Feature | Description |
|---------|-------------|
| `TenureGroup` | Categorized customer tenure |
| `AvgChargesPerTenure` | Monthly charges normalized by tenure |
| `TotalServices` | Count of subscribed services |
| `HasMultipleServices` | Binary indicator for service diversity |
| `RevenuePerService` | Revenue efficiency metric |
| `SeniorWithPartner` | Combined demographic indicator |
| `HighValueCustomer` | Top 25% by total charges |
| `ContractQualityScore` | Contract commitment level |
| `AutoPayment` | Automated payment method flag |
| `LoyaltyScore` | Composite loyalty metric |

---

## 📊 Visualizations

Automated generation of professional visualizations:

- **Churn Distribution** - Target variable analysis
- **Tenure Analysis** - Customer lifetime patterns
- **Charges Analysis** - Revenue and pricing insights
- **Categorical Analysis** - Contract, payment, service patterns
- **Correlation Matrix** - Feature relationships heatmap

---

## 🧪 Testing & CI/CD

The project includes a **GitHub Actions pipeline** that:

- Tests across Python 3.8, 3.9, and 3.10
- Runs code linting with flake8
- Validates all module imports
- Ensures code quality standards

---

## 📈 Results & Performance

The pipeline automatically:
- Trains all 7 models
- Generates performance comparison table
- Identifies the best performing model
- Saves trained model for deployment
- Exports results to CSV

---

## 💡 Usage Examples

### Load Data

```python
from data_loader import DataLoader

loader = DataLoader()
data = loader.load_data()
loader.get_data_info()
```

### Preprocess Data

```python
from data_preprocessing import TelcoPreprocessor

preprocessor = TelcoPreprocessor(data)
X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
```

### Train Models

```python
from ml_models import ChurnModelSuite

model_suite = ChurnModelSuite()
model_suite.train_all_models(X_train, y_train)
results = model_suite.evaluate_all_models(X_test, y_test)
```

---

## 📦 Dependencies

- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.2.0` - Machine learning
- `matplotlib>=3.6.0` - Visualization
- `seaborn>=0.12.0` - Statistical plotting
- `joblib>=1.2.0` - Model persistence
- `xgboost>=1.7.0` - Gradient boosting
- `imbalanced-learn>=0.10.0` - Handling imbalanced data

---

## 👨‍💻 Author

**Abhinav Rana**  
Data Science Portfolio Project | 2024

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: IBM Sample Data Sets
- Built for data science portfolio demonstration
- Implements industry best practices for ML pipelines

---

**⭐ If you find this project useful, please consider giving it a star!**
