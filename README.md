# Credit Scoring Model

A machine learning-based credit scoring system that predicts credit risk using a Random Forest classifier.

## 🚀 Project Overview

This project implements a credit scoring model that assesses the creditworthiness of loan applicants based on various demographic and financial factors. The model uses historical credit data to predict whether an applicant poses a 'good' or 'bad' credit risk.

## 📊 Dataset

The model uses the German Credit Dataset with the following features:
- **Age**: Applicant's age
- **Sex**: Gender (male/female)
- **Job**: Employment level (1-4)
- **Housing**: Housing situation (own/rent/free)
- **Saving accounts**: Savings account balance
- **Checking account**: Checking account balance
- **Credit amount**: Loan amount requested
- **Duration**: Loan duration in months
- **Purpose**: Purpose of the loan

## 🎯 Target Variable

Since the original dataset lacked a target variable, a synthetic risk classification was created:
- **Good Risk**: Credit amount ≤ median OR duration ≤ median
- **Bad Risk**: Credit amount > median AND duration > median

## 🛠️ Technology Stack

- **Python 3.14**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and metrics
- **Random Forest**: Classification algorithm

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CodeAlpha_CreditScoringModel
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

## 🏃‍♂️ Usage

Run the credit scoring model:

```bash
python credit_model.py
```

## 📈 Model Performance

The current model achieves the following performance metrics:

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision | 100% |
| Recall | 96.9% |
| F1 Score | 98.4% |
| ROC-AUC | 1.0 |

## 🔧 Model Pipeline

1. **Data Loading**: Load and inspect the dataset
2. **Preprocessing**:
   - Remove unnecessary columns
   - Handle missing values using mode imputation
   - Encode categorical variables using LabelEncoder
3. **Feature Engineering**: Create synthetic target variable
4. **Model Training**: Train Random Forest classifier
5. **Evaluation**: Assess model performance using multiple metrics

## 📁 Project Structure

```
CodeAlpha_CreditScoringModel/
├── credit_model.py      # Main model implementation
├── data.csv            # Dataset
├── README.md           # Project documentation
└── venv/               # Virtual environment
```

## 🔍 Key Features

- **Automated Data Preprocessing**: Handles missing values and categorical encoding
- **Feature Scaling**: Standardizes numerical features for optimal performance
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Risk Classification**: Binary classification for credit decision making

## 🚀 Future Enhancements

- [ ] Feature importance analysis
- [ ] Cross-validation implementation
- [ ] Hyperparameter tuning
- [ ] Additional classification algorithms comparison
- [ ] Web interface for real-time predictions
- [ ] Model interpretability using SHAP values

## 📝 License

This project is part of the CodeAlpha internship program.

## 👤 Author

Created as part of CodeAlpha Machine Learning Internship

---

**Note**: This model is for educational purposes and should not be used for actual credit decisions without proper validation and regulatory compliance.
