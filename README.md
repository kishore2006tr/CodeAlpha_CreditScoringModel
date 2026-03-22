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
pip install pandas numpy scikit-learn streamlit
```

## 🏃‍♂️ Usage

### Option 1: Command Line
Run the credit scoring model:
```bash
python credit_model.py
```

### Option 2: Web Interface (Recommended)
Launch the interactive web application:
```bash
streamlit run app.py
```

The web interface will open in your browser at `http://localhost:8501`

### Option 3: Model Validation
Validate the model performance and accuracy:
```bash
python validate_model.py
```

### Option 4: Manual Testing
Test specific prediction scenarios:
```bash
python test_predictions.py
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

## 🧪 Model Validation

The project includes comprehensive validation tools to ensure model reliability:

### Validation Results
- **Cross-Validation**: 99.9% ± 0.5% accuracy
- **Feature Importance**: Duration (54.3%), Credit Amount (39.5%)
- **Business Logic**: Consistent with risk assessment rules
- **Test Cases**: All manual test scenarios pass validation

### Validation Tools
1. **`validate_model.py`**: Comprehensive model evaluation with metrics, cross-validation, and feature importance
2. **`test_predictions.py`**: Manual testing of specific prediction scenarios
3. **Web Interface**: Real-time prediction testing with risk factor analysis

### Model Reliability
- ✅ High precision (100%) - minimal false positives
- ✅ High recall (96.9%) - minimal false negatives  
- ✅ Stable performance across cross-validation folds
- ⚠️ High accuracy indicates clear synthetic target patterns

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
├── app.py              # Streamlit web interface
├── validate_model.py   # Model validation and testing
├── test_predictions.py # Manual prediction testing
├── data.csv            # Dataset
├── README.md           # Project documentation
└── venv/               # Virtual environment
```

## 🔍 Key Features

- **Interactive Web Interface**: User-friendly Streamlit application
- **Automated Data Preprocessing**: Handles missing values and categorical encoding
- **Feature Scaling**: Standardizes numerical features for optimal performance
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Risk Classification**: Binary classification for credit decision making
- **Real-time Predictions**: Instant credit risk assessment
- **Risk Factor Analysis**: Identifies potential risk factors in applications
- **Model Validation Tools**: Built-in validation and testing utilities
- **Performance Monitoring**: Cross-validation and accuracy tracking

## 🚀 Future Enhancements

- [x] Feature importance analysis ✅
- [x] Cross-validation implementation ✅
- [ ] Hyperparameter tuning
- [ ] Additional classification algorithms comparison
- [x] Web interface for real-time predictions ✅
- [ ] Model interpretability using SHAP values
- [ ] User authentication and data persistence
- [ ] Batch prediction capabilities

## 📝 License

This project is part of the CodeAlpha internship program.

## 👤 Author

Created as part of CodeAlpha Machine Learning Internship

---

**Note**: This model is for educational purposes and should not be used for actual credit decisions without proper validation and regulatory compliance.
