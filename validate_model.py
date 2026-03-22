import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess the data"""
    df = pd.read_csv('data.csv')
    
    # Data preprocessing
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Fill all missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Create target variable
    median_credit = df['Credit amount'].median()
    median_duration = df['Duration'].median()
    df['Risk'] = np.where((df['Credit amount'] > median_credit) & (df['Duration'] > median_duration), 'bad', 'good')
    
    # Prepare features and target
    X = df.drop('Risk', axis=1)
    y = df['Risk'].map({'good': 0, 'bad': 1})
    
    return X, y, df

def validate_model():
    """Comprehensive model validation"""
    print("🔍 Credit Scoring Model Validation")
    print("=" * 50)
    
    # Load data
    X, y, df = load_and_preprocess_data()
    
    print(f"📊 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Risk distribution - Good: {sum(y==0)}, Bad: {sum(y==1)}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Basic metrics
    print("📈 Model Performance Metrics:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"   Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"   F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print()
    
    # Detailed classification report
    print("📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Risk', 'Bad Risk']))
    print()
    
    # Confusion Matrix
    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negatives (Good, predicted Good): {cm[0,0]}")
    print(f"   False Positives (Good, predicted Bad): {cm[0,1]}")
    print(f"   False Negatives (Bad, predicted Good): {cm[1,0]}")
    print(f"   True Positives (Bad, predicted Bad): {cm[1,1]}")
    print()
    
    # Cross-validation
    print("🔄 Cross-Validation (5-fold):")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print()
    
    # Feature importance
    print("🎯 Top 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:5], feature_importance['Importance'][:5])):
        print(f"   {i+1}. {feature}: {importance:.3f}")
    print()
    
    # Business logic validation
    print("💼 Business Logic Validation:")
    
    # Test cases
    test_cases = [
        {"name": "Low Risk Profile", "credit": 1000, "duration": 6, "expected": "good"},
        {"name": "High Risk Profile", "credit": 10000, "duration": 48, "expected": "bad"},
        {"name": "Medium Risk Profile", "credit": 3000, "duration": 12, "expected": "good"},
    ]
    
    for case in test_cases:
        # Create test sample similar to training data
        test_sample = X_test.iloc[0:1].copy()
        test_sample['Credit amount'] = case['credit']
        test_sample['Duration'] = case['duration']
        
        # Scale and predict
        test_scaled = scaler.transform(test_sample)
        prediction = model.predict(test_scaled)[0]
        result = "good" if prediction == 0 else "bad"
        
        status = "✅" if result == case['expected'] else "❌"
        print(f"   {status} {case['name']}: Credit=${case['credit']}, Duration={case['duration']} months -> Predicted: {result}")
    
    print()
    print("🎉 Validation Complete!")
    
    # Model reliability check
    print("\n🔍 Model Reliability Check:")
    if accuracy_score(y_test, y_pred) > 0.95:
        print("   ⚠️  Warning: Very high accuracy (>95%) may indicate overfitting")
    elif accuracy_score(y_test, y_pred) > 0.85:
        print("   ✅ Good accuracy range")
    else:
        print("   ❌ Low accuracy - model may need improvement")
    
    if precision_score(y_test, y_pred) > 0.9:
        print("   ✅ High precision - few false positives")
    else:
        print("   ⚠️  Low precision - many false positives")
    
    if recall_score(y_test, y_pred) > 0.9:
        print("   ✅ High recall - few false negatives")
    else:
        print("   ⚠️  Low recall - many false negatives")

if __name__ == "__main__":
    validate_model()
