import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_model():
    """Load the trained model"""
    df = pd.read_csv('data.csv')
    
    # Preprocess same as training
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Create target
    median_credit = df['Credit amount'].median()
    median_duration = df['Duration'].median()
    df['Risk'] = np.where((df['Credit amount'] > median_credit) & (df['Duration'] > median_duration), 'bad', 'good')
    
    X = df.drop('Risk', axis=1)
    y = df['Risk'].map({'good': 0, 'bad': 1})
    
    # Train and scale
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    scaler = StandardScaler()
    scaler.fit(X)
    
    return model, scaler, X.columns

def test_manual_cases():
    """Test specific manual cases"""
    model, scaler, feature_names = load_model()
    
    print("🧪 Manual Prediction Testing")
    print("=" * 40)
    
    # Test cases with expected outcomes
    test_cases = [
        {
            "name": "Very Safe Applicant",
            "data": {
                "Age": 25, "Sex": "male", "Job": 2, "Housing": "own",
                "Saving accounts": "moderate", "Checking account": "moderate",
                "Credit amount": 500, "Duration": 6, "Purpose": "radio/TV"
            },
            "expected": "good"
        },
        {
            "name": "Risky Applicant",
            "data": {
                "Age": 35, "Sex": "female", "Job": 1, "Housing": "rent",
                "Saving accounts": "little", "Checking account": "little",
                "Credit amount": 15000, "Duration": 60, "Purpose": "car"
            },
            "expected": "bad"
        },
        {
            "name": "Borderline Case",
            "data": {
                "Age": 30, "Sex": "male", "Job": 3, "Housing": "own",
                "Saving accounts": "little", "Checking account": "moderate",
                "Credit amount": 4000, "Duration": 24, "Purpose": "education"
            },
            "expected": "good"  # Based on our logic
        }
    ]
    
    # Load training data for encoding reference
    df = pd.read_csv('data.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    for case in test_cases:
        print(f"\n📋 Testing: {case['name']}")
        print(f"   Input: {case['data']}")
        
        # Create DataFrame
        test_df = pd.DataFrame([case['data']])
        
        # Handle categorical encoding
        for col in test_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # Fit on training data + test data to handle unseen categories
            all_values = list(df[col].fillna('Unknown')) + list(test_df[col])
            le.fit(all_values)
            test_df[col] = le.transform(test_df[col])
        
        # Ensure columns match training data
        for col in feature_names:
            if col not in test_df.columns:
                test_df[col] = 0
        
        test_df = test_df[feature_names]
        
        # Scale and predict
        test_scaled = scaler.transform(test_df)
        prediction = model.predict(test_scaled)[0]
        probability = model.predict_proba(test_scaled)[0]
        
        result = "good" if prediction == 0 else "bad"
        confidence = max(probability) * 100
        
        # Check if correct
        is_correct = "✅" if result == case['expected'] else "❌"
        
        print(f"   Predicted: {result} (Confidence: {confidence:.1f}%)")
        print(f"   Expected: {case['expected']} {is_correct}")
        print(f"   Probabilities - Good: {probability[0]*100:.1f}%, Bad: {probability[1]*100:.1f}%")

def check_model_logic():
    """Verify the model's logic matches our expectations"""
    print("\n🔍 Model Logic Verification")
    print("=" * 40)
    
    df = pd.read_csv('data.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    median_credit = df['Credit amount'].median()
    median_duration = df['Duration'].median()
    
    print(f"   Median Credit Amount: ${median_credit:.0f}")
    print(f"   Median Duration: {median_duration} months")
    print(f"   Risk Rule: Bad if Credit > {median_credit:.0f} AND Duration > {median_duration}")
    
    # Test the rule
    test_credit = 5000
    test_duration = 36
    
    is_bad = test_credit > median_credit and test_duration > median_duration
    risk = "bad" if is_bad else "good"
    
    print(f"\n   Test Case: Credit=${test_credit}, Duration={test_duration} months")
    print(f"   Rule Result: {risk}")

if __name__ == "__main__":
    test_manual_cases()
    check_model_logic()
    print("\n🎉 Manual Testing Complete!")
