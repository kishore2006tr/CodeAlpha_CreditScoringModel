# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Load dataset
df = pd.read_csv('data.csv')

print("First 5 rows:\n", df.head())


# Data Preprocessing
# Drop unwanted column
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical to numeric
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])


# Create target variable based on credit characteristics
# High risk: credit amount > median AND duration > median
median_credit = df['Credit amount'].median()
median_duration = df['Duration'].median()
df['Risk'] = np.where((df['Credit amount'] > median_credit) & (df['Duration'] > median_duration), 'bad', 'good')

print(f"Created target variable - Risk distribution:")
print(df['Risk'].value_counts())

# Features and Target
X = df.drop('Risk', axis=1)
y = df['Risk']

# Convert target
y = y.map({'good': 0, 'bad': 1})


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))