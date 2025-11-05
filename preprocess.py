import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


os.makedirs("data", exist_ok=True)

# Download the Titanic dataset if it doesn't exist
if not os.path.exists("data/raw_data.csv"):
    print("Downloading Titanic dataset...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df.to_csv("data/raw_data.csv", index=False)
else:
    print("Using existing data/raw_data.csv")

# Continue with your preprocessing as before
df = pd.read_csv("data/raw_data.csv")

# Drop columns with too many missing values or not useful
df.drop(['PassengerId','Name', 'Cabin', 'Fare', 'Ticket', 'Embarked'], axis=1, inplace=True)

# Fill missing Age values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Feature engineering: Create FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Create age category features (convert to integers)
df['Child'] = (df['Age'] < 18).astype(int)
df['Teen'] = ((df['Age'] >= 18) & (df['Age'] < 25)).astype(int)
df['Adult'] = ((df['Age'] >= 25) & (df['Age'] < 60)).astype(int)
df['Senior'] = (df['Age'] >= 60).astype(int)
df.drop(['Age'], axis=1, inplace=True)

# Encode Sex (Male/Female to 0/1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split into train and test sets (80-20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  
    stratify=y          
)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# Combine features and target back together for saving
train_data = X_train_scaled.copy()
train_data['Survived'] = y_train.values

test_data = X_test_scaled.copy()
test_data['Survived'] = y_test.values

# Save to CSV files
train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

print("Preprocessing complete!")
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print(f"Features: {list(X_train_scaled.columns)}")
