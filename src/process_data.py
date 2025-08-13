import pandas as pd
import numpy as np
import os

def load_data():
    """Load train and test datasets"""
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")
    return train_data, test_data

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Fill missing Embarked with most common value
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fill missing Cabin with 'Unknown'
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    
    return df

def create_features(df):
    """Create additional features from existing data"""
    # Extract title from name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create is_alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract deck from cabin
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')
    
    return df

def process_data():
    """Main function to process the data"""
    print("Loading data...")
    train_data, test_data = load_data()
    
    print("Processing train data...")
    train_data = handle_missing_values(train_data)
    train_data = create_features(train_data)
    
    print("Processing test data...")
    test_data = handle_missing_values(test_data)
    test_data = create_features(test_data)
    
    # Select features for modeling
    features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "Deck"]
    
    # Create dummy variables
    print("Creating dummy variables...")
    X_train = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    
    # Ensure both datasets have the same columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # Save processed data
    print("Saving processed data...")
    train_processed = pd.concat([train_data[['PassengerId', 'Survived']], X_train], axis=1)
    test_processed = pd.concat([test_data[['PassengerId']], X_test], axis=1)
    
    # Create output directory if it doesn't exist
    os.makedirs("./data/processed", exist_ok=True)
    
    train_processed.to_csv("./data/processed/train_processed.csv", index=False)
    test_processed.to_csv("./data/processed/test_processed.csv", index=False)
    
    print(f"Processed data saved:")
    print(f"- Train: {train_processed.shape}")
    print(f"- Test: {test_processed.shape}")
    
    return train_processed, test_processed

if __name__ == "__main__":
    train_processed, test_processed = process_data()
