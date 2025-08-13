import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_processed_data():
    """Load the processed training data"""
    train_processed = pd.read_csv("./data/processed/train_processed.csv")
    return train_processed

def prepare_features(train_data):
    """Prepare features and target for training"""
    # Separate features and target
    X = train_data.drop(['PassengerId', 'Survived'], axis=1)
    y = train_data['Survived']
    
    return X, y

def train_model(X, y):
    """Train the RandomForest model"""
    print("Training RandomForest model...")
    
    # Initialize the model with the same parameters as in the notebook
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=1
    )
    
    # Train the model
    model.fit(X, y)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model

def evaluate_model(model, X, y):
    """Evaluate the model performance"""
    print("\nModel Evaluation:")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return accuracy

def save_model(model, accuracy):
    """Save the trained model"""
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Save the model
    model_path = "./models/titanic_random_forest.joblib"
    joblib.dump(model, model_path)
    
    # Save model info
    model_info = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 1,
        'accuracy': accuracy,
        'features': list(model.feature_names_in_)
    }
    
    info_path = "./models/model_info.txt"
    with open(info_path, 'w') as f:
        f.write("Titanic Survival Prediction Model\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model Type: {model_info['model_type']}\n")
        f.write(f"Number of Estimators: {model_info['n_estimators']}\n")
        f.write(f"Max Depth: {model_info['max_depth']}\n")
        f.write(f"Random State: {model_info['random_state']}\n")
        f.write(f"Accuracy: {model_info['accuracy']:.4f}\n\n")
        f.write("Features:\n")
        for feature in model_info['features']:
            f.write(f"- {feature}\n")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Model info saved to: {info_path}")
    
    return model_path

def make_predictions(model, test_data_path="./data/processed/test_processed.csv"):
    """Make predictions on test data and save submission file"""
    if os.path.exists(test_data_path):
        print("\nMaking predictions on test data...")
        test_processed = pd.read_csv(test_data_path)
        
        # Prepare test features
        X_test = test_processed.drop(['PassengerId'], axis=1)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Create submission file
        output = pd.DataFrame({
            'PassengerId': test_processed.PassengerId, 
            'Survived': predictions
        })
        
        submission_path = "./data/submission.csv"
        output.to_csv(submission_path, index=False)
        print(f"Submission file saved to: {submission_path}")
        
        return output
    else:
        print(f"Test data not found at {test_data_path}")
        return None

def main():
    """Main function to run the training pipeline"""
    print("Titanic Survival Prediction - Model Training")
    print("=" * 50)
    
    # Load processed data
    print("Loading processed data...")
    train_data = load_processed_data()
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(train_data)
    
    # Train model
    model = train_model(X, y)
    
    # Evaluate model
    accuracy = evaluate_model(model, X, y)
    
    # Save model
    model_path = save_model(model, accuracy)
    
    # Make predictions on test data
    make_predictions(model)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
