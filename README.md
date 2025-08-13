# Titanic Survival Prediction Project

This project refactors a Jupyter notebook into modular Python scripts for the Titanic survival prediction challenge.

## Project Structure

```
Titanic_Project/
├── data/
│   ├── train.csv              # Original training data
│   ├── test.csv               # Original test data
│   ├── processed/             # Processed data (generated)
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   └── submission.csv         # Predictions (generated)
├── models/                    # Trained models (generated)
│   ├── titanic_random_forest.joblib
│   └── model_info.txt
├── src/
│   ├── process_data.py        # Data processing script
│   └── train.py              # Model training script
├── notebooks/                 # Original notebooks
├── requirements.txt           # Python dependencies
├── run_pipeline.py           # Complete pipeline runner
└── README.md                 # This file
```

## Installation

### Option 1: Using Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Quick Start Script
```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the complete pipeline
./run.sh
```

### Option 3: Manual Installation
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the complete pipeline
```bash
python run_pipeline.py
```

This will:
1. Process the raw data (handle missing values, create features, generate dummy variables)
2. Train the RandomForest model
3. Save the trained model and make predictions

### Option 2: Run scripts individually

#### Data Processing
```bash
cd src
python process_data.py
```

This script:
- Loads the original train.csv and test.csv files
- Handles missing values (Age, Fare, Embarked, Cabin)
- Creates new features (Title, FamilySize, IsAlone, Deck)
- Generates dummy variables for categorical features
- Saves processed data to `data/processed/`

#### Model Training
```bash
cd src
python train.py
```

This script:
- Loads the processed training data
- Trains a RandomForestClassifier with the same parameters as the notebook
- Performs cross-validation and model evaluation
- Saves the trained model using joblib
- Makes predictions on test data and creates a submission file

## Features

The data processing includes:

### Original Features
- Pclass (Passenger Class)
- Sex
- SibSp (Siblings/Spouses)
- Parch (Parents/Children)
- Age
- Fare
- Embarked

### Engineered Features
- **Title**: Extracted from Name (Mr, Mrs, Miss, Master, Rare)
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: Binary indicator for passengers traveling alone
- **Deck**: First letter of Cabin (A, B, C, D, E, F, G, T, Unknown)

## Model

- **Algorithm**: RandomForestClassifier
- **Parameters**: 
  - n_estimators: 100
  - max_depth: 5
  - random_state: 1
- **Storage**: Saved using joblib for easy loading and prediction

## Output Files

After running the pipeline, you'll have:

1. **Processed Data**: `data/processed/train_processed.csv` and `data/processed/test_processed.csv`
2. **Trained Model**: `models/titanic_random_forest.joblib`
3. **Model Information**: `models/model_info.txt`
4. **Predictions**: `data/submission.csv`

## Loading the Model

To use the trained model for new predictions:

```python
import joblib

# Load the model
model = joblib.load('models/titanic_random_forest.joblib')

# Make predictions
predictions = model.predict(X_new)
```

## Notes

- The scripts use relative paths, so make sure to run them from the correct directory
- The data processing handles missing values and ensures both train and test datasets have the same features
- Cross-validation is performed during training to assess model performance
- The model parameters match those used in the original notebook
