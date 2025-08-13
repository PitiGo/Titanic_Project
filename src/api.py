from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional
import os

# Define el modelo de datos de entrada con Pydantic
class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Gender (male or female)")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    Fare: Optional[float] = Field(None, ge=0, description="Passenger fare")
    Embarked: Optional[str] = Field(None, description="Port of embarkation (C, Q, S)")
    Name: Optional[str] = Field(None, description="Passenger name")
    Cabin: Optional[str] = Field(None, description="Cabin number")
    Ticket: Optional[str] = Field(None, description="Ticket number")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Survival prediction (0 = died, 1 = survived)")
    survival_probability: float = Field(..., description="Probability of survival")
    passenger_id: Optional[str] = Field(None, description="Passenger identifier")

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting survival on the Titanic using machine learning",
    version="1.0.0"
)

# Load the trained model
def load_model():
    """Load the trained model"""
    model_path = "./models/titanic_random_forest.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts"""
    global model
    try:
        model = load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def preprocess_passenger(passenger_data: dict) -> pd.DataFrame:
    """Preprocess passenger data to match the model's expected format"""
    
    # Create DataFrame from passenger data
    df = pd.DataFrame([passenger_data])
    
    # Handle missing values
    if df['Age'].isna().any():
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    if df['Fare'].isna().any():
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    if df['Embarked'].isna().any():
        df['Embarked'] = df['Embarked'].fillna('S')
    
    if df['Cabin'].isna().any():
        df['Cabin'] = df['Cabin'].fillna('Unknown')
    
    # Create engineered features (same as in process_data.py)
    if 'Name' in df.columns and not df['Name'].isna().all():
        # Extract title from name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    else:
        df['Title'] = 'Unknown'
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create is_alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract deck from cabin
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')
    
    # Select features for modeling
    features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "Deck"]
    
    # Create dummy variables
    X = pd.get_dummies(df[features])
    
    # Ensure columns match the model's expected features
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    # Reorder columns to match model
    X = X[model_features]
    
    return X

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "RandomForestClassifier" if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: Passenger):
    """Predict survival for a passenger"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert passenger data to dict
        passenger_dict = passenger.dict()
        
        # Preprocess the data
        X = preprocess_passenger(passenger_dict)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return PredictionResponse(
            prediction=int(prediction),
            survival_probability=float(probability[1]),
            passenger_id=passenger_dict.get('Name', 'Unknown')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_batch(passengers: list[Passenger]):
    """Predict survival for multiple passengers"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        
        for passenger in passengers:
            # Convert passenger data to dict
            passenger_dict = passenger.dict()
            
            # Preprocess the data
            X = preprocess_passenger(passenger_dict)
            
            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            results.append({
                "passenger_id": passenger_dict.get('Name', 'Unknown'),
                "prediction": int(prediction),
                "survival_probability": float(probability[1])
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
def get_model_info():
    """Get information about the trained model"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "random_state": model.random_state,
        "n_features": len(model.feature_names_in_),
        "features": list(model.feature_names_in_)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
