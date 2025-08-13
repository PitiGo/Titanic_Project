# Titanic Survival Prediction - Project Summary

## 🎯 Project Overview

This project demonstrates a complete Machine Learning pipeline from data processing to model deployment, using the classic Titanic survival prediction dataset.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Processed Data │    │  Trained Model  │
│  (train.csv)    │───▶│ (processed/)    │───▶│ (models/)       │
│  (test.csv)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   FastAPI       │    │   Docker        │
                       │   Application   │    │   Container     │
                       │   (src/api.py)  │    │   (Dockerfile)  │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   REST API      │    │   Production    │
                       │   Endpoints     │    │   Deployment    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Titanic_Project/
├── 📊 Data Processing
│   ├── data/
│   │   ├── train.csv              # Original training data
│   │   ├── test.csv               # Original test data
│   │   ├── processed/             # Processed data (generated)
│   │   └── submission.csv         # Predictions (generated)
│   └── src/process_data.py        # Data cleaning & feature engineering
│
├── 🤖 Model Training
│   ├── models/                    # Trained models (generated)
│   │   ├── titanic_random_forest.joblib
│   │   └── model_info.txt
│   └── src/train.py              # Model training & evaluation
│
├── 🌐 API Deployment
│   ├── src/api.py                # FastAPI application
│   ├── Dockerfile                # Container configuration
│   ├── docker-compose.yml        # Multi-container setup
│   └── .dockerignore             # Docker build optimization
│
├── 🧪 Testing & Demo
│   ├── test_api.py               # API test suite
│   ├── demo_api.py               # Interactive demo
│   └── run_pipeline.py           # Complete pipeline runner
│
└── 📚 Documentation
    ├── README.md                 # Project documentation
    ├── requirements.txt          # Python dependencies
    └── PROJECT_SUMMARY.md        # This file
```

## 🔄 Complete Pipeline

### 1. Data Processing (`src/process_data.py`)
- **Missing Value Handling**: Age (median), Fare (median), Embarked (mode), Cabin (Unknown)
- **Feature Engineering**:
  - Title extraction from Name
  - FamilySize = SibSp + Parch + 1
  - IsAlone binary indicator
  - Deck extraction from Cabin
- **Categorical Encoding**: One-hot encoding with `pd.get_dummies()`
- **Output**: Processed datasets ready for modeling

### 2. Model Training (`src/train.py`)
- **Algorithm**: RandomForestClassifier
- **Parameters**: n_estimators=100, max_depth=5, random_state=1
- **Evaluation**: 5-fold cross-validation
- **Metrics**: Accuracy, Classification Report, Confusion Matrix
- **Output**: Trained model saved with joblib

### 3. API Development (`src/api.py`)
- **Framework**: FastAPI
- **Endpoints**:
  - `GET /` - API information
  - `GET /health` - Health check
  - `GET /model_info` - Model details
  - `POST /predict` - Single prediction
  - `POST /predict_batch` - Batch predictions
  - `GET /docs` - Interactive documentation
- **Features**:
  - Automatic data preprocessing
  - Input validation with Pydantic
  - Error handling and logging
  - Probability scores

### 4. Containerization (`Dockerfile`)
- **Base Image**: Python 3.9-slim
- **Security**: Non-root user
- **Health Checks**: Automatic monitoring
- **Optimization**: Multi-stage build, .dockerignore

## 🚀 Deployment Options

### Local Development
```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run pipeline
./run.sh

# 3. Start API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Option 1: Docker Compose (Recommended)
docker-compose up --build

# Option 2: Manual Docker
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

### Production Considerations
- **Load Balancing**: Use nginx or similar
- **Monitoring**: Prometheus + Grafana
- **Logging**: Structured logging with JSON
- **Security**: HTTPS, API keys, rate limiting
- **Scaling**: Kubernetes or Docker Swarm

## 📈 Model Performance

- **Cross-validation Score**: 83.16% (±3.85%)
- **Training Accuracy**: 83.95%
- **Features Used**: 27 engineered features
- **Model Size**: ~412KB (joblib format)

## 🔧 Key Features

### Data Processing
- ✅ Handles missing values intelligently
- ✅ Creates meaningful engineered features
- ✅ Ensures feature consistency between train/test
- ✅ Scales to new data automatically

### Model Training
- ✅ Cross-validation for robust evaluation
- ✅ Comprehensive performance metrics
- ✅ Model persistence with metadata
- ✅ Reproducible results (random_state)

### API Design
- ✅ RESTful endpoints
- ✅ Input validation and error handling
- ✅ Automatic documentation (Swagger UI)
- ✅ Health checks and monitoring
- ✅ Batch processing capability

### Deployment
- ✅ Containerized for portability
- ✅ Environment isolation
- ✅ Easy scaling and replication
- ✅ Production-ready configuration

## 🎯 Use Cases

1. **Educational**: Learn ML pipeline best practices
2. **Prototyping**: Quick model deployment
3. **Production**: Scalable ML service
4. **Integration**: Easy to integrate with other systems

## 🔮 Future Enhancements

- **Model Versioning**: MLflow integration
- **A/B Testing**: Multiple model comparison
- **Feature Store**: Centralized feature management
- **Monitoring**: Model drift detection
- **CI/CD**: Automated testing and deployment

## 📊 API Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "SibSp": 1,
    "Parch": 0,
    "Age": 29.0,
    "Fare": 211.3375,
    "Embarked": "S",
    "Name": "Mrs. John Bradley Cumings"
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[{...passenger1...}, {...passenger2...}]'
```

This project demonstrates a production-ready ML pipeline that can be easily adapted for other classification problems.
