#!/usr/bin/env python3
"""
Demo script for the Titanic Survival Prediction API
This script demonstrates the API functionality with real examples
"""

import requests
import json
import time

def demo_api():
    """Demonstrate the API functionality"""
    
    print("🚢 Titanic Survival Prediction API Demo")
    print("=" * 50)
    
    # API base URL
    base_url = "http://localhost:8000"
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Test 1: Health check
    print("\n1️⃣ Testing API Health...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"   Model loaded: {response.json()['model_loaded']}")
        else:
            print("❌ API health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running with:")
        print("   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Model info
    print("\n2️⃣ Getting Model Information...")
    response = requests.get(f"{base_url}/model_info")
    if response.status_code == 200:
        model_info = response.json()
        print("✅ Model information retrieved:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Estimators: {model_info['n_estimators']}")
        print(f"   Max Depth: {model_info['max_depth']}")
        print(f"   Features: {model_info['n_features']}")
    
    # Test 3: Single prediction - Rose (survived)
    print("\n3️⃣ Predicting Rose's Survival (Titanic Movie)...")
    rose_data = {
        "Pclass": 1,
        "Sex": "female",
        "SibSp": 0,
        "Parch": 0,
        "Age": 17.0,
        "Fare": 211.3375,
        "Embarked": "S",
        "Name": "Rose DeWitt Bukater",
        "Cabin": "B51 B53 B55",
        "Ticket": "PC 17599"
    }
    
    response = requests.post(f"{base_url}/predict", json=rose_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Rose's prediction:")
        print(f"   Prediction: {'Survived' if result['prediction'] == 1 else 'Died'}")
        print(f"   Survival Probability: {result['survival_probability']:.2%}")
    
    # Test 4: Single prediction - Jack (died)
    print("\n4️⃣ Predicting Jack's Survival (Titanic Movie)...")
    jack_data = {
        "Pclass": 3,
        "Sex": "male",
        "SibSp": 0,
        "Parch": 0,
        "Age": 20.0,
        "Fare": 7.2292,
        "Embarked": "S",
        "Name": "Jack Dawson",
        "Cabin": "",
        "Ticket": "A/5 21171"
    }
    
    response = requests.post(f"{base_url}/predict", json=jack_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Jack's prediction:")
        print(f"   Prediction: {'Survived' if result['prediction'] == 1 else 'Died'}")
        print(f"   Survival Probability: {result['survival_probability']:.2%}")
    
    # Test 5: Batch prediction
    print("\n5️⃣ Batch Prediction - Multiple Passengers...")
    passengers = [
        {
            "Pclass": 1,
            "Sex": "female",
            "SibSp": 1,
            "Parch": 0,
            "Age": 29.0,
            "Fare": 211.3375,
            "Embarked": "S",
            "Name": "Mrs. John Bradley Cumings",
            "Cabin": "C85",
            "Ticket": "PC 17599"
        },
        {
            "Pclass": 3,
            "Sex": "male",
            "SibSp": 1,
            "Parch": 0,
            "Age": 22.0,
            "Fare": 7.25,
            "Embarked": "S",
            "Name": "Mr. Owen Harris Braund",
            "Cabin": "",
            "Ticket": "A/5 21171"
        },
        {
            "Pclass": 2,
            "Sex": "female",
            "SibSp": 0,
            "Parch": 0,
            "Age": 26.0,
            "Fare": 7.925,
            "Embarked": "S",
            "Name": "Miss. Laina Heikkinen",
            "Cabin": "",
            "Ticket": "STON/O2. 3101282"
        }
    ]
    
    response = requests.post(f"{base_url}/predict_batch", json=passengers)
    if response.status_code == 200:
        results = response.json()
        print("✅ Batch predictions:")
        for i, pred in enumerate(results['predictions']):
            status = "Survived" if pred['prediction'] == 1 else "Died"
            prob = pred['survival_probability']
            print(f"   {pred['passenger_id']}: {status} ({prob:.2%})")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📖 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")

if __name__ == "__main__":
    demo_api()
