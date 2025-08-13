#!/usr/bin/env python3
"""
Test script for the Titanic Survival Prediction API
This script demonstrates how to use the API endpoints
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n📊 Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
        return False

def test_single_prediction():
    """Test single passenger prediction"""
    print("\n👤 Testing single passenger prediction...")
    
    # Example passenger data
    passenger_data = {
        "Pclass": 1,
        "Sex": "female",
        "SibSp": 1,
        "Parch": 0,
        "Age": 29.0,
        "Fare": 211.3375,
        "Embarked": "S",
        "Name": "Mrs. John Bradley (Florence Briggs Thayer) Cumings",
        "Cabin": "C85",
        "Ticket": "PC 17599"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=passenger_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Passenger: {data['passenger_id']}")
            print(f"   Prediction: {'Survived' if data['prediction'] == 1 else 'Died'}")
            print(f"   Survival Probability: {data['survival_probability']:.2%}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print("\n👥 Testing batch prediction...")
    
    # Example batch of passengers
    passengers_data = [
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
            "Pclass": 1,
            "Sex": "female",
            "SibSp": 1,
            "Parch": 0,
            "Age": 38.0,
            "Fare": 71.2833,
            "Embarked": "C",
            "Name": "Mrs. John Bradley (Florence Briggs Thayer) Cumings",
            "Cabin": "C85",
            "Ticket": "PC 17599"
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=passengers_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful:")
            for i, pred in enumerate(data['predictions']):
                print(f"   Passenger {i+1}: {pred['passenger_id']}")
                print(f"     Prediction: {'Survived' if pred['prediction'] == 1 else 'Died'}")
                print(f"     Survival Probability: {pred['survival_probability']:.2%}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
        return False

def main():
    """Run all tests"""
    print("🧪 Titanic API Test Suite")
    print("=" * 40)
    
    # Wait a moment for the API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        test_health,
        test_model_info,
        test_single_prediction,
        test_batch_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API logs for more details.")

if __name__ == "__main__":
    main()
