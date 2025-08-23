#!/usr/bin/env python3
"""
Test script for the Disease Prediction API
This script tests all endpoints with sample data
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
            print(f"   Models loaded: {response.json()['models_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Message: {response.json()['message']}")
            print(f"   Version: {response.json()['version']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")

def test_heart_disease_prediction():
    """Test heart disease prediction"""
    print("\nTesting heart disease prediction...")
    
    # Sample heart disease data
    heart_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/heart",
            json=heart_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Heart disease prediction passed")
            print(f"   Disease: {result['disease']}")
            print(f"   Risk Percentage: {result['risk_percentage']}%")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Factors: {result['additional_info']['risk_factors']}")
        else:
            print(f"❌ Heart disease prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Heart disease prediction error: {str(e)}")

def test_diabetes_prediction():
    """Test diabetes prediction"""
    print("\nTesting diabetes prediction...")
    
    # Sample diabetes data
    diabetes_data = {
        "pregnancies": 6,
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/diabetes",
            json=diabetes_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Diabetes prediction passed")
            print(f"   Disease: {result['disease']}")
            print(f"   Risk Percentage: {result['risk_percentage']}%")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Factors: {result['additional_info']['risk_factors']}")
        else:
            print(f"❌ Diabetes prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Diabetes prediction error: {str(e)}")

def test_kidney_disease_prediction():
    """Test kidney disease prediction"""
    print("\nTesting kidney disease prediction...")
    
    # Sample kidney disease data
    kidney_data = {
        "age": 48.0,
        "bp": 80.0,
        "sg": 1.02,
        "al": 1.0,
        "su": 0.0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 121.0,
        "bu": 36.0,
        "sc": 1.2,
        "sod": 0.0,
        "pot": 0.0,
        "hemo": 15.4,
        "pcv": 44.0,
        "wc": 7800.0,
        "rc": 5.2,
        "htn": "yes",
        "dm": "yes",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/kidney",
            json=kidney_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Kidney disease prediction passed")
            print(f"   Disease: {result['disease']}")
            print(f"   Risk Percentage: {result['risk_percentage']}%")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Factors: {result['additional_info']['risk_factors']}")
        else:
            print(f"❌ Kidney disease prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Kidney disease prediction error: {str(e)}")

def test_liver_disease_prediction():
    """Test liver disease prediction"""
    print("\nTesting liver disease prediction...")
    
    # Sample liver disease data
    liver_data = {
        "age": 65,
        "gender": "Female",
        "total_bilirubin": 0.7,
        "direct_bilirubin": 0.1,
        "alkaline_phosphotase": 187,
        "alamine_aminotransferase": 16,
        "aspartate_aminotransferase": 18,
        "total_protiens": 6.8,
        "albumin": 3.3,
        "albumin_and_globulin_ratio": 0.9
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/liver",
            json=liver_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Liver disease prediction passed")
            print(f"   Disease: {result['disease']}")
            print(f"   Risk Percentage: {result['risk_percentage']}%")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Factors: {result['additional_info']['risk_factors']}")
        else:
            print(f"❌ Liver disease prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Liver disease prediction error: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 Starting Disease Prediction API Tests")
    print("=" * 50)
    
    # Wait a moment for the API to be ready
    print("Waiting for API to be ready...")
    time.sleep(5)
    
    # Run tests
    test_health_check()
    test_root_endpoint()
    test_heart_disease_prediction()
    test_diabetes_prediction()
    test_kidney_disease_prediction()
    test_liver_disease_prediction()
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")

if __name__ == "__main__":
    main()

