from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Prediction API",
    description="API for predicting disease risk based on medical parameters",
    version="1.0.0"
)

# Data models for request validation
class HeartDiseaseRequest(BaseModel):
    age: int
    sex: int  # 1 for male, 0 for female
    cp: int  # chest pain type
    trestbps: int  # resting blood pressure
    chol: int  # cholesterol
    fbs: int  # fasting blood sugar > 120 mg/dl
    restecg: int  # resting electrocardiographic results
    thalach: int  # maximum heart rate achieved
    exang: int  # exercise induced angina
    oldpeak: float  # ST depression induced by exercise
    slope: int  # slope of peak exercise ST segment
    ca: int  # number of major vessels colored by fluoroscopy
    thal: int  # thalassemia

class DiabetesRequest(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int

class KidneyDiseaseRequest(BaseModel):
    age: float
    bp: float  # blood pressure
    sg: float  # specific gravity
    al: float  # albumin
    su: float  # sugar
    rbc: str  # red blood cells
    pc: str  # pus cell
    pcc: str  # pus cell clumps
    ba: str  # bacteria
    bgr: float  # blood glucose random
    bu: float  # blood urea
    sc: float  # serum creatinine
    sod: float  # sodium
    pot: float  # potassium
    hemo: float  # hemoglobin
    pcv: float  # packed cell volume
    wc: float  # white blood cell count
    rc: float  # red blood cell count
    htn: str  # hypertension
    dm: str  # diabetes mellitus
    cad: str  # coronary artery disease
    appet: str  # appetite
    pe: str  # pedal edema
    ane: str  # anemia

class LiverDiseaseRequest(BaseModel):
    age: int
    gender: str  # Male/Female
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: int
    alamine_aminotransferase: int
    aspartate_aminotransferase: int
    total_protiens: float
    albumin: float
    albumin_and_globulin_ratio: float

class PredictionResponse(BaseModel):
    disease: str
    risk_percentage: float
    prediction: str
    confidence: float
    timestamp: str
    additional_info: Dict[str, Any]

# Global variables to store trained models
models = {}
scalers = {}
feature_names = {}

def load_and_train_models():
    """Load datasets and train models for all diseases"""
    models_trained = 0
    
    # Heart Disease Model
    try:
        logger.info("Training Heart Disease model...")
        heart_data = pd.read_csv('datasets/heart.csv')
        X_heart = heart_data.drop('target', axis=1)
        y_heart = heart_data['target']
        
        heart_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        heart_pipeline.fit(X_heart, y_heart)
        models['heart'] = heart_pipeline
        feature_names['heart'] = list(X_heart.columns)
        logger.info("Heart Disease model trained successfully")
        models_trained += 1
    except Exception as e:
        logger.error(f"Error training Heart Disease model: {str(e)}")
    
    # Diabetes Model
    try:
        logger.info("Training Diabetes model...")
        diabetes_data = pd.read_csv('datasets/diabetes.csv')
        X_diabetes = diabetes_data.drop('Outcome', axis=1)
        y_diabetes = diabetes_data['Outcome']
        
        diabetes_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        diabetes_pipeline.fit(X_diabetes, y_diabetes)
        models['diabetes'] = diabetes_pipeline
        feature_names['diabetes'] = list(X_diabetes.columns)
        logger.info("Diabetes model trained successfully")
        models_trained += 1
    except Exception as e:
        logger.error(f"Error training Diabetes model: {str(e)}")
    
    # Kidney Disease Model
    try:
        logger.info("Training Kidney Disease model...")
        kidney_data = pd.read_csv('datasets/kidney_disease.csv')
        
        # Clean the data - remove problematic characters and handle missing values
        kidney_data = kidney_data.replace(['\t?', '?', '\t'], np.nan)
        
        # Convert numeric columns to float, handling errors
        numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in numeric_cols:
            if col in kidney_data.columns:
                kidney_data[col] = pd.to_numeric(kidney_data[col], errors='coerce')
        
        # Handle categorical variables
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in categorical_cols:
            if col in kidney_data.columns:
                kidney_data[col] = kidney_data[col].map({
                    'normal': 1, 'abnormal': 0,
                    'present': 1, 'notpresent': 0,
                    'yes': 1, 'no': 0,
                    'good': 1, 'poor': 0
                }).fillna(0)
        
        X_kidney = kidney_data.drop(['id', 'classification'], axis=1)
        y_kidney = (kidney_data['classification'] == 'ckd').astype(int)
        
        # Remove rows with too many missing values
        X_kidney = X_kidney.dropna(thresh=len(X_kidney.columns) * 0.5)
        y_kidney = y_kidney[X_kidney.index]
        
        kidney_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        kidney_pipeline.fit(X_kidney, y_kidney)
        models['kidney'] = kidney_pipeline
        feature_names['kidney'] = list(X_kidney.columns)
        logger.info("Kidney Disease model trained successfully")
        models_trained += 1
    except Exception as e:
        logger.error(f"Error training Kidney Disease model: {str(e)}")
    
    # Liver Disease Model
    try:
        logger.info("Training Liver Disease model...")
        liver_data = pd.read_csv('datasets/Liver.csv')
        liver_data['Gender'] = liver_data['Gender'].map({'Male': 1, 'Female': 0})
        
        X_liver = liver_data.drop('Dataset', axis=1)
        y_liver = liver_data['Dataset']
        
        liver_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        liver_pipeline.fit(X_liver, y_liver)
        models['liver'] = liver_pipeline
        feature_names['liver'] = list(X_liver.columns)
        logger.info("Liver Disease model trained successfully")
        models_trained += 1
    except Exception as e:
        logger.error(f"Error training Liver Disease model: {str(e)}")
    
    if models_trained > 0:
        logger.info(f"Successfully trained {models_trained} out of 4 models!")
    else:
        raise Exception("Failed to train any models. Please check the dataset files.")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_and_train_models()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict/heart": "Predict heart disease risk",
            "/predict/diabetes": "Predict diabetes risk", 
            "/predict/kidney": "Predict kidney disease risk",
            "/predict/liver": "Predict liver disease risk",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = list(models.keys())
    status = "healthy" if len(loaded_models) > 0 else "unhealthy"
    
    return {
        "status": status,
        "models_loaded": loaded_models,
        "models_count": len(loaded_models),
        "total_expected_models": 4,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/heart", response_model=PredictionResponse)
async def predict_heart_disease(request: HeartDiseaseRequest):
    """Predict heart disease risk"""
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'age': request.age,
            'sex': request.sex,
            'cp': request.cp,
            'trestbps': request.trestbps,
            'chol': request.chol,
            'fbs': request.fbs,
            'restecg': request.restecg,
            'thalach': request.thalach,
            'exang': request.exang,
            'oldpeak': request.oldpeak,
            'slope': request.slope,
            'ca': request.ca,
            'thal': request.thal
        }])
        
        # Make prediction
        model = models['heart']
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_percentage = risk_probability * 100
        
        # Determine prediction and confidence
        prediction = "High Risk" if risk_percentage > 50 else "Low Risk"
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Additional insights
        additional_info = {
            "risk_factors": get_heart_risk_factors(request),
            "recommendations": get_heart_recommendations(risk_percentage),
            "model_confidence": confidence
        }
        
        return PredictionResponse(
            disease="Heart Disease",
            risk_percentage=round(risk_percentage, 2),
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info
        )
        
    except Exception as e:
        logger.error(f"Error in heart disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(request: DiabetesRequest):
    """Predict diabetes risk"""
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'Pregnancies': request.pregnancies,
            'Glucose': request.glucose,
            'BloodPressure': request.blood_pressure,
            'SkinThickness': request.skin_thickness,
            'Insulin': request.insulin,
            'BMI': request.bmi,
            'DiabetesPedigreeFunction': request.diabetes_pedigree_function,
            'Age': request.age
        }])
        
        # Make prediction
        model = models['diabetes']
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_percentage = risk_probability * 100
        
        # Determine prediction and confidence
        prediction = "High Risk" if risk_percentage > 50 else "Low Risk"
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Additional insights
        additional_info = {
            "risk_factors": get_diabetes_risk_factors(request),
            "recommendations": get_diabetes_recommendations(risk_percentage),
            "model_confidence": confidence
        }
        
        return PredictionResponse(
            disease="Diabetes",
            risk_percentage=round(risk_percentage, 2),
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info
        )
        
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/kidney", response_model=PredictionResponse)
async def predict_kidney_disease(request: KidneyDiseaseRequest):
    """Predict kidney disease risk"""
    try:
        # Convert categorical variables
        rbc_val = 1 if request.rbc == 'normal' else 0
        pc_val = 1 if request.pc == 'normal' else 0
        pcc_val = 1 if request.pcc == 'present' else 0
        ba_val = 1 if request.ba == 'present' else 0
        htn_val = 1 if request.htn == 'yes' else 0
        dm_val = 1 if request.dm == 'yes' else 0
        cad_val = 1 if request.cad == 'yes' else 0
        appet_val = 1 if request.appet == 'good' else 0
        pe_val = 1 if request.pe == 'yes' else 0
        ane_val = 1 if request.ane == 'yes' else 0
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'age': request.age,
            'bp': request.bp,
            'sg': request.sg,
            'al': request.al,
            'su': request.su,
            'rbc': rbc_val,
            'pc': pc_val,
            'pcc': pcc_val,
            'ba': ba_val,
            'bgr': request.bgr,
            'bu': request.bu,
            'sc': request.sc,
            'sod': request.sod,
            'pot': request.pot,
            'hemo': request.hemo,
            'pcv': request.pcv,
            'wc': request.wc,
            'rc': request.rc,
            'htn': htn_val,
            'dm': dm_val,
            'cad': cad_val,
            'appet': appet_val,
            'pe': pe_val,
            'ane': ane_val
        }])
        
        # Make prediction
        model = models['kidney']
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_percentage = risk_probability * 100
        
        # Determine prediction and confidence
        prediction = "High Risk" if risk_percentage > 50 else "Low Risk"
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Additional insights
        additional_info = {
            "risk_factors": get_kidney_risk_factors(request),
            "recommendations": get_kidney_recommendations(risk_percentage),
            "model_confidence": confidence
        }
        
        return PredictionResponse(
            disease="Kidney Disease",
            risk_percentage=round(risk_percentage, 2),
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info
        )
        
    except Exception as e:
        logger.error(f"Error in kidney disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/liver", response_model=PredictionResponse)
async def predict_liver_disease(request: LiverDiseaseRequest):
    """Predict liver disease risk"""
    try:
        # Convert gender
        gender_val = 1 if request.gender.lower() == 'male' else 0
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'Age': request.age,
            'Gender': gender_val,
            'Total_Bilirubin': request.total_bilirubin,
            'Direct_Bilirubin': request.direct_bilirubin,
            'Alkaline_Phosphotase': request.alkaline_phosphotase,
            'Alamine_Aminotransferase': request.alamine_aminotransferase,
            'Aspartate_Aminotransferase': request.aspartate_aminotransferase,
            'Total_Protiens': request.total_protiens,
            'Albumin': request.albumin,
            'Albumin_and_Globulin_Ratio': request.albumin_and_globulin_ratio
        }])
        
        # Make prediction
        model = models['liver']
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_percentage = risk_probability * 100
        
        # Determine prediction and confidence
        prediction = "High Risk" if risk_percentage > 50 else "Low Risk"
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Additional insights
        additional_info = {
            "risk_factors": get_liver_risk_factors(request),
            "recommendations": get_liver_recommendations(risk_percentage),
            "model_confidence": confidence
        }
        
        return PredictionResponse(
            disease="Liver Disease",
            risk_percentage=round(risk_percentage, 2),
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info
        )
        
    except Exception as e:
        logger.error(f"Error in liver disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Helper functions for risk factors and recommendations
def get_heart_risk_factors(request: HeartDiseaseRequest) -> List[str]:
    """Identify heart disease risk factors"""
    risk_factors = []
    if request.age > 65:
        risk_factors.append("Age over 65")
    if request.sex == 1 and request.age > 45:
        risk_factors.append("Male over 45")
    if request.sex == 0 and request.age > 55:
        risk_factors.append("Female over 55")
    if request.trestbps > 140:
        risk_factors.append("High blood pressure")
    if request.chol > 240:
        risk_factors.append("High cholesterol")
    if request.fbs == 1:
        risk_factors.append("High fasting blood sugar")
    if request.exang == 1:
        risk_factors.append("Exercise induced angina")
    return risk_factors

def get_heart_recommendations(risk_percentage: float) -> List[str]:
    """Get heart disease recommendations based on risk"""
    recommendations = []
    if risk_percentage > 70:
        recommendations.extend([
            "Immediate medical consultation recommended",
            "Consider cardiac stress test",
            "Monitor blood pressure daily",
            "Follow low-sodium, low-fat diet"
        ])
    elif risk_percentage > 40:
        recommendations.extend([
            "Regular medical check-ups",
            "Exercise regularly",
            "Maintain healthy diet",
            "Monitor cholesterol levels"
        ])
    else:
        recommendations.extend([
            "Maintain healthy lifestyle",
            "Regular exercise",
            "Balanced diet",
            "Annual health check-up"
        ])
    return recommendations

def get_diabetes_risk_factors(request: DiabetesRequest) -> List[str]:
    """Identify diabetes risk factors"""
    risk_factors = []
    if request.glucose > 140:
        risk_factors.append("High glucose levels")
    if request.bmi > 30:
        risk_factors.append("High BMI (obese)")
    if request.blood_pressure > 140:
        risk_factors.append("High blood pressure")
    if request.age > 45:
        risk_factors.append("Age over 45")
    if request.pregnancies > 0:
        risk_factors.append("Previous pregnancies")
    return risk_factors

def get_diabetes_recommendations(risk_percentage: float) -> List[str]:
    """Get diabetes recommendations based on risk"""
    recommendations = []
    if risk_percentage > 70:
        recommendations.extend([
            "Immediate medical consultation",
            "Monitor blood glucose regularly",
            "Follow diabetic diet",
            "Regular exercise program"
        ])
    elif risk_percentage > 40:
        recommendations.extend([
            "Regular blood glucose monitoring",
            "Weight management",
            "Low-carb diet",
            "Regular exercise"
        ])
    else:
        recommendations.extend([
            "Maintain healthy weight",
            "Balanced diet",
            "Regular exercise",
            "Annual glucose screening"
        ])
    return recommendations

def get_kidney_risk_factors(request: KidneyDiseaseRequest) -> List[str]:
    """Identify kidney disease risk factors"""
    risk_factors = []
    if request.age > 60:
        risk_factors.append("Age over 60")
    if request.bp > 140:
        risk_factors.append("High blood pressure")
    if request.bgr > 140:
        risk_factors.append("High blood glucose")
    if request.bu > 20:
        risk_factors.append("High blood urea")
    if request.sc > 1.2:
        risk_factors.append("High serum creatinine")
    if request.htn == 'yes':
        risk_factors.append("Hypertension")
    if request.dm == 'yes':
        risk_factors.append("Diabetes mellitus")
    return risk_factors

def get_kidney_recommendations(risk_percentage: float) -> List[str]:
    """Get kidney disease recommendations based on risk"""
    recommendations = []
    if risk_percentage > 70:
        recommendations.extend([
            "Immediate nephrology consultation",
            "Monitor kidney function regularly",
            "Low-protein diet",
            "Control blood pressure strictly"
        ])
    elif risk_percentage > 40:
        recommendations.extend([
            "Regular kidney function tests",
            "Blood pressure monitoring",
            "Low-sodium diet",
            "Stay hydrated"
        ])
    else:
        recommendations.extend([
            "Regular health check-ups",
            "Maintain healthy blood pressure",
            "Stay hydrated",
            "Avoid excessive protein intake"
        ])
    return recommendations

def get_liver_risk_factors(request: LiverDiseaseRequest) -> List[str]:
    """Identify liver disease risk factors"""
    risk_factors = []
    if request.total_bilirubin > 1.2:
        risk_factors.append("High total bilirubin")
    if request.direct_bilirubin > 0.3:
        risk_factors.append("High direct bilirubin")
    if request.alkaline_phosphotase > 120:
        risk_factors.append("High alkaline phosphatase")
    if request.alamine_aminotransferase > 40:
        risk_factors.append("High ALT levels")
    if request.aspartate_aminotransferase > 40:
        risk_factors.append("High AST levels")
    if request.albumin < 3.5:
        risk_factors.append("Low albumin levels")
    return risk_factors

def get_liver_recommendations(risk_percentage: float) -> List[str]:
    """Get liver disease recommendations based on risk"""
    recommendations = []
    if risk_percentage > 70:
        recommendations.extend([
            "Immediate hepatology consultation",
            "Avoid alcohol completely",
            "Monitor liver function regularly",
            "Follow liver-friendly diet"
        ])
    elif risk_percentage > 40:
        recommendations.extend([
            "Regular liver function tests",
            "Limit alcohol consumption",
            "Healthy diet",
            "Avoid hepatotoxic medications"
        ])
    else:
        recommendations.extend([
            "Maintain healthy lifestyle",
            "Moderate alcohol consumption",
            "Balanced diet",
            "Regular health check-ups"
        ])
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

