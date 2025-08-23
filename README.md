# Disease Prediction API

A comprehensive API for predicting disease risk based on medical parameters using machine learning models. The API supports prediction for Heart Disease, Diabetes, Kidney Disease, and Liver Disease.

## Features

- **Multiple Disease Predictions**: Support for Heart Disease, Diabetes, Kidney Disease, and Liver Disease
- **Risk Assessment**: Provides risk percentage and confidence scores
- **Detailed Insights**: Includes risk factors and personalized recommendations
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **RESTful API**: Clean, documented API endpoints
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

### Base URL
- **Local**: `http://localhost:8000`
- **Docker**: `http://localhost:8000`

### Available Endpoints

1. **GET /** - API information and available endpoints
2. **GET /health** - Health check endpoint
3. **POST /predict/heart** - Heart disease prediction
4. **POST /predict/diabetes** - Diabetes prediction
5. **POST /predict/kidney** - Kidney disease prediction
6. **POST /predict/liver** - Liver disease prediction

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd disease-prediction-api
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Using Docker directly

1. **Build the Docker image**
   ```bash
   docker build -t disease-prediction-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 disease-prediction-api
   ```

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

## API Testing with cURL

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Root Endpoint (API Information)
```bash
curl -X GET "http://localhost:8000/"
```

### 3. Heart Disease Prediction
```bash
curl -X POST "http://localhost:8000/predict/heart" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 4. Diabetes Prediction
```bash
curl -X POST "http://localhost:8000/predict/diabetes" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

### 5. Kidney Disease Prediction
```bash
curl -X POST "http://localhost:8000/predict/kidney" \
  -H "Content-Type: application/json" \
  -d '{
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
    "sod": 137.0,
    "pot": 4.0,
    "hemo": 15.4,
    "pcv": 44.0,
    "wc": 7800.0,
    "rc": 5.2,
    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
  }'
```

### 6. Liver Disease Prediction
```bash
curl -X POST "http://localhost:8000/predict/liver" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Single Line cURL Commands (Easy Copy-Paste)

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Heart Disease (Single Line)
```bash
curl -X POST "http://localhost:8000/predict/heart" -H "Content-Type: application/json" -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

### Diabetes (Single Line)
```bash
curl -X POST "http://localhost:8000/predict/diabetes" -H "Content-Type: application/json" -d '{"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35, "insulin": 0, "bmi": 33.6, "diabetes_pedigree_function": 0.627, "age": 50}'
```

### Kidney Disease (Single Line)
```bash
curl -X POST "http://localhost:8000/predict/kidney" -H "Content-Type: application/json" -d '{"age": 48.0, "bp": 80.0, "sg": 1.02, "al": 1.0, "su": 0.0, "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 121.0, "bu": 36.0, "sc": 1.2, "sod": 137.0, "pot": 4.0, "hemo": 15.4, "pcv": 44.0, "wc": 7800.0, "rc": 5.2, "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no"}'
```

### Liver Disease (Single Line)
```bash
curl -X POST "http://localhost:8000/predict/liver" -H "Content-Type: application/json" -d '{"age": 65, "gender": "Female", "total_bilirubin": 0.7, "direct_bilirubin": 0.1, "alkaline_phosphotase": 187, "alamine_aminotransferase": 16, "aspartate_aminotransferase": 18, "total_protiens": 6.8, "albumin": 3.3, "albumin_and_globulin_ratio": 0.9}'
```

## Expected Response Format

All prediction endpoints return a standardized response:

```json
{
  "disease": "Heart Disease",
  "risk_percentage": 85.67,
  "prediction": "High Risk",
  "confidence": 85.67,
  "timestamp": "2024-01-15T10:30:00.123456",
  "additional_info": {
    "risk_factors": ["Age over 65", "High blood pressure", "High cholesterol"],
    "recommendations": [
      "Immediate medical consultation recommended",
      "Consider cardiac stress test",
      "Monitor blood pressure daily",
      "Follow low-sodium, low-fat diet"
    ],
    "model_confidence": 0.8567
  }
}
```

## Testing with Python Script

You can also use the provided test script:
```bash
python test_api.py
```

## Data Models

### Heart Disease Parameters
- `age`: Age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment (0-2)
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia (0-3)

### Diabetes Parameters
- `pregnancies`: Number of pregnancies
- `glucose`: Glucose concentration (mg/dl)
- `blood_pressure`: Blood pressure (mm Hg)
- `skin_thickness`: Skin thickness (mm)
- `insulin`: Insulin (mu U/ml)
- `bmi`: Body mass index
- `diabetes_pedigree_function`: Diabetes pedigree function
- `age`: Age in years

### Kidney Disease Parameters
- `age`: Age in years
- `bp`: Blood pressure (mm Hg)
- `sg`: Specific gravity
- `al`: Albumin
- `su`: Sugar
- `rbc`: Red blood cells (normal/abnormal)
- `pc`: Pus cell (normal/abnormal)
- `pcc`: Pus cell clumps (present/notpresent)
- `ba`: Bacteria (present/notpresent)
- `bgr`: Blood glucose random (mg/dl)
- `bu`: Blood urea (mg/dl)
- `sc`: Serum creatinine (mg/dl)
- `sod`: Sodium (mEq/L)
- `pot`: Potassium (mEq/L)
- `hemo`: Hemoglobin (g/dl)
- `pcv`: Packed cell volume
- `wc`: White blood cell count
- `rc`: Red blood cell count
- `htn`: Hypertension (yes/no)
- `dm`: Diabetes mellitus (yes/no)
- `cad`: Coronary artery disease (yes/no)
- `appet`: Appetite (good/poor)
- `pe`: Pedal edema (yes/no)
- `ane`: Anemia (yes/no)

### Liver Disease Parameters
- `age`: Age in years
- `gender`: Gender (Male/Female)
- `total_bilirubin`: Total bilirubin (mg/dl)
- `direct_bilirubin`: Direct bilirubin (mg/dl)
- `alkaline_phosphotase`: Alkaline phosphatase (IU/L)
- `alamine_aminotransferase`: ALT (IU/L)
- `aspartate_aminotransferase`: AST (IU/L)
- `total_protiens`: Total proteins (g/dl)
- `albumin`: Albumin (g/dl)
- `albumin_and_globulin_ratio`: Albumin and globulin ratio

## Response Format

All prediction endpoints return a standardized response:

```json
{
  "disease": "Disease Name",
  "risk_percentage": 75.5,
  "prediction": "High Risk",
  "confidence": 85.2,
  "timestamp": "2024-01-15T10:30:00.123456",
  "additional_info": {
    "risk_factors": ["Risk factor 1", "Risk factor 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "model_confidence": 0.852
  }
}
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler and SimpleImputer
- **Training Data**: Medical datasets for each disease type
- **Model Performance**: Models are trained on startup for optimal performance

## Health Monitoring

The API includes health monitoring endpoints:

- **GET /health**: Returns API health status and loaded models
- **Docker Health Check**: Automatic health monitoring in Docker containers

## Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid input parameters
- **500 Internal Server Error**: Model prediction errors
- **Detailed Error Messages**: Helpful error descriptions

## Security Considerations

- Non-root user in Docker containers
- Input validation using Pydantic models
- Error handling without exposing sensitive information

## Development

### Project Structure
```
disease-prediction-api/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── README.md             # This file
├── datasets/             # Training datasets
│   ├── heart.csv
│   ├── diabetes.csv
│   ├── kidney_disease.csv
│   └── Liver.csv
└── models/               # Jupyter notebooks (not used in API)
```

### Adding New Diseases

To add support for a new disease:

1. Add the dataset to the `datasets/` directory
2. Create a new Pydantic model for request validation
3. Add training logic in `load_and_train_models()`
4. Create a new prediction endpoint
5. Add risk factors and recommendations functions

## License

This project is licensed under the MIT License.

## Disclaimer

This API is for educational and research purposes only. Medical predictions should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.

