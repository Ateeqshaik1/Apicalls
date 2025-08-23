#!/bin/bash

# Disease Prediction API Startup Script

echo "🚀 Starting Disease Prediction API..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if datasets directory exists
if [ ! -d "datasets" ]; then
    echo "❌ Datasets directory not found. Please ensure the datasets folder exists."
    exit 1
fi

# Check if required dataset files exist
required_files=("heart.csv" "diabetes.csv" "kidney_disease.csv" "Liver.csv")
for file in "${required_files[@]}"; do
    if [ ! -f "datasets/$file" ]; then
        echo "❌ Required dataset file not found: datasets/$file"
        exit 1
    fi
done

echo "✅ All prerequisites met"

# Build and start the API
echo "🔨 Building and starting the API..."
docker-compose up --build -d

# Wait for the API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 10

# Check if the API is running
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ API is running successfully!"
    echo ""
    echo "📋 API Information:"
    echo "   URL: http://localhost:8000"
    echo "   Documentation: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo ""
    echo "🔧 Available endpoints:"
    echo "   POST /predict/heart - Heart disease prediction"
    echo "   POST /predict/diabetes - Diabetes prediction"
    echo "   POST /predict/kidney - Kidney disease prediction"
    echo "   POST /predict/liver - Liver disease prediction"
    echo ""
    echo "🧪 To test the API, run: python test_api.py"
    echo ""
    echo "🛑 To stop the API, run: docker-compose down"
else
    echo "❌ API failed to start. Check the logs with: docker-compose logs"
    exit 1
fi

