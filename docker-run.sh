#!/bin/bash

# Titanic API Docker Runner
# This script builds and runs the Titanic API in a Docker container

echo "🚢 Titanic API Docker Runner"
echo "============================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if model exists
if [ ! -f "./models/titanic_random_forest.joblib" ]; then
    echo "❌ Model not found. Please run the training pipeline first:"
    echo "   ./run.sh"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t titanic-api .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Run the container
echo "🚀 Starting Titanic API..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

docker run -p 8000:8000 titanic-api
