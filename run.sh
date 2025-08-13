#!/bin/bash

# Titanic Project Pipeline Runner
# This script activates the virtual environment and runs the complete pipeline

echo "🚢 Titanic Survival Prediction Pipeline"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Run the pipeline
echo "Running pipeline..."
python run_pipeline.py

echo ""
echo "✅ Pipeline completed!"
echo "📁 Check the generated files in data/ and models/ directories"
