#!/usr/bin/env python3
"""
Titanic Survival Prediction Pipeline
This script runs the complete pipeline from data processing to model training.
"""

import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function to run the complete pipeline"""
    print("Titanic Survival Prediction - Complete Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Process the data
    success = run_script("src/process_data.py", "Data Processing")
    if not success:
        print("Data processing failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Train the model
    success = run_script("src/train.py", "Model Training")
    if not success:
        print("Model training failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("- data/processed/train_processed.csv")
    print("- data/processed/test_processed.csv")
    print("- models/titanic_random_forest.joblib")
    print("- models/model_info.txt")
    print("- data/submission.csv")

if __name__ == "__main__":
    main()
