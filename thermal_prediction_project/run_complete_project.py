#!/usr/bin/env python3
"""
Master Project Execution Script
===============================
Runs the complete thermal prediction project pipeline.
For demonstration and testing purposes.
"""

import os
import sys
import time
import subprocess

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"   {e.stderr}")
        return False

def main():
    """
    Main execution pipeline
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     PREDICTIVE THERMAL MANAGEMENT SYSTEM                ║
    ║          Complete Project Execution                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if we're in the right directory
    if not os.path.exists('data_collection'):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("This script will run the complete project pipeline:")
    print("  1. Generate sample thermal data")
    print("  2. Preprocess and engineer features")
    print("  3. Train and compare multiple ML models")
    print("  4. Generate visualizations and reports")
    print("  5. Compare with Kaggle dataset")
    print("\nEstimated time: 5-10 minutes")
    
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Execution cancelled")
        sys.exit(0)
    
    # Step 1: Generate sample data
    print_header("STEP 1: GENERATING SAMPLE DATA")
    print("Creating synthetic thermal telemetry data...")
    
    # Create sample data generation script
    sample_data_script = """
import pandas as pd
import numpy as np
import os

print("Generating sample thermal data...")

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
duration_minutes = 30
samples_per_minute = 60
total_samples = duration_minutes * samples_per_minute

# Initialize arrays
timestamps = pd.date_range(start='2026-01-30 10:00:00', 
                          periods=total_samples, freq='1S')

# Workload pattern: idle -> light -> medium -> heavy -> max -> cooling
def get_workload_pattern(sample_idx, total):
    phase = sample_idx / total
    if phase < 0.15:  # Idle
        return 5 + np.random.uniform(-2, 5)
    elif phase < 0.30:  # Light
        return 25 + np.random.uniform(-5, 10)
    elif phase < 0.55:  # Medium
        return 50 + np.random.uniform(-10, 15)
    elif phase < 0.75:  # Heavy
        return 75 + np.random.uniform(-10, 10)
    elif phase < 0.90:  # Maximum
        return 95 + np.random.uniform(-5, 3)
    else:  # Cooling
        return 10 + np.random.uniform(-5, 10)

# Generate data
data = []
cpu_temp = 35.0  # Initial temperature

for i in range(total_samples):
    # CPU load with pattern
    cpu_load = max(0, min(100, get_workload_pattern(i, total_samples)))
    
    # RAM usage (correlated with load but with own pattern)
    ram_usage = 30 + cpu_load * 0.3 + np.random.uniform(-5, 10)
    ram_usage = max(20, min(95, ram_usage))
    
    # Ambient temperature (slow variation)
    ambient_temp = 24 + 2 * np.sin(i / 200) + np.random.normal(0, 0.3)
    
    # CPU temperature physics
    # Heat generation proportional to load
    heat_gen = cpu_load * 0.45
    
    # Cooling proportional to temp difference
    cooling = (cpu_temp - ambient_temp) * 0.15
    
    # Temperature change with thermal inertia
    temp_change = (heat_gen - cooling) * 0.05 + np.random.normal(0, 0.8)
    cpu_temp += temp_change
    
    # Constrain temperature
    cpu_temp = max(ambient_temp + 10, min(95, cpu_temp))
    
    data.append({
        'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
        'unix_time': int(timestamps[i].timestamp()),
        'cpu_load': round(cpu_load, 2),
        'ram_usage': round(ram_usage, 2),
        'ambient_temp': round(ambient_temp, 2),
        'cpu_temp': round(cpu_temp, 2)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
os.makedirs('collected_data', exist_ok=True)
df.to_csv('collected_data/thermal_data.csv', index=False)

print(f"✓ Generated {len(df)} samples")
print(f"  Duration: {duration_minutes} minutes")
print(f"  Temperature range: {df['cpu_temp'].min():.1f}°C - {df['cpu_temp'].max():.1f}°C")
print(f"  Saved to: collected_data/thermal_data.csv")
"""
    
    with open('_generate_sample_data.py', 'w',encoding="utf-8") as f:
        f.write(sample_data_script)
    
    if not run_command('python _generate_sample_data.py', 
                      'Sample data generation'):
        sys.exit(1)
    
    os.remove('_generate_sample_data.py')
    
    # Step 2: Preprocessing
    print_header("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    if not run_command('python data_collection/preprocess_data.py',
                      'Data preprocessing'):
        sys.exit(1)
    
    # Step 3: Model Training
    print_header("STEP 3: MODEL TRAINING & COMPARISON")
    if not run_command('python models/train_model.py',
                      'Model training'):
        sys.exit(1)
    
    # Step 4: Dataset Comparison
    print_header("STEP 4: KAGGLE DATASET COMPARISON")
    if not run_command('python models/compare_datasets.py',
                      'Dataset comparison'):
        sys.exit(1)
    
    # Final Summary
    print_header("PROJECT EXECUTION COMPLETE")
    
    print("✓ All pipeline steps completed successfully!\n")
    print("Generated Files:")
    print("  📁 collected_data/")
    print("     └── thermal_data.csv (raw data)")
    print("  📁 processed_data/")
    print("     └── thermal_processed.csv (engineered features)")
    print("  📁 models/")
    print("     ├── best_thermal_model.pkl (trained model)")
    print("     ├── feature_scaler.pkl (feature scaler)")
    print("     └── model_info.json (model metadata)")
    print("  📁 results/")
    print("     ├── model_comparison.png")
    print("     ├── prediction_analysis.png")
    print("     ├── feature_importance.png")
    print("     ├── temporal_prediction.png")
    print("     ├── model_performance_report.csv")
    print("     └── dataset_comparison/")
    print("         ├── performance_comparison.png")
    print("         ├── prediction_scatter.png")
    print("         └── error_distribution.png")
    print("  📁 visualizations/")
    print("     ├── 01_time_series.png")
    print("     ├── 02_correlation_matrix.png")
    print("     ├── 03_scatter_plots.png")
    print("     └── 04_distributions.png")
    print("  📁 documentation/")
    print("     ├── system_flowchart.png")
    print("     ├── data_flow_diagram.png")
    print("     └── PROJECT_README.md")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review visualizations in results/ folder")
    print("2. Check model performance in results/model_performance_report.csv")
    print("3. For real-time prediction (if on compatible system):")
    print("   python models/predict_realtime.py")
    print("\n4. Read full documentation:")
    print("   documentation/PROJECT_README.md")
    print("\n5. View system architecture:")
    print("   documentation/system_flowchart.png")
    
    print("\n" + "="*70)
    print("PROJECT METRICS SUMMARY")
    print("="*70)
    
    # Try to read and display key metrics
    try:
        report = pd.read_csv('results/model_performance_report.csv')
        print("\nBest Model Performance:")
        best = report.iloc[0]
        print(f"  Model: {best['Model']}")
        print(f"  Test RMSE: {best['Test RMSE']}°C")
        print(f"  Test R²: {best['Test R²']}")
    except:
        pass
    
    print("\n✓ Project execution completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
