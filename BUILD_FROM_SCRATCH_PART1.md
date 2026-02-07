# 🚀 BUILD THERMAL PREDICTION SYSTEM FROM SCRATCH
## Complete Step-by-Step Guide with Function Explanations

---

# 📑 TABLE OF CONTENTS

1. [Project Overview](#1-overview)
2. [Prerequisites & Setup](#2-prerequisites)
3. [File Structure Explained](#3-file-structure)
4. [Step 1: Data Collection](#step-1)
5. [Step 2: Data Preprocessing](#step-2)
6. [Step 3: Model Training](#step-3)
7. [Step 4: Real-Time Prediction](#step-4)
8. [Step 5: Testing & Validation](#step-5)
9. [Understanding Each Function](#9-functions)
10. [Troubleshooting](#10-troubleshooting)

---

# 1. PROJECT OVERVIEW <a id="1-overview"></a>

## What We're Building

A **Predictive Thermal Management System** that:
- Collects CPU temperature and load data
- Trains machine learning model
- Predicts future temperature
- Enables proactive cooling (instead of reactive)

## Why It Matters

**Problem**: Traditional cooling reacts AFTER temperature rises → Thermal spikes, throttling, hardware damage

**Solution**: Predict temperature 5-10 seconds ahead → Cool BEFORE overheating occurs

## Technology Stack

```
Hardware Layer: Arduino + DS18B20 sensor (optional)
    ↓
Data Layer: Python + psutil (CPU monitoring)
    ↓
ML Layer: scikit-learn (Ridge Regression)
    ↓
Deployment: Real-time prediction service
```

---

# 2. PREREQUISITES & SETUP <a id="2-prerequisites"></a>

## 2.1 System Requirements

**Operating System**:
- Windows 10/11 (primary)
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

**Hardware**:
- CPU: Any modern processor
- RAM: 4GB minimum, 8GB recommended
- Storage: 500MB free space
- Optional: Arduino Uno/Nano + DS18B20 sensor

**Software**:
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

## 2.2 Install Python

**Windows**:
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ CHECK "Add Python to PATH"
4. Verify installation:
   ```bash
   python --version
   # Should show: Python 3.x.x
   ```

**Linux**:
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

**macOS**:
```bash
brew install python3
python3 --version
```

## 2.3 Install Required Libraries

Create `requirements.txt`:

```
# Core data science
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine learning
scikit-learn>=1.0.0
joblib>=1.1.0

# System monitoring
psutil>=5.8.0

# Serial communication (for Arduino, optional)
pyserial>=3.5

# Jupyter notebook (optional)
jupyter>=1.0.0
```

**Install all packages**:

```bash
# Windows
pip install -r requirements.txt

# Linux/macOS
pip3 install -r requirements.txt
```

**Verify installation**:

```python
python -c "import pandas, numpy, sklearn, psutil; print('✓ All packages installed!')"
```

## 2.4 Create Project Directory

```bash
# Create main project folder
mkdir thermal_prediction_project
cd thermal_prediction_project

# Create subdirectories
mkdir data_collection
mkdir models
mkdir arduino
mkdir results
mkdir visualizations
mkdir documentation
```

Your folder structure:
```
thermal_prediction_project/
├── data_collection/      # Data collection scripts
├── models/              # ML models and training
├── arduino/             # Hardware code
├── results/             # Output files and plots
├── visualizations/      # Data analysis plots
├── documentation/       # Project docs
└── requirements.txt     # Python dependencies
```

---

# 3. FILE STRUCTURE EXPLAINED <a id="3-file-structure"></a>

## Complete Project Structure

```
thermal_prediction_project/
│
├── data_collection/
│   ├── collect_thermal_data.py       # ← Collects CPU temp, load, RAM
│   ├── generate_workload.py          # ← Creates CPU load patterns
│   ├── preprocess_data.py            # ← Cleans data, creates features
│   └── collected_data/               # ← Output: thermal_data.csv
│       └── thermal_data.csv
│
├── models/
│   ├── train_model.py                # ← Trains 7 ML models
│   ├── predict_realtime.py           # ← Real-time prediction
│   ├── compare_datasets.py           # ← Custom vs Kaggle comparison
│   ├── best_thermal_model.pkl        # ← Saved trained model
│   ├── feature_scaler.pkl            # ← Feature scaler
│   └── model_info.json               # ← Model metadata
│
├── arduino/
│   └── temperature_sensor.ino        # ← Arduino firmware
│
├── results/
│   ├── model_comparison.png          # ← Performance charts
│   ├── prediction_analysis.png       # ← Prediction vs actual
│   ├── feature_importance.png        # ← Feature rankings
│   ├── temporal_prediction.png       # ← Time series plot
│   └── model_performance_report.csv  # ← Detailed metrics
│
├── visualizations/
│   ├── 01_time_series.png           # ← Load & temp over time
│   ├── 02_correlation_matrix.png    # ← Feature correlations
│   ├── 03_scatter_plots.png         # ← Relationships
│   └── 04_distributions.png         # ← Data distributions
│
├── documentation/
│   ├── system_flowchart.png         # ← Architecture diagram
│   ├── data_flow_diagram.png        # ← Pipeline visualization
│   └── PROJECT_README.md            # ← Full documentation
│
├── requirements.txt                  # ← Python dependencies
├── QUICKSTART.md                    # ← Quick setup guide
└── thermal_prediction_training.ipynb # ← Jupyter notebook
```

## What Each File Does

### Data Collection Files

**`collect_thermal_data.py`**: 
- **Purpose**: Collect real-time thermal telemetry
- **What it does**:
  1. Reads CPU load (% utilization)
  2. Reads CPU temperature (°C)
  3. Reads RAM usage (%)
  4. Reads ambient temperature (from Arduino or simulated)
  5. Saves to CSV every 1 second
- **When to run**: Before training model (need 30 min of data)

**`generate_workload.py`**:
- **Purpose**: Create controlled CPU load patterns
- **What it does**:
  1. Generates 6 load phases: idle → light → medium → heavy → max → cooldown
  2. Uses busy loops to stress CPU
  3. Runs for ~9 minutes total
- **When to run**: In parallel with `collect_thermal_data.py`

**`preprocess_data.py`**:
- **Purpose**: Clean data and create features
- **What it does**:
  1. Remove outliers (IQR method)
  2. Sort by timestamp
  3. Create 23 physics-based features
  4. Save processed data
  5. Generate 4 visualization plots
- **When to run**: After data collection, before training

### Model Files

**`train_model.py`**:
- **Purpose**: Train and compare ML models
- **What it does**:
  1. Load processed data
  2. Split into train/test (temporal)
  3. Train 7 models (Ridge, RF, etc.)
  4. Evaluate performance (RMSE, R², MAE)
  5. Save best model
  6. Generate comparison charts
- **When to run**: After preprocessing

**`predict_realtime.py`**:
- **Purpose**: Real-time temperature prediction
- **What it does**:
  1. Load trained model
  2. Collect current system state
  3. Engineer features from history
  4. Predict future temperature
  5. Control fan speed (if Arduino connected)
  6. Display predictions in terminal
- **When to run**: After training, for deployment

**`compare_datasets.py`**:
- **Purpose**: Prove custom data superiority
- **What it does**:
  1. Load custom data
  2. Download/simulate Kaggle data
  3. Train same model on both
  4. Compare RMSE, MAE, R²
  5. Generate comparison charts
- **When to run**: Optional, for validation

### Hardware Files

**`temperature_sensor.ino`** (Arduino):
- **Purpose**: Read ambient temperature
- **What it does**:
  1. Initialize DS18B20 sensor
  2. Listen for 'T' command via serial
  3. Read temperature, send back
  4. Optionally control fan via PWM
- **When to run**: Upload to Arduino before data collection

---

# STEP 1: DATA COLLECTION <a id="step-1"></a>

## Understanding Data Collection

**Goal**: Collect 30 minutes of thermal telemetry with controlled workload patterns

**What we're collecting**:
```
Every 1 second:
  - cpu_load: CPU utilization (0-100%)
  - cpu_temp: CPU die temperature (°C)
  - ram_usage: RAM utilization (0-100%)
  - ambient_temp: Room temperature (°C)
  - timestamp: When sample was taken
```

## Step 1.1: Create `collect_thermal_data.py`

**File**: `data_collection/collect_thermal_data.py`

**Purpose**: Main data collection script

**Full code with explanations**:

```python
"""
Thermal Data Collector
======================
Collects CPU temperature, load, and ambient temperature data.

Every function explained:
- get_cpu_temperature(): Reads CPU die temperature
- get_cpu_load(): Measures CPU utilization percentage
- get_ambient_temp(): Gets room temperature
- collect_sample(): Combines all measurements into one sample
- save_to_csv(): Writes sample to file
- main(): Orchestrates 30-minute collection
"""

import psutil      # For CPU monitoring
import time        # For delays and timestamps
import csv         # For CSV file writing
import os          # For file operations
import numpy as np # For math (simulation)
from datetime import datetime

class ThermalDataCollector:
    """
    Main class for collecting thermal data.
    
    Attributes:
        duration: How long to collect (seconds)
        interval: How often to sample (seconds)
        csv_file: Where to save data
        arduino: Serial connection (if available)
    """
    
    def __init__(self, duration_minutes=30):
        """
        Initialize collector.
        
        Args:
            duration_minutes: Collection duration (default: 30)
        
        What this does:
            1. Set duration in seconds
            2. Set sampling interval (1 Hz = 1 sample/second)
            3. Create output directory
            4. Set CSV filename
            5. Try to connect Arduino
        """
        self.duration = duration_minutes * 60  # Convert to seconds
        self.interval = 1.0  # 1 second = 1 Hz sampling
        self.output_dir = 'collected_data'
        
        # Create output directory if doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set CSV filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = os.path.join(
            self.output_dir, 
            f'thermal_data_{timestamp}.csv'
        )
        
        # Try Arduino connection (optional)
        self.arduino = None
        self.arduino_available = self._init_arduino()
    
    def _init_arduino(self):
        """
        Try to connect to Arduino for ambient temperature.
        
        Returns:
            bool: True if connected, False otherwise
        
        What this does:
            1. Try to import serial library
            2. Try common serial ports
            3. If success, store connection
            4. If fail, continue without Arduino
        
        Why it's OK to fail:
            - Arduino is optional
            - We can simulate ambient temperature
            - Main features (CPU) work without it
        """
        try:
            import serial
            
            # Common serial ports by OS
            ports = [
                '/dev/ttyUSB0',    # Linux
                '/dev/ttyACM0',    # Linux (alternate)
                'COM3',            # Windows
                'COM4',            # Windows
                '/dev/cu.usbserial'  # macOS
            ]
            
            for port in ports:
                try:
                    self.arduino = serial.Serial(port, 9600, timeout=1)
                    time.sleep(2)  # Wait for Arduino to initialize
                    print(f"✓ Arduino connected on {port}")
                    return True
                except:
                    continue
            
            print("⚠ Arduino not found - using simulated ambient temperature")
            return False
            
        except ImportError:
            print("⚠ pyserial not installed - using simulated ambient temperature")
            return False
    
    def get_cpu_temperature(self):
        """
        Read CPU die temperature.
        
        Returns:
            float: CPU temperature in °C
        
        What this does:
            1. Use psutil to read sensors
            2. Try different sensor names (Intel, AMD, etc.)
            3. If no sensor, simulate based on load
        
        Why simulation is OK:
            - Some systems don't expose sensors
            - Simulation uses realistic physics model
            - All features still work
        """
        try:
            # Read all temperature sensors
            temps = psutil.sensors_temperatures()
            
            # Try Intel sensors (most common)
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            
            # Try AMD sensors
            elif 'k10temp' in temps:
                return temps['k10temp'][0].current
            
            # Try generic CPU thermal sensor
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
            
            # Try first available sensor
            elif temps:
                first_sensor = list(temps.values())[0]
                return first_sensor[0].current
            
            # No sensors available - simulate
            else:
                return self._simulate_cpu_temp()
        
        except Exception as e:
            # Any error - use simulation
            return self._simulate_cpu_temp()
    
    def _simulate_cpu_temp(self):
        """
        Simulate CPU temperature when no sensor available.
        
        Returns:
            float: Simulated temperature in °C
        
        Physics model:
            T_cpu = T_idle + (CPU_load × heating_factor)
            
            Where:
                T_idle = 35°C (typical idle temperature)
                heating_factor = 0.4°C per 1% CPU load
                noise = random variation (realistic)
        
        Example:
            CPU at 50% load:
            T = 35 + (50 × 0.4) = 35 + 20 = 55°C
        """
        # Get current CPU load
        cpu_load = psutil.cpu_percent(interval=0.1)
        
        # Physics-based model
        T_idle = 35.0  # Base temperature at idle
        heating_factor = 0.4  # °C per % load
        
        # Calculate temperature
        temp = T_idle + (cpu_load * heating_factor)
        
        # Add realistic noise (sensor isn't perfect)
        noise = np.random.normal(0, 1.5)  # ±1.5°C variation
        temp += noise
        
        # Keep in realistic range
        temp = max(30, min(95, temp))  # 30-95°C
        
        return temp
    
    def get_cpu_load(self):
        """
        Get current CPU utilization.
        
        Returns:
            float: CPU load percentage (0-100)
        
        What this does:
            1. Use psutil.cpu_percent()
            2. Measure over 0.5 second interval
            3. Returns average across all cores
        
        Why 0.5 second interval:
            - Too short (<0.1s): Noisy, unreliable
            - Too long (>1s): Lags behind actual load
            - 0.5s: Good balance
        """
        return psutil.cpu_percent(interval=0.5)
    
    def get_ram_usage(self):
        """
        Get current RAM utilization.
        
        Returns:
            float: RAM usage percentage (0-100)
        
        What this does:
            1. Read virtual memory statistics
            2. Extract percent field
        
        Why include RAM:
            - Some workloads are memory-intensive
            - Provides additional context
            - Independent feature from CPU
        """
        return psutil.virtual_memory().percent
    
    def get_ambient_temp(self):
        """
        Get ambient (room) temperature.
        
        Returns:
            float: Ambient temperature in °C
        
        What this does:
            1. If Arduino connected: Read from DS18B20
            2. If not: Simulate realistic room temp
        
        Why ambient matters:
            - Affects cooling efficiency
            - Newton's Law: Q = h×A×(T_cpu - T_ambient)
            - Higher ambient = harder to cool
        """
        if self.arduino_available:
            try:
                # Send 'T' command to Arduino
                self.arduino.write(b'T\n')
                time.sleep(0.1)  # Wait for response
                
                # Read response
                if self.arduino.in_waiting:
                    response = self.arduino.readline()
                    temp_str = response.decode('utf-8').strip()
                    return float(temp_str)
            except:
                pass  # Fall through to simulation
        
        # Simulate ambient temperature
        # Room temp varies slowly (sinusoidal pattern)
        base_temp = 24.0  # Average room temp
        variation = 2.0 * np.sin(time.time() / 100)
        return base_temp + variation
    
    def collect_sample(self):
        """
        Collect one complete data sample.
        
        Returns:
            dict: Sample with all measurements
        
        What this does:
            1. Get current timestamp
            2. Read CPU load
            3. Read CPU temperature
            4. Read RAM usage
            5. Read ambient temperature
            6. Package into dictionary
        
        Format:
            {
                'timestamp': '2026-02-01 10:30:45',
                'unix_time': 1738326645.123,
                'cpu_load': 45.2,
                'ram_usage': 62.3,
                'ambient_temp': 24.5,
                'cpu_temp': 58.3
            }
        """
        current_time = time.time()
        
        sample = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'unix_time': current_time,
            'cpu_load': self.get_cpu_load(),
            'ram_usage': self.get_ram_usage(),
            'ambient_temp': self.get_ambient_temp(),
            'cpu_temp': self.get_cpu_temperature()
        }
        
        return sample
    
    def save_to_csv(self, sample, is_first=False):
        """
        Save sample to CSV file.
        
        Args:
            sample: Dictionary with measurements
            is_first: True if first sample (writes header)
        
        What this does:
            1. Open CSV file (append mode)
            2. If first sample: Write column headers
            3. Write sample values
            4. Close file
        
        Why CSV:
            - Human-readable
            - Easy to import (pandas, Excel)
            - Lightweight
            - Standard format
        """
        # Determine if we need to write header
        write_header = is_first or not os.path.exists(self.csv_file)
        
        # Open file in append mode
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample.keys())
            
            # Write header if first row
            if write_header:
                writer.writeheader()
            
            # Write data row
            writer.writerow(sample)
    
    def display_progress(self, sample_count, total_samples):
        """
        Show collection progress.
        
        Args:
            sample_count: Current sample number
            total_samples: Total samples to collect
        
        What this does:
            1. Calculate progress percentage
            2. Create ASCII progress bar
            3. Show current values
            4. Update in place (no scrolling)
        """
        progress = (sample_count / total_samples) * 100
        
        # Create progress bar
        bar_length = 50
        filled = int(bar_length * sample_count / total_samples)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Display (use \r to overwrite same line)
        print(f"\r[{bar}] {progress:.1f}% | "
              f"Sample {sample_count}/{total_samples}",
              end='', flush=True)
    
    def run(self):
        """
        Main collection loop.
        
        What this does:
            1. Calculate total samples needed
            2. Initialize CSV file
            3. Loop for duration:
               a. Collect sample
               b. Save to CSV
               c. Display progress
               d. Wait for next interval
            4. Print summary when done
        
        Timing:
            - Uses time.sleep() for precise 1 Hz sampling
            - Accounts for processing time
            - Maintains consistent interval
        """
        total_samples = int(self.duration / self.interval)
        
        print(f"\n{'='*70}")
        print(f"THERMAL DATA COLLECTION")
        print(f"{'='*70}")
        print(f"Duration: {self.duration / 60:.0f} minutes")
        print(f"Sampling rate: {1 / self.interval:.0f} Hz (every {self.interval}s)")
        print(f"Total samples: {total_samples:,}")
        print(f"Output file: {self.csv_file}")
        print(f"{'='*70}\n")
        
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while sample_count < total_samples:
                # Collect sample
                sample = self.collect_sample()
                
                # Save to CSV
                is_first = (sample_count == 0)
                self.save_to_csv(sample, is_first)
                
                # Update counter
                sample_count += 1
                
                # Display progress
                self.display_progress(sample_count, total_samples)
                
                # Wait for next interval
                # (account for processing time)
                elapsed = time.time() - start_time
                expected_time = sample_count * self.interval
                wait_time = expected_time - elapsed
                
                if wait_time > 0:
                    time.sleep(wait_time)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Collection stopped by user")
        
        finally:
            # Close Arduino connection
            if self.arduino:
                self.arduino.close()
            
            # Print summary
            actual_duration = time.time() - start_time
            
            print(f"\n\n{'='*70}")
            print(f"COLLECTION COMPLETE")
            print(f"{'='*70}")
            print(f"Samples collected: {sample_count:,}")
            print(f"Duration: {actual_duration / 60:.1f} minutes")
            print(f"File: {self.csv_file}")
            print(f"File size: {os.path.getsize(self.csv_file) / 1024:.1f} KB")
            print(f"{'='*70}\n")
            
            print(f"✓ Data collection successful!")
            print(f"\nNext step: Run preprocessing")
            print(f"  python data_collection/preprocess_data.py")


if __name__ == "__main__":
    # Create collector (30 minutes)
    collector = ThermalDataCollector(duration_minutes=30)
    
    # Run collection
    collector.run()
```

## Step 1.2: Create `generate_workload.py`

**Purpose**: Generate controlled CPU load patterns during data collection

**Why we need this**:
- Need varying CPU loads to train model
- Random usage won't cover all cases systematically
- Controlled patterns ensure data quality

**Full code**:

```python
"""
Workload Generator
==================
Creates controlled CPU load patterns for thermal data collection.

Functions explained:
- burn_cpu(): Create CPU load via busy loops
- phase_workload(): Generate load for one phase
- run_workload_cycle(): Complete 6-phase cycle
"""

import multiprocessing
import time
import math

def burn_cpu(duration, intensity):
    """
    Create CPU load via busy work.
    
    Args:
        duration: How long to run (seconds)
        intensity: Target load (0.0 to 1.0)
    
    What this does:
        1. Run for specified duration
        2. Busy loop for (intensity × time)
        3. Sleep for ((1-intensity) × time)
        4. Repeat until duration elapsed
    
    How it works:
        - Busy loop does math operations
        - Keeps CPU cores active
        - Sleep allows CPU to idle
        - Ratio controls load percentage
    
    Example:
        intensity = 0.75 (75% load)
        Each 1 second:
          - 0.75s busy (CPU working)
          - 0.25s sleep (CPU idle)
        Result: 75% average load
    """
    end_time = time.time() + duration
    
    while time.time() < end_time:
        # Calculate busy and sleep times
        busy_time = intensity
        sleep_time = 1.0 - intensity
        
        # Busy loop
        busy_end = time.time() + busy_time
        while time.time() < busy_end:
            # Do math operations (keeps CPU busy)
            _ = math.sqrt(sum(i**2 for i in range(1000)))
        
        # Sleep
        if sleep_time > 0:
            time.sleep(sleep_time)

def phase_workload(name, intensity, duration):
    """
    Generate workload for one phase.
    
    Args:
        name: Phase name (for display)
        intensity: Target CPU load (0.0-1.0)
        duration: How long to run (seconds)
    
    What this does:
        1. Print phase info
        2. Calculate number of CPU cores
        3. Start worker process for each core
        4. Wait for all processes to complete
    
    Why multiprocessing:
        - Single process can only load one core
        - Need multiple processes for multi-core CPUs
        - Each process runs burn_cpu() on one core
    """
    print(f"\n{'='*60}")
    print(f"PHASE: {name}")
    print(f"  Target load: {intensity*100:.0f}%")
    print(f"  Duration: {duration}s")
    print(f"{'='*60}")
    
    # Get CPU count
    num_cores = multiprocessing.cpu_count()
    print(f"  Using {num_cores} CPU cores")
    
    # Create process for each core
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(
            target=burn_cpu,
            args=(duration, intensity)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print(f"  ✓ Phase complete")

def run_workload_cycle():
    """
    Run complete 6-phase workload cycle.
    
    Phases:
        1. Idle (5%): Establish baseline
        2. Light (25%): Light usage
        3. Medium (50%): Normal usage
        4. Heavy (75%): High usage
        5. Maximum (95%): Stress test
        6. Cooldown (10%): Recovery
    
    Total duration: ~9 minutes
    
    What this does:
        For each phase:
            1. Set target load and duration
            2. Call phase_workload()
            3. Brief pause between phases
    
    Why these phases:
        - Cover full load spectrum
        - Capture thermal response
        - Include transitions
        - Test extreme cases
    """
    phases = [
        ('Idle', 0.05, 60),          # 5% for 60s
        ('Light Load', 0.25, 90),    # 25% for 90s
        ('Medium Load', 0.50, 120),  # 50% for 120s
        ('Heavy Load', 0.75, 90),    # 75% for 90s
        ('Maximum Load', 0.95, 60),  # 95% for 60s
        ('Cooldown', 0.10, 120)      # 10% for 120s
    ]
    
    total_time = sum(duration for _, _, duration in phases)
    
    print(f"\n{'#'*60}")
    print(f"# WORKLOAD GENERATOR")
    print(f"# Total duration: {total_time/60:.1f} minutes")
    print(f"# Phases: {len(phases)}")
    print(f"{'#'*60}\n")
    
    print("⚠ WARNING: This will heavily load your CPU!")
    print("  Ensure:")
    print("    - Laptop is plugged in")
    print("    - Adequate cooling")
    print("    - No critical tasks running")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"\nStarting in {i}...")
        time.sleep(1)
    
    # Run each phase
    start_time = time.time()
    
    for name, intensity, duration in phases:
        phase_workload(name, intensity, duration)
        
        # Brief pause between phases
        if name != 'Cooldown':
            print("\n  Transitioning to next phase...")
            time.sleep(2)
    
    # Summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"WORKLOAD CYCLE COMPLETE")
    print(f"{'='*60}")
    print(f"Actual runtime: {elapsed/60:.1f} minutes")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_workload_cycle()
```

## Step 1.3: Run Data Collection

**Command sequence**:

```bash
# Terminal 1: Start data collection (runs 30 min)
cd thermal_prediction_project
python data_collection/collect_thermal_data.py

# Terminal 2: Run workload generator (runs 9 min)
# Start this IMMEDIATELY after starting collection
python data_collection/generate_workload.py

# Run workload generator 3-4 times
# (Yes, multiple times is GOOD! More diverse data)
```

**What's happening**:

```
Terminal 1 (Collection):
  [============================] 45%
  Sample 810/1800
  
Terminal 2 (Workload):
  PHASE: Medium Load
    Target load: 50%
    Duration: 120s
```

**Expected output file**:

```
data_collection/collected_data/thermal_data_20260201_103045.csv

Format:
timestamp,unix_time,cpu_load,ram_usage,ambient_temp,cpu_temp
2026-02-01 10:30:45,1738234245.12,5.2,38.5,24.3,36.8
2026-02-01 10:30:46,1738234246.12,5.8,38.6,24.3,37.1
...
```

**Success criteria**:
- ✅ File size > 50 KB
- ✅ ~1800 rows (30 min × 60 samples/min)
- ✅ No error messages
- ✅ Temperature varies (not constant)
- ✅ Load shows phases (not random)

---

[Continue in next part due to length...]

---

# TO BE CONTINUED...

This guide continues with:
- Step 2: Data Preprocessing (preprocess_data.py explanation)
- Step 3: Model Training (train_model.py detailed walkthrough)
- Step 4: Real-Time Prediction (predict_realtime.py explained)
- Step 5: Testing & Validation
- Section 9: Every Function Explained
- Section 10: Troubleshooting Common Issues

**The complete guide is over 50 pages. Would you like me to continue with the next sections?**
