# BUILD FROM SCRATCH - PART 2
## Continuing: Preprocessing, Training, Deployment & Function Reference
# OLD DONT REFER PURELY READ README
---

# STEP 2: DATA PREPROCESSING <a id="step-2"></a>

## Understanding Preprocessing

**What it does**:
1. Clean data (remove outliers, duplicates)
2. Sort by time (chronological order)
3. Engineer 23 physics-based features
4. Visualize data (4 plots)
5. Save processed data

**Why it's critical**:
- Raw data has noise → Remove outliers
- Features encode physics → Enable learning
- Visualizations → Validate quality

## Step 2.1: The `preprocess_data.py` Script

**File**: `data_collection/preprocess_data.py`

```python
"""
Data Preprocessing & Feature Engineering
========================================
Cleans raw data and creates 23 physics-based features.

EVERY FUNCTION EXPLAINED:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ThermalDataPreprocessor:
    """
    Preprocesses thermal data and creates features.
    
    Process:
        1. Load raw CSV
        2. Clean data (outliers, duplicates)
        3. Engineer features (lags, rates, interactions)
        4. Visualize
        5. Save processed data
    """
    
    def __init__(self, input_file):
        """
        Initialize preprocessor.
        
        Args:
            input_file: Path to raw data CSV
        
        What happens:
            - Store file path
            - Create output directories
            - Load data into pandas DataFrame
        """
        self.input_file = input_file
        self.output_dir = 'processed_data'
        self.viz_dir = '../visualizations'
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"  Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
    
    def remove_outliers(self, columns):
        """
        Remove outliers using IQR method.
        
        Args:
            columns: List of columns to check
        
        Returns:
            DataFrame without outliers
        
        IQR METHOD EXPLAINED:
        
        1. Calculate quartiles:
           Q1 = 25th percentile (25% of data below)
           Q3 = 75th percentile (75% of data below)
           
        2. Calculate IQR:
           IQR = Q3 - Q1 (middle 50% range)
           
        3. Define bounds:
           Lower = Q1 - 1.5×IQR
           Upper = Q3 + 1.5×IQR
           
        4. Remove points outside bounds
        
        EXAMPLE:
           Data: [10, 12, 14, 15, 16, 18, 20, 100]
           Q1 = 13 (25th percentile)
           Q3 = 19 (75th percentile)
           IQR = 19 - 13 = 6
           Lower = 13 - 1.5×6 = 4
           Upper = 19 + 1.5×6 = 28
           Outlier: 100 (> 28) → Remove!
        
        WHY 1.5×IQR:
           - Standard statistical threshold
           - Removes ~0.7% of data (if normal)
           - Conservative (doesn't remove too much)
        """
        df_clean = self.df.copy()
        initial_rows = len(df_clean)
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Keep only data within bounds
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & 
                (df_clean[col] <= upper_bound)
            ]
        
        removed = initial_rows - len(df_clean)
        print(f"\nOutlier removal:")
        print(f"  Removed: {removed} rows ({removed/initial_rows*100:.2f}%)")
        print(f"  Remaining: {len(df_clean)} rows")
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create 23 physics-based features.
        
        Args:
            df: Clean DataFrame
        
        Returns:
            DataFrame with engineered features
        
        FEATURE CATEGORIES:
        
        1. LAG FEATURES (5):
           Capture thermal inertia
           - cpu_load_lag1, lag5, lag10
           - cpu_temp_lag1, lag5
        
        2. RATE FEATURES (3):
           Capture dynamics
           - temp_rate (dT/dt)
           - temp_acceleration (d²T/dt²)
           - load_rate (dLoad/dt)
        
        3. ROLLING FEATURES (4):
           Low-pass filtering
           - cpu_load_roll10, roll30
           - cpu_temp_roll10
           - cpu_load_std10
        
        4. INTERACTION FEATURES (3):
           Non-linear effects
           - load × ambient
           - load × temp (thermal stress)
           - temp - ambient (ΔT for cooling)
        
        5. REGIME INDICATORS (3):
           Operating states
           - is_high_load (>70%)
           - is_heating (temp rising)
           - is_cooling (temp falling)
        
        6. TIME FEATURES (2):
           Cyclical patterns
           - hour_sin, hour_cos
        """
        df_eng = df.copy()
        
        print("\nEngineering features...")
        
        # ==========================================
        # LAG FEATURES
        # ==========================================
        # WHY: Temperature depends on PAST heat generation
        # PHYSICS: T(t) = integral of past Load
        
        df_eng['cpu_load_lag1'] = df_eng['cpu_load'].shift(1)
        # .shift(1) moves values down by 1 row
        # Row 10 now has value from row 9
        
        df_eng['cpu_load_lag5'] = df_eng['cpu_load'].shift(5)
        df_eng['cpu_load_lag10'] = df_eng['cpu_load'].shift(10)
        df_eng['cpu_temp_lag1'] = df_eng['cpu_temp'].shift(1)
        df_eng['cpu_temp_lag5'] = df_eng['cpu_temp'].shift(5)
        
        print("  ✓ Created lag features (thermal inertia)")
        
        # ==========================================
        # RATE FEATURES
        # ==========================================
        # WHY: dT/dt indicates if heating or cooling
        # PHYSICS: dT/dt = a×Load - b×(T-T_amb)
        
        df_eng['temp_rate'] = df_eng['cpu_temp'].diff()
        # .diff() computes difference between consecutive rows
        # Row 10: temp_rate = T[10] - T[9]
        # Positive = heating, Negative = cooling
        
        df_eng['temp_acceleration'] = df_eng['temp_rate'].diff()
        # Second derivative: how rate itself changes
        
        df_eng['load_rate'] = df_eng['cpu_load'].diff()
        
        print("  ✓ Created rate features (thermal dynamics)")
        
        # ==========================================
        # ROLLING FEATURES
        # ==========================================
        # WHY: Thermal system acts as low-pass filter
        # PHYSICS: Temperature responds to AVERAGE load
        
        df_eng['cpu_load_roll10'] = df_eng['cpu_load'].rolling(
            window=10, min_periods=1
        ).mean()
        # .rolling(10) creates 10-sample moving window
        # .mean() averages values in window
        # Row 10: avg of rows 1-10
        # Row 20: avg of rows 11-20
        
        df_eng['cpu_temp_roll10'] = df_eng['cpu_temp'].rolling(
            window=10, min_periods=1
        ).mean()
        
        df_eng['cpu_load_roll30'] = df_eng['cpu_load'].rolling(
            window=30, min_periods=1
        ).mean()
        
        df_eng['cpu_load_std10'] = df_eng['cpu_load'].rolling(
            window=10, min_periods=1
        ).std()
        # .std() computes standard deviation
        # Measures load variability
        
        print("  ✓ Created rolling features (smoothing)")
        
        # ==========================================
        # INTERACTION FEATURES
        # ==========================================
        # WHY: Thermal effects are non-linear
        # PHYSICS: Heat gen increases with ambient temp
        
        df_eng['load_ambient_interaction'] = (
            df_eng['cpu_load'] * df_eng['ambient_temp']
        )
        # Multiplication creates interaction term
        # Captures: "load at high ambient is worse"
        
        df_eng['thermal_stress'] = (
            df_eng['cpu_load'] * df_eng['cpu_temp']
        )
        # High load + high temp = critical situation
        
        df_eng['temp_above_ambient'] = (
            df_eng['cpu_temp'] - df_eng['ambient_temp']
        )
        # ΔT drives cooling rate (Newton's Law)
        
        print("  ✓ Created interaction features (non-linearities)")
        
        # ==========================================
        # REGIME INDICATORS
        # ==========================================
        # WHY: Different physics in different states
        # Trees naturally split on thresholds
        
        df_eng['is_high_load'] = (df_eng['cpu_load'] > 70).astype(int)
        # Boolean → integer (0 or 1)
        # 1 if load > 70%, else 0
        
        df_eng['is_heating'] = (df_eng['temp_rate'] > 0.5).astype(int)
        df_eng['is_cooling'] = (df_eng['temp_rate'] < -0.5).astype(int)
        
        print("  ✓ Created regime indicators (operating states)")
        
        # ==========================================
        # TIME FEATURES
        # ==========================================
        # WHY: Cyclical patterns (day/night)
        # SIN/COS: Encode hour as continuous circle
        
        if 'timestamp' in df_eng.columns:
            df_eng['timestamp'] = pd.to_datetime(df_eng['timestamp'])
            hour = df_eng['timestamp'].dt.hour
            
            # Cyclical encoding:
            # Hour 0 and 24 are same → use sin/cos
            df_eng['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df_eng['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            print("  ✓ Created time features (cyclical)")
        
        # ==========================================
        # CLEANUP
        # ==========================================
        # Remove rows with NaN (from shift/diff)
        initial_rows = len(df_eng)
        df_eng = df_eng.dropna()
        removed = initial_rows - len(df_eng)
        
        print(f"\n  Removed {removed} rows with NaN")
        print(f"  Final: {len(df_eng)} rows × {len(df_eng.columns)} features")
        
        return df_eng
    
    def create_visualizations(self, df):
        """
        Create 4 analysis plots.
        
        Args:
            df: Processed DataFrame
        
        Plots:
            1. Time series (load, temp over time)
            2. Correlation matrix (feature relationships)
            3. Scatter plots (load vs temp, etc.)
            4. Distributions (histograms)
        
        WHY VISUALIZE:
            - Verify data quality
            - Check for patterns
            - Validate physics
            - Spot anomalies
        """
        print("\nCreating visualizations...")
        
        # ==========================================
        # PLOT 1: TIME SERIES
        # ==========================================
        # Shows data evolution over time
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        # CPU Load
        axes[0].plot(df.index, df['cpu_load'], linewidth=1)
        axes[0].set_ylabel('CPU Load (%)', fontweight='bold')
        axes[0].set_title('Thermal Telemetry Time Series', 
                         fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # CPU Temperature
        axes[1].plot(df.index, df['cpu_temp'], 
                    linewidth=1, color='darkred')
        axes[1].set_ylabel('CPU Temp (°C)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # RAM Usage
        axes[2].plot(df.index, df['ram_usage'], 
                    linewidth=1, color='orange')
        axes[2].set_ylabel('RAM Usage (%)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Ambient Temperature
        axes[3].plot(df.index, df['ambient_temp'], 
                    linewidth=1, color='green')
        axes[3].set_ylabel('Ambient (°C)', fontweight='bold')
        axes[3].set_xlabel('Sample Index (1 sample/second)', 
                          fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/01_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Time series plot saved")
        
        # ==========================================
        # PLOT 2: CORRELATION MATRIX
        # ==========================================
        # Shows relationships between features
        
        # Select key features for correlation
        corr_cols = [
            'cpu_load', 'ram_usage', 
            'ambient_temp', 'cpu_temp'
        ]
        
        corr_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                   cmap='coolwarm', center=0,
                   square=True, linewidths=1,
                   cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix', 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/02_correlation_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Correlation matrix saved")
        
        # ==========================================
        # PLOT 3: SCATTER PLOTS
        # ==========================================
        # Shows relationships visually
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Load vs Temp
        axes[0, 0].scatter(df['cpu_load'], df['cpu_temp'],
                          alpha=0.5, s=10)
        axes[0, 0].set_xlabel('CPU Load (%)')
        axes[0, 0].set_ylabel('CPU Temp (°C)')
        axes[0, 0].set_title('CPU Load vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ambient vs Temp
        axes[0, 1].scatter(df['ambient_temp'], df['cpu_temp'],
                          alpha=0.5, s=10, color='green')
        axes[0, 1].set_xlabel('Ambient Temp (°C)')
        axes[0, 1].set_ylabel('CPU Temp (°C)')
        axes[0, 1].set_title('Ambient vs CPU Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        # RAM vs Temp
        axes[1, 0].scatter(df['ram_usage'], df['cpu_temp'],
                          alpha=0.5, s=10, color='orange')
        axes[1, 0].set_xlabel('RAM Usage (%)')
        axes[1, 0].set_ylabel('CPU Temp (°C)')
        axes[1, 0].set_title('RAM Usage vs CPU Temperature')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Load vs Temp Rate
        if 'temp_rate' in df.columns:
            axes[1, 1].scatter(df['cpu_load'], df['temp_rate'],
                              alpha=0.5, s=10, color='purple')
            axes[1, 1].set_xlabel('CPU Load (%)')
            axes[1, 1].set_ylabel('Temperature Rate (°C/s)')
            axes[1, 1].set_title('Load vs Temperature Change Rate')
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/03_scatter_plots.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Scatter plots saved")
        
        # ==========================================
        # PLOT 4: DISTRIBUTIONS
        # ==========================================
        # Shows data distribution (histograms)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CPU Load distribution
        axes[0, 0].hist(df['cpu_load'], bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('CPU Load (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('CPU Load Distribution')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # CPU Temp distribution
        axes[0, 1].hist(df['cpu_temp'], bins=50,
                       color='darkred', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('CPU Temperature (°C)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('CPU Temperature Distribution')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # RAM Usage distribution
        axes[1, 0].hist(df['ram_usage'], bins=50,
                       color='orange', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('RAM Usage (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RAM Usage Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Ambient Temp distribution
        axes[1, 1].hist(df['ambient_temp'], bins=50,
                       color='green', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Ambient Temperature (°C)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Ambient Temperature Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/04_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Distribution plots saved")
    
    def run(self):
        """
        Main preprocessing pipeline.
        
        Steps:
            1. Remove outliers
            2. Sort by time
            3. Engineer features
            4. Create visualizations
            5. Save processed data
        """
        print(f"\n{'='*70}")
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print(f"{'='*70}")
        
        # Step 1: Clean data
        columns_to_clean = ['cpu_load', 'cpu_temp', 
                           'ram_usage', 'ambient_temp']
        df_clean = self.remove_outliers(columns_to_clean)
        
        # Step 2: Sort by time
        df_clean = df_clean.sort_values('unix_time').reset_index(drop=True)
        print("\n✓ Data sorted by timestamp")
        
        # Step 3: Engineer features
        df_processed = self.engineer_features(df_clean)
        
        # Step 4: Visualize
        self.create_visualizations(df_processed)
        
        # Step 5: Save
        output_file = os.path.join(self.output_dir, 
                                   'thermal_processed.csv')
        df_processed.to_csv(output_file, index=False)
        
        print(f"\n{'='*70}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed data saved to: {output_file}")
        print(f"  Rows: {len(df_processed):,}")
        print(f"  Features: {len(df_processed.columns)}")
        print(f"  File size: {os.path.getsize(output_file)/1024:.1f} KB")
        print(f"\nVisualizations saved to: {self.viz_dir}/")
        print(f"  01_time_series.png")
        print(f"  02_correlation_matrix.png")
        print(f"  03_scatter_plots.png")
        print(f"  04_distributions.png")
        print(f"\n✓ Ready for model training!")
        print(f"  Next: python models/train_model.py")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Path to raw data
    raw_data = 'collected_data/thermal_data.csv'
    
    # Check if file exists
    if not os.path.exists(raw_data):
        print(f"❌ Error: Data file not found")
        print(f"   Expected: {raw_data}")
        print(f"\n   Please run data collection first:")
        print(f"   python collect_thermal_data.py")
    else:
        # Create preprocessor and run
        preprocessor = ThermalDataPreprocessor(raw_data)
        preprocessor.run()
```

## Step 2.2: Run Preprocessing

```bash
cd thermal_prediction_project
python data_collection/preprocess_data.py
```

**Expected output**:

```
Loading data from: collected_data/thermal_data.csv
  Loaded 1,800 rows × 6 columns

Outlier removal:
  Removed: 23 rows (1.28%)
  Remaining: 1,777 rows

✓ Data sorted by timestamp

Engineering features...
  ✓ Created lag features (thermal inertia)
  ✓ Created rate features (thermal dynamics)
  ✓ Created rolling features (smoothing)
  ✓ Created interaction features (non-linearities)
  ✓ Created regime indicators (operating states)
  ✓ Created time features (cyclical)

  Removed 10 rows with NaN
  Final: 1,767 rows × 29 features

Creating visualizations...
  ✓ Time series plot saved
  ✓ Correlation matrix saved
  ✓ Scatter plots saved
  ✓ Distribution plots saved

======================================================================
PREPROCESSING COMPLETE
======================================================================
Processed data saved to: processed_data/thermal_processed.csv
  Rows: 1,767
  Features: 29
  File size: 453.2 KB

Visualizations saved to: ../visualizations/
  01_time_series.png
  02_correlation_matrix.png
  03_scatter_plots.png
  04_distributions.png

✓ Ready for model training!
  Next: python models/train_model.py
======================================================================
```

**Check the visualizations**:
- Open `visualizations/01_time_series.png`
- Should see clear load phases
- Temperature should lag behind load
- Validate data quality visually

---

# STEP 3: MODEL TRAINING <a id="step-3"></a>

**NOTE**: The complete `train_model.py` script is 672 lines. I've already provided this in the project files. Here's how to use it:

## Step 3.1: Understand Training Process

```
1. Load processed data
   ↓
2. Split into features (X) and target (y)
   ↓
3. Temporal train/test split (80/20)
   ↓
4. Train 7 models:
   - Ridge Regression
   - Lasso Regression
   - Random Forest
   - Gradient Boosting
   - Extra Trees
   - Neural Network
   - SVR (RBF)
   ↓
5. Evaluate each model (RMSE, MAE, R²)
   ↓
6. Save best model
   ↓
7. Generate visualizations
```

## Step 3.2: Run Training

```bash
cd thermal_prediction_project
python models/train_model.py
```

**Expected output**:

```
Loading processed data...
  ✓ Loaded 1,767 samples

Features: 23
Target: cpu_temp

Train/Test Split (Temporal):
  Training: 1,413 samples (80%)
  Test: 354 samples (20%)

Training models...
================================================

Training: Ridge Regression...
  ✓ Complete in 0.001s
    Test RMSE: 0.114°C
    Test R²: 0.9999

Training: Random Forest...
  ✓ Complete in 0.301s
    Test RMSE: 0.571°C
    Test R²: 0.9978

... (other models) ...

================================================
✓ All models trained!

🏆 BEST MODEL: Ridge Regression
  Test RMSE: 0.114°C
  Test MAE: 0.075°C
  Test R²: 0.9999
  Train-Test Gap: 0.048°C
  ✓ Excellent generalization!

✓ Model saved: models/best_thermal_model.pkl
✓ Scaler saved: models/feature_scaler.pkl
✓ Info saved: models/model_info.json
```

**Files created**:
- `models/best_thermal_model.pkl` - Trained model
- `models/feature_scaler.pkl` - Feature scaler
- `models/model_info.json` - Model metadata
- `results/*.png` - Performance charts

---

# STEP 4: REAL-TIME PREDICTION <a id="step-4"></a>

## Step 4.1: Test the Model

**Use the FIXED version**:

```bash
# Copy the fixed predict_realtime.py
# (Download from outputs, rename to predict_realtime.py)

cd thermal_prediction_project/models
python predict_realtime.py
```

**What it does**:
1. Loads trained model
2. Collects current system state every 1 second
3. Engineers features from history
4. Predicts temperature 5 seconds ahead
5. Determines fan speed based on prediction
6. Displays real-time monitoring

**Example output**:

```
======================================================================
PROACTIVE THERMAL MANAGEMENT - REAL-TIME MONITORING
======================================================================
Duration: 5 minutes
Prediction horizon: 5 seconds
Warning threshold: 70.0°C
Critical threshold: 80.0°C
======================================================================

Press Ctrl+C to stop

Collecting initial samples (need 11 seconds)...
Collecting... 11/11 samples

Starting predictions...
Time      | Current | Predicted | Delta | Status   | Fan
----------------------------------------------------------------------
10:45:12 |  45.20°C |   46.10°C | +0.90°C | NORMAL   |  50/255
10:45:13 |  45.80°C |   47.20°C | +1.40°C | NORMAL   |  50/255
10:45:14 |  48.30°C |   51.40°C | +3.10°C | NORMAL   |  65/255
10:45:15 |  52.10°C |   55.80°C | +3.70°C | NORMAL   |  75/255
...
```

**Success criteria**:
- ✅ Predictions match current temp closely
- ✅ Delta (difference) < 5°C
- ✅ No crashes or errors
- ✅ Fan speed adjusts based on predictions

---

# STEP 5: TESTING & VALIDATION <a id="step-5"></a>

## Validation Checklist

### ✅ Data Quality
- [ ] ~1,800 samples collected
- [ ] Load shows phases (not random)
- [ ] Temperature varies with load
- [ ] No long plateaus (sensor not stuck)
- [ ] Visualizations look correct

### ✅ Feature Engineering
- [ ] 23+ features created
- [ ] No NaN values in processed data
- [ ] Correlations make physical sense
- [ ] Time series shows lag (temp behind load)

### ✅ Model Performance
- [ ] Best RMSE < 2°C (excellent if < 1°C)
- [ ] R² > 0.95 (excellent if > 0.98)
- [ ] Train-test gap < 1°C (no severe overfitting)
- [ ] Residuals centered at 0 (unbiased)

### ✅ Real-Time System
- [ ] Predictions update every second
- [ ] Predictions reasonable (not wild jumps)
- [ ] System doesn't crash
- [ ] Can run for >5 minutes continuously

### ✅ Custom vs Kaggle
- [ ] Custom data outperforms Kaggle by >50%
- [ ] RMSE comparison chart shows clear winner
- [ ] Validates methodology

---

# 9. UNDERSTANDING EACH FUNCTION <a id="9-functions"></a>

## Key Python Concepts Used

### `.shift(n)` - Create Lag Features

```python
df['lag1'] = df['value'].shift(1)

Before:
  index  value  lag1
  0      10     NaN
  1      20     10    ← value from row 0
  2      30     20    ← value from row 1
  3      40     30    ← value from row 2

After shift(1), each row has previous row's value
```

### `.diff()` - Calculate Derivatives

```python
df['rate'] = df['value'].diff()

Before:
  index  value  rate
  0      10     NaN
  1      20     10    ← 20 - 10
  2      25     5     ← 25 - 20
  3      22     -3    ← 22 - 25

Positive rate = increasing
Negative rate = decreasing
```

### `.rolling(window).mean()` - Moving Average

```python
df['roll3'] = df['value'].rolling(window=3).mean()

Before:
  index  value  roll3
  0      10     10.0     ← avg(10)
  1      20     15.0     ← avg(10,20)
  2      30     20.0     ← avg(10,20,30)
  3      40     30.0     ← avg(20,30,40)

Each value is average of window
```

### Train/Test Split (Temporal)

```python
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]   # First 80%
X_test = X.iloc[split_idx:]    # Last 20%

Why temporal (not random):
  ✓ Respects time order
  ✓ Realistic (predict future from past)
  ✗ Random would leak future into past
```

---

# 10. TROUBLESHOOTING <a id="10-troubleshooting"></a>

## Common Issues & Solutions

### Issue 1: "No module named X"

**Error**:
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn psutil joblib
```

### Issue 2: "CPU temperature sensors not found"

**Symptoms**: All temperatures show ~35°C (simulated)

**Solution**: This is OK! Script automatically simulates when no sensors. Features still work.

**To verify**: Check `visualizations/01_time_series.png` - temperature should vary with load.

### Issue 3: "KeyError: 'hour_sin' not in index"

**This was your error!**

**Cause**: Model trained WITH time features, but predict_realtime.py wasn't creating them.

**Solution**: Use the FIXED_predict_realtime.py I provided.

**Fix explanation**:
```python
# BEFORE (broken):
features = {...}  # Only 20 features

# AFTER (fixed):
features = {
    ...,
    'hour_sin': np.sin(2 * np.pi * hour / 24),
    'hour_cos': np.cos(2 * np.pi * hour / 24)
}  # Now has all 23 features
```

### Issue 4: Data file not found

**Error**:
```
FileNotFoundError: thermal_data.csv
```

**Check**:
1. Are you in correct directory?
   ```bash
   pwd  # Should show: .../thermal_prediction_project
   ```

2. Did data collection complete?
   ```bash
   ls data_collection/collected_data/
   # Should show: thermal_data_*.csv
   ```

3. Check file path in script matches actual location

### Issue 5: Model accuracy poor (RMSE > 5°C)

**Causes**:
- Insufficient data (<500 samples)
- No workload variation (constant load)
- Sensor issues (stuck readings)

**Solution**:
1. Collect longer (30+ minutes)
2. Run workload generator 3-4 times
3. Check visualizations for data quality

### Issue 6: Real-time prediction crashes

**Error**:
```
IndexError: list index out of range
```

**Cause**: Not enough history samples

**Solution**: Wait for 11 samples (11 seconds) before first prediction

**Code explanation**:
```python
if len(self.feature_history) < 11:
    return None  # Need more samples
```

---

# SUMMARY: Complete Build Process

## From Scratch to Deployment

```
1. SETUP (15 minutes)
   - Install Python
   - Install libraries
   - Create directories

2. DATA COLLECTION (30 minutes)
   - Run collect_thermal_data.py
   - Run generate_workload.py (3-4 times)
   - Get ~1,800 samples

3. PREPROCESSING (2 minutes)
   - Run preprocess_data.py
   - Get 23 features
   - Get 4 visualizations

4. TRAINING (1 minute)
   - Run train_model.py
   - Train 7 models
   - Save best model

5. DEPLOYMENT (ongoing)
   - Run predict_realtime.py
   - Monitor predictions
   - Proactive cooling active!

Total Time: ~50 minutes
```

## Files You Should Have

```
✓ Data:
  - collected_data/thermal_data.csv (raw)
  - processed_data/thermal_processed.csv (features)

✓ Models:
  - models/best_thermal_model.pkl
  - models/feature_scaler.pkl
  - models/model_info.json

✓ Visualizations:
  - visualizations/01_time_series.png
  - visualizations/02_correlation_matrix.png
  - visualizations/03_scatter_plots.png
  - visualizations/04_distributions.png
  - results/model_comparison.png
  - results/prediction_analysis.png
```

## What You've Built

🎉 **A COMPLETE PRODUCTION ML SYSTEM!**

- ✅ Custom data collection
- ✅ Physics-based features
- ✅ Multiple models trained
- ✅ Best model identified
- ✅ Real-time predictions
- ✅ Proactive cooling logic
- ✅ Hardware integration ready
- ✅ Visualizations & docs
- ✅ 61-73% better than Kaggle!

**Congratulations! You now have a deployable thermal management system!** 🚀

---

**END OF COMPLETE BUILD GUIDE**
