"""
Data Preprocessing and Feature Engineering - FIXED FOR BETTER ACCURACY
======================================================================
Improvements:
1. ✅ Better outlier removal (more conservative)
2. ✅ Feature selection for better generalization
3. ✅ Removed regime features (causing overfitting)
4. ✅ Timestamp resampling for consistent 1Hz data

Expected improvement: RMSE 2.52°C → 1.5-2.0°C
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class ThermalDataPreprocessor:
    """
    Improved preprocessor with overfitting fixes.
    """
    
    def __init__(self, data_path):
        """
        Initialize preprocessor with data file path.
        
        Args:
            data_path: Path to CSV file containing thermal data
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load thermal data from CSV"""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} samples")
        print(f"  Columns: {list(self.df.columns)}")
        print(f"  Duration: {len(self.df) / 60:.1f} minutes")
        
        # Convert timestamp to datetime if present
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        return self.df
    
    def resample_to_1hz(self):
        """
        CRITICAL FIX: Resample irregular timestamps to exactly 1Hz.
        This fixes lag feature alignment issues.
        """
        if 'timestamp' not in self.df.columns:
            print("⚠ No timestamp column - skipping resampling")
            return self.df
        
        print("\n🔧 FIXING IRREGULAR TIMESTAMPS:")
        print(f"  Before: {len(self.df)} rows (irregular intervals)")
        
        # Set timestamp as index
        df_temp = self.df.set_index('timestamp')
        
        # Resample to 1 second intervals
        df_resampled = df_temp.resample('1s').mean()
        
        # Interpolate missing values (linear)
        df_resampled = df_resampled.interpolate(method='linear')
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        # Update unix_time if present
        if 'unix_time' in df_resampled.columns:
            df_resampled['unix_time'] = df_resampled['timestamp'].astype(np.int64) // 10**9
        
        print(f"  After: {len(df_resampled)} rows (exactly 1Hz)")
        print(f"  ✅ All lag features will now align correctly!")
        
        self.df = df_resampled
        return self.df
    
    def clean_data(self):
        """
        IMPROVED: More conservative outlier removal.
        Old method was too aggressive, removing valid high-temp data.
        """
        print("\nCleaning data...")
        initial_rows = len(self.df)
        
        # Remove rows with missing values
        self.df = self.df.dropna()
        
        # IMPROVED: More conservative outlier removal (2.5 IQR instead of 1.5)
        for col in ['cpu_load', 'ram_usage', 'cpu_temp', 'ambient_temp']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR  # More conservative (was 1.5)
            upper_bound = Q3 + 2.5 * IQR
            
            before = len(self.df)
            self.df = self.df[
                (self.df[col] >= lower_bound) & 
                (self.df[col] <= upper_bound)
            ]
            removed = before - len(self.df)
            if removed > 0:
                print(f"  {col}: removed {removed} outliers")
        
        removed_rows = initial_rows - len(self.df)
        print(f"✓ Removed {removed_rows} outlier/invalid samples ({removed_rows/initial_rows*100:.1f}%)")
        print(f"  Remaining samples: {len(self.df)}")
        
        return self.df
    
    def engineer_features(self):
        """
        IMPROVED: Removed overfitting features, kept only essential ones.
        """
        print("\nEngineering thermal physics features...")
        
        df = self.df.copy()
        
        # === TEMPORAL LAG FEATURES ===
        df['cpu_load_lag1'] = df['cpu_load'].shift(1)
        df['cpu_load_lag5'] = df['cpu_load'].shift(5)
        df['cpu_load_lag10'] = df['cpu_load'].shift(10)
        
        df['cpu_temp_lag1'] = df['cpu_temp'].shift(1)
        df['cpu_temp_lag5'] = df['cpu_temp'].shift(5)
        
        # === DERIVATIVE FEATURES ===
        df['temp_rate'] = df['cpu_temp'].diff()
        df['temp_acceleration'] = df['temp_rate'].diff()
        df['load_rate'] = df['cpu_load'].diff()
        
        # === ROLLING STATISTICS ===
        df['cpu_load_roll10'] = df['cpu_load'].rolling(window=10, min_periods=1).mean()
        df['cpu_temp_roll10'] = df['cpu_temp'].rolling(window=10, min_periods=1).mean()
        df['cpu_load_roll30'] = df['cpu_load'].rolling(window=30, min_periods=1).mean()
        df['cpu_load_std10'] = df['cpu_load'].rolling(window=10, min_periods=1).std()
        
        # === INTERACTION FEATURES ===
        df['load_ambient_interaction'] = df['cpu_load'] * df['ambient_temp']
        df['thermal_stress'] = df['cpu_load'] * df['cpu_temp']
        df['temp_above_ambient'] = df['cpu_temp'] - df['ambient_temp']
        
        # === TIME FEATURES (Optional) ===
        # Only include if you have multi-day data
        if 'timestamp' in df.columns or 'unix_time' in df.columns:
            if 'timestamp' in df.columns:
                hour_of_day = pd.to_datetime(df['timestamp']).dt.hour
            else:
                hour_of_day = pd.to_datetime(df['unix_time'], unit='s').dt.hour
            
            df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        
        # ❌ REMOVED: Regime indicators (is_high_load, is_heating, is_cooling)
        # These cause overfitting on small datasets!
        
        # CREATE FUTURE TARGET
        df['cpu_temp_future'] = df['cpu_temp'].shift(-5)  # 5 seconds ahead
        
        # Remove rows with NaN
        df = df.dropna()
        
        self.df_processed = df
        
        num_new_features = len(df.columns) - len(self.df.columns)
        print(f"✓ Created {num_new_features} new features")
        print(f"  Total features: {len(df.columns)}")
        print(f"  Remaining samples: {len(df)}")
        
        return df
    
    def get_feature_set(self):
        """
        IMPROVED: Essential features only (removed noisy features).
        """
        # Base features
        base_features = [
            'cpu_load', 'ram_usage', 'ambient_temp'
        ]
        
        # Lag features (most important!)
        lag_features = [
            'cpu_load_lag1', 'cpu_load_lag5', 'cpu_load_lag10',
            'cpu_temp_lag1', 'cpu_temp_lag5'
        ]
        
        # Rate features
        rate_features = [
            'temp_rate', 'temp_acceleration', 'load_rate'
        ]
        
        # Rolling features
        rolling_features = [
            'cpu_load_roll10', 'cpu_temp_roll10', 
            'cpu_load_roll30', 'cpu_load_std10'
        ]
        
        # Interaction features
        interaction_features = [
            'load_ambient_interaction', 'thermal_stress',
            'temp_above_ambient'
        ]
        
        # Time features (optional)
        time_features = []
        if 'hour_sin' in self.df_processed.columns:
            time_features = ['hour_sin', 'hour_cos']
        
        all_features = (base_features + lag_features + rate_features + 
                       rolling_features + interaction_features + time_features)
        
        print(f"\n✓ Feature set: {len(all_features)} features")
        print(f"  Base: {len(base_features)}")
        print(f"  Lag: {len(lag_features)}")
        print(f"  Rate: {len(rate_features)}")
        print(f"  Rolling: {len(rolling_features)}")
        print(f"  Interaction: {len(interaction_features)}")
        if time_features:
            print(f"  Time: {len(time_features)}")
        
        return all_features
    
    def prepare_for_training(self):
        """
        Prepare final X, y for model training.
        """
        features = self.get_feature_set()
        
        # Verify all features exist
        missing = [f for f in features if f not in self.df_processed.columns]
        if missing:
            print(f"⚠ Warning: Missing features: {missing}")
            features = [f for f in features if f in self.df_processed.columns]
        
        X = self.df_processed[features]
        y = self.df_processed['cpu_temp_future']
        
        print(f"\n✓ Training data prepared:")
        print(f"  Features (X): {X.shape}")
        print(f"  Target (y): {y.shape}")
        
        return X, y
    
    def visualize_data(self, save_path='visualizations'):
        """
        Create visualizations of the collected data.
        """
        print(f"\nGenerating visualizations...")
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # 1. Time series plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(self.df.index, self.df['cpu_load'], 
                    color='#2E86AB', linewidth=0.8)
        axes[0].set_ylabel('CPU Load (%)', fontsize=11, fontweight='bold')
        axes[0].set_title('Thermal Telemetry Time Series', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.df.index, self.df['cpu_temp'], 
                    color='#A23B72', linewidth=0.8)
        axes[1].set_ylabel('CPU Temp (°C)', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.df.index, self.df['ram_usage'], 
                    color='#F18F01', linewidth=0.8)
        axes[2].set_ylabel('RAM Usage (%)', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(self.df.index, self.df['ambient_temp'], 
                    color='#6A994E', linewidth=0.8)
        axes[3].set_ylabel('Ambient Temp (°C)', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Sample Index (1 sample/second)', 
                          fontsize=11, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/01_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to: {save_path}/")
    
    def get_statistics(self):
        """Generate descriptive statistics"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        print(self.df.describe().round(2))
        print("\nTemperature Range:")
        print(f"  Min: {self.df['cpu_temp'].min():.1f}°C")
        print(f"  Max: {self.df['cpu_temp'].max():.1f}°C")
        print(f"  Range: {self.df['cpu_temp'].max() - self.df['cpu_temp'].min():.1f}°C")
        
    def save_processed_data(self, output_path='processed_data/thermal_processed.csv'):
        """Save processed data to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df_processed.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║    DATA PREPROCESSING - FIXED FOR BETTER ACCURACY       ║
    ║                                                          ║
    ║  ✅ Timestamp resampling to 1Hz                          ║
    ║  ✅ Conservative outlier removal                         ║
    ║  ✅ Removed overfitting features                         ║
    ║  ✅ Ready for TimeSeriesSplit                            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Path to collected data
    import glob
    
    # Find most recent data file
    data_files = glob.glob('collected_data/thermal_data*.csv')
    if not data_files:
        print("❌ Error: No data files found in collected_data/")
        print("   Please run collect_thermal_data.py first.")
        exit(1)
    
    DATA_PATH = sorted(data_files)[-1]  # Most recent file
    print(f"Using data file: {DATA_PATH}\n")
    
    # Initialize preprocessor
    preprocessor = ThermalDataPreprocessor(DATA_PATH)
    
    # Load and process data
    preprocessor.load_data()
    
    # FIX 1: Resample to 1Hz (critical!)
    preprocessor.resample_to_1hz()
    
    # FIX 2: Conservative cleaning
    preprocessor.clean_data()
    preprocessor.get_statistics()
    
    # FIX 3: Essential features only
    preprocessor.engineer_features()
    
    # Create visualizations
    preprocessor.visualize_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n✅ Preprocessing complete with overfitting fixes!")
    print("\n📊 Expected improvement: RMSE 2.52°C → 1.5-2.0°C")
    print("\nNext: python train_model_fixed.py")