"""
Data Preprocessing and Feature Engineering
==========================================
Prepares thermal data for machine learning model training.
Includes physics-based feature engineering for thermal inertia.

FIXED: hour_of_day is NOT saved to CSV (only hour_sin/hour_cos)
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
    Preprocesses thermal telemetry data and engineers physics-based features.
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
        return self.df
    
    def clean_data(self):
        """
        Clean data by handling missing values and outliers.
        """
        print("\nCleaning data...")
        initial_rows = len(self.df)
        
        # Remove rows with missing values
        self.df = self.df.dropna()
        
        # Remove outliers using IQR method
        for col in ['cpu_load', 'ram_usage', 'cpu_temp', 'ambient_temp']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df = self.df[
                (self.df[col] >= lower_bound) & 
                (self.df[col] <= upper_bound)
            ]
        
        removed_rows = initial_rows - len(self.df)
        print(f"✓ Removed {removed_rows} outlier/invalid samples")
        print(f"  Remaining samples: {len(self.df)}")
        
        return self.df
    
    def engineer_features(self):
        """
        Create physics-based features that capture thermal dynamics.
        
        Thermal inertia means current temperature depends on:
        1. Past CPU loads (heat accumulation)
        2. Rate of temperature change (heating/cooling)
        3. Environmental conditions
        """
        print("\nEngineering thermal physics features...")
        
        df = self.df.copy()
        
        # === TEMPORAL LAG FEATURES ===
        # Capture thermal inertia: past loads affect current temperature
        
        # CPU load history (rolling windows)
        df['cpu_load_lag1'] = df['cpu_load'].shift(1)  # 1 second ago
        df['cpu_load_lag5'] = df['cpu_load'].shift(5)  # 5 seconds ago
        df['cpu_load_lag10'] = df['cpu_load'].shift(10)  # 10 seconds ago
        
        # Temperature history
        df['cpu_temp_lag1'] = df['cpu_temp'].shift(1)
        df['cpu_temp_lag5'] = df['cpu_temp'].shift(5)
        
        # === DERIVATIVE FEATURES ===
        # Rate of temperature change indicates heating/cooling regime
        
        df['temp_rate'] = df['cpu_temp'].diff()  # °C per second
        df['temp_acceleration'] = df['temp_rate'].diff()  # Change in rate
        
        # Load rate of change
        df['load_rate'] = df['cpu_load'].diff()
        
        # === ROLLING STATISTICS ===
        # Capture average thermal behavior over time windows
        
        # 10-second rolling average (smooths noise)
        df['cpu_load_roll10'] = df['cpu_load'].rolling(window=10, min_periods=1).mean()
        df['cpu_temp_roll10'] = df['cpu_temp'].rolling(window=10, min_periods=1).mean()
        
        # 30-second rolling average (longer-term trend)
        df['cpu_load_roll30'] = df['cpu_load'].rolling(window=30, min_periods=1).mean()
        
        # Rolling standard deviation (load variability)
        df['cpu_load_std10'] = df['cpu_load'].rolling(window=10, min_periods=1).std()
        
        # === INTERACTION FEATURES ===
        # Combine multiple factors affecting heat generation
        
        # Heat generation proxy: load × ambient temp
        df['load_ambient_interaction'] = df['cpu_load'] * df['ambient_temp']
        
        # Thermal stress: high load with already high temp
        df['thermal_stress'] = df['cpu_load'] * df['cpu_temp']
        
        # Temperature delta from ambient
        df['temp_above_ambient'] = df['cpu_temp'] - df['ambient_temp']
        
        # === CYCLICAL TIME FEATURES ===
        # Time of day can affect ambient conditions
        
        if 'unix_time' in df.columns or 'timestamp' in df.columns:
            if 'timestamp' in df.columns:
                hour_of_day = pd.to_datetime(df['timestamp']).dt.hour
            else:
                hour_of_day = pd.to_datetime(df['unix_time'], unit='s').dt.hour
            
            # Create sin/cos encoding (DON'T save hour_of_day!)
            df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        
        # === REGIME INDICATORS ===
        # Binary features for operating regimes
        
        df['is_high_load'] = (df['cpu_load'] > 70).astype(int)
        df['is_heating'] = (df['temp_rate'] > 0.5).astype(int)
        df['is_cooling'] = (df['temp_rate'] < -0.5).astype(int)
        
        # CREATE FUTURE TARGET
        df['cpu_temp_future'] = df['cpu_temp'].shift(-5)  # 5 seconds ahead
        
        # Remove rows with NaN from lag features and future target
        df = df.dropna()
        
        self.df_processed = df
        
        num_new_features = len(df.columns) - len(self.df.columns)
        print(f"✓ Created {num_new_features} new features")
        print(f"  Total features: {len(df.columns)}")
        print(f"  Remaining samples: {len(df)}")
        
        return df
    
    def get_feature_set(self):
        """
        Define which features to use for model training.
        
        Returns:
            list: Feature column names
        """
        # Original features
        base_features = [
            'cpu_load', 'ram_usage', 'ambient_temp'
        ]
        
        # Lag features (thermal inertia)
        lag_features = [
            'cpu_load_lag1', 'cpu_load_lag5', 'cpu_load_lag10',
            'cpu_temp_lag1', 'cpu_temp_lag5'
        ]
        
        # Rate features (heating/cooling dynamics)
        rate_features = [
            'temp_rate', 'temp_acceleration', 'load_rate'
        ]
        
        # Rolling features (smoothed trends)
        rolling_features = [
            'cpu_load_roll10', 'cpu_temp_roll10', 
            'cpu_load_roll30', 'cpu_load_std10'
        ]
        
        # Interaction features
        interaction_features = [
            'load_ambient_interaction', 'thermal_stress',
            'temp_above_ambient'
        ]
        
        # Regime indicators
        regime_features = [
            'is_high_load', 'is_heating', 'is_cooling'
        ]
        
        # Temporal features (ONLY sin/cos, NOT hour_of_day!)
        temporal_features = [
            'hour_sin', 'hour_cos'
        ]
        
        # Combine all
        all_features = (base_features + lag_features + rate_features + 
                       rolling_features + interaction_features + 
                       regime_features + temporal_features)
        
        return all_features
    
    def prepare_training_data(self, target='cpu_temp_future'):
        """
        Prepare features and target for model training.
        
        Args:
            target: Target variable name (default: cpu_temp_future)
            
        Returns:
            X, y: Features and target arrays
        """
        features = self.get_feature_set()
        
        # Verify all features exist
        available_features = [f for f in features if f in self.df_processed.columns]
        missing_features = set(features) - set(available_features)
        
        if missing_features:
            print(f"⚠ Warning: Missing features: {missing_features}")
            features = available_features
        
        X = self.df_processed[features]
        
        # Use future target if available, otherwise current temp
        if target in self.df_processed.columns:
            y = self.df_processed[target]
            print(f"✓ Using FUTURE target ({target})")
        else:
            y = self.df_processed['cpu_temp']
            print(f"⚠ Using CURRENT target (cpu_temp)")
        
        print(f"\nPreparing training data:")
        print(f"  Features: {len(features)}")
        print(f"  Feature list: {features}")
        print(f"  Samples: {len(X)}")
        print(f"  Target: {target if target in self.df_processed.columns else 'cpu_temp'}")
        
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
        
        # 2. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_data = self.df[['cpu_load', 'ram_usage', 'ambient_temp', 'cpu_temp']]
        corr_matrix = corr_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].scatter(self.df['cpu_load'], self.df['cpu_temp'], 
                          alpha=0.3, s=10, c=self.df['ambient_temp'],
                          cmap='viridis')
        axes[0, 0].set_xlabel('CPU Load (%)')
        axes[0, 0].set_ylabel('CPU Temp (°C)')
        axes[0, 0].set_title('CPU Load vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(self.df['ambient_temp'], self.df['cpu_temp'],
                          alpha=0.3, s=10, c=self.df['cpu_load'],
                          cmap='plasma')
        axes[0, 1].set_xlabel('Ambient Temp (°C)')
        axes[0, 1].set_ylabel('CPU Temp (°C)')
        axes[0, 1].set_title('Ambient vs CPU Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].scatter(self.df['ram_usage'], self.df['cpu_temp'],
                          alpha=0.3, s=10, c=self.df['cpu_load'],
                          cmap='magma')
        axes[1, 0].set_xlabel('RAM Usage (%)')
        axes[1, 0].set_ylabel('CPU Temp (°C)')
        axes[1, 0].set_title('RAM Usage vs CPU Temperature')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Temperature rate vs load
        if 'temp_rate' in self.df_processed.columns:
            axes[1, 1].scatter(self.df_processed['cpu_load'], 
                              self.df_processed['temp_rate'],
                              alpha=0.3, s=10, c=self.df_processed['cpu_temp'],
                              cmap='coolwarm')
            axes[1, 1].set_xlabel('CPU Load (%)')
            axes[1, 1].set_ylabel('Temperature Rate (°C/s)')
            axes[1, 1].set_title('Load vs Temperature Change Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/03_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].hist(self.df['cpu_load'], bins=50, color='#2E86AB', 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('CPU Load (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('CPU Load Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(self.df['cpu_temp'], bins=50, color='#A23B72',
                       alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('CPU Temperature (°C)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('CPU Temperature Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(self.df['ram_usage'], bins=50, color='#F18F01',
                       alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('RAM Usage (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RAM Usage Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(self.df['ambient_temp'], bins=30, color='#6A994E',
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Ambient Temperature (°C)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Ambient Temperature Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/04_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved 4 visualization files to: {save_path}/")
    
    def get_statistics(self):
        """Generate descriptive statistics"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        print(self.df.describe().round(2))
        print("\nData Quality:")
        print(f"  Missing values: {self.df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {self.df.duplicated().sum()}")
        
    def save_processed_data(self, output_path='processed_data/thermal_processed.csv'):
        """Save processed data to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df_processed.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        # Verify saved columns
        saved_df = pd.read_csv(output_path)
        print(f"  Columns in saved file: {len(saved_df.columns)}")
        if 'hour_of_day' in saved_df.columns:
            print(f"  ⚠ WARNING: hour_of_day was saved (should not be!)")
        else:
            print(f"  ✓ hour_of_day correctly excluded")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      DATA PREPROCESSING & FEATURE ENGINEERING           ║
    ║   Physics-Based Thermal Model Preparation                ║
    ║   FIXED: hour_of_day NOT saved (only hour_sin/cos)       ║
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
    print(f"Using most recent data file: {DATA_PATH}\n")
    
    # Initialize preprocessor
    preprocessor = ThermalDataPreprocessor(DATA_PATH)
    
    # Load and process data
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.get_statistics()
    
    # Engineer features
    preprocessor.engineer_features()
    
    # Create visualizations
    preprocessor.visualize_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n✓ Preprocessing complete!")
    print("\n🔥 IMPORTANT: You must RETRAIN the model now!")
    print("   Run: python models/train_model.py")
    print("\n   The model needs to be retrained without hour_of_day feature.")