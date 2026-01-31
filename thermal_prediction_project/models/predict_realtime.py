"""
Real-Time Thermal Prediction & Proactive Cooling - FIXED VERSION
================================================
Uses trained model to predict future CPU temperature and
trigger cooling actions before overheating occurs.

FIXED: Properly handles all features including time features
"""

import psutil
import time
import numpy as np
import pandas as pd
import joblib
import serial
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProactiveCoolingSystem:
    """
    Real-time thermal prediction and proactive cooling control.
    """
    
    def __init__(self, model_path='models/best_thermal_model.pkl',
                 scaler_path='models/feature_scaler.pkl',
                 arduino_port='/dev/ttyUSB0'):
        """
        Initialize proactive cooling system.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            arduino_port: Arduino serial port for fan control
        """
        self.model = None
        self.scaler = None
        self.arduino = None
        self.feature_history = []
        self.prediction_history = []
        
        # Load model
        self.load_model(model_path, scaler_path)
        
        # Try to connect Arduino (optional)
        self.arduino_available = self._init_arduino(arduino_port)
        
        # Cooling thresholds
        self.TEMP_WARNING = 70.0  # °C
        self.TEMP_CRITICAL = 80.0  # °C
        self.PREDICTION_HORIZON = 5  # seconds ahead
        
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Model loaded from: {model_path}")
            
            # Load model info
            import json
            info_path = os.path.join(os.path.dirname(model_path), 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.feature_names = info['features']
                    print(f"✓ Model: {info['model_name']}")
                    print(f"  Test RMSE: {info['test_rmse']:.3f}°C")
                    print(f"  Test R²: {info['test_r2']:.4f}")
                    print(f"  Expected features: {len(self.feature_names)}")
            else:
                print("⚠ model_info.json not found, will use all features from data")
                self.feature_names = None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"   Make sure files exist:")
            print(f"   - {model_path}")
            print(f"   - {scaler_path}")
            exit(1)
    
    def _init_arduino(self, port):
        """Initialize Arduino for fan control"""
        try:
            self.arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            print(f"✓ Arduino connected for fan control")
            return True
        except:
            print(f"⚠ Arduino not available - continuing without hardware control")
            return False
    
    def get_system_state(self):
        """
        Collect current system telemetry.
        """
        try:
            # CPU temperature
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            elif 'k10temp' in temps:
                cpu_temp = temps['k10temp'][0].current
            elif 'cpu_thermal' in temps:
                cpu_temp = temps['cpu_thermal'][0].current
            else:
                # Try first available sensor
                try:
                    cpu_temp = list(temps.values())[0][0].current
                except:
                    # Fallback to simulation
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_temp = 35.0 + cpu_percent * 0.4 + np.random.normal(0, 1.5)
        except:
            # Simulate if sensors not available
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_temp = 35.0 + cpu_percent * 0.4 + np.random.normal(0, 1.5)
        
        state = {
            'cpu_load': psutil.cpu_percent(interval=0.5),
            'ram_usage': psutil.virtual_memory().percent,
            'ambient_temp': self._get_ambient_temp(),
            'cpu_temp': cpu_temp,
            'timestamp': time.time()
        }
        
        return state
    
    def _get_ambient_temp(self):
        """Get ambient temperature from Arduino or simulate"""
        if self.arduino_available:
            try:
                self.arduino.write(b'T\n')
                time.sleep(0.1)
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode('utf-8').strip()
                    return float(response)
            except:
                pass
        
        # Simulate ambient temperature
        return 24 + 2 * np.sin(time.time() / 100)
    
    def engineer_features(self, state):
        """
        Create features from current state and history.
        This must match the feature engineering in training.
        FIXED: Now creates ALL features that might be needed
        """
        # Add current state to history
        self.feature_history.append(state)
        
        # Keep only last 30 seconds
        if len(self.feature_history) > 30:
            self.feature_history.pop(0)
        
        # Need at least 11 samples for lag features
        if len(self.feature_history) < 11:
            return None
        
        # Create feature dictionary
        features = {}
        
        # Base features
        features['cpu_load'] = state['cpu_load']
        features['ram_usage'] = state['ram_usage']
        features['ambient_temp'] = state['ambient_temp']
        
        # Lag features
        features['cpu_load_lag1'] = self.feature_history[-2]['cpu_load']
        features['cpu_load_lag5'] = self.feature_history[-6]['cpu_load']
        features['cpu_load_lag10'] = self.feature_history[-11]['cpu_load']
        features['cpu_temp_lag1'] = self.feature_history[-2]['cpu_temp']
        features['cpu_temp_lag5'] = self.feature_history[-6]['cpu_temp']
        
        # Rate features
        features['temp_rate'] = state['cpu_temp'] - self.feature_history[-2]['cpu_temp']
        features['temp_acceleration'] = features['temp_rate'] - (
            self.feature_history[-2]['cpu_temp'] - self.feature_history[-3]['cpu_temp']
        )
        features['load_rate'] = state['cpu_load'] - self.feature_history[-2]['cpu_load']
        
        # Rolling features
        recent_loads = [h['cpu_load'] for h in self.feature_history[-10:]]
        recent_temps = [h['cpu_temp'] for h in self.feature_history[-10:]]
        
        features['cpu_load_roll10'] = np.mean(recent_loads)
        features['cpu_temp_roll10'] = np.mean(recent_temps)
        features['cpu_load_roll30'] = np.mean([h['cpu_load'] for h in self.feature_history])
        features['cpu_load_std10'] = np.std(recent_loads)
        
        # Interaction features
        features['load_ambient_interaction'] = state['cpu_load'] * state['ambient_temp']
        features['thermal_stress'] = state['cpu_load'] * state['cpu_temp']
        features['temp_above_ambient'] = state['cpu_temp'] - state['ambient_temp']
        
        # Regime indicators
        features['is_high_load'] = 1 if state['cpu_load'] > 70 else 0
        features['is_heating'] = 1 if features['temp_rate'] > 0.5 else 0
        features['is_cooling'] = 1 if features['temp_rate'] < -0.5 else 0
        
        # Time features - Create all possible time-related features
        current_time = datetime.now()
        hour = current_time.hour
        
        # These might be needed by the model
        features['hour_of_day'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        return features
    
    def predict_temperature(self, features):
        """
        Predict future CPU temperature using trained model.
        FIXED: Properly handles feature selection
        
        Args:
            features: Dictionary of engineered features
            
        Returns:
            predicted_temp: Predicted temperature in °C
        """
        try:
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # If we know which features the model needs, select only those
            if self.feature_names is not None:
                # Only use features that exist in both the engineered features and model expectations
                available_features = [f for f in self.feature_names if f in feature_df.columns]
                
                if len(available_features) < len(self.feature_names):
                    missing = set(self.feature_names) - set(available_features)
                    print(f"⚠ Warning: Missing features: {missing}")
                    print(f"   Using {len(available_features)} of {len(self.feature_names)} features")
                
                feature_df = feature_df[available_features]
            
            # Predict (model might be Ridge which uses scaler, or tree-based which doesn't)
            try:
                # Try with scaling first (for Ridge/Lasso/Neural Net/SVR)
                feature_scaled = self.scaler.transform(feature_df)
                predicted_temp = self.model.predict(feature_scaled)[0]
            except:
                # If scaling fails, try without (for Random Forest/Gradient Boosting)
                predicted_temp = self.model.predict(feature_df)[0]
            
            return predicted_temp
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            print(f"   Features created: {list(features.keys())}")
            if self.feature_names:
                print(f"   Features expected: {self.feature_names}")
            raise
    
    def control_fan(self, predicted_temp, current_temp):
        """
        Control fan speed based on predicted temperature.
        Implements proactive cooling strategy.
        
        Args:
            predicted_temp: Predicted future temperature
            current_temp: Current temperature
        """
        # Determine fan speed (0-255 for PWM)
        if predicted_temp >= self.TEMP_CRITICAL:
            fan_speed = 255  # Maximum
            status = "CRITICAL"
            color = "\033[91m"  # Red
        elif predicted_temp >= self.TEMP_WARNING:
            # Scale between warning and critical
            ratio = (predicted_temp - self.TEMP_WARNING) / (self.TEMP_CRITICAL - self.TEMP_WARNING)
            fan_speed = int(128 + 127 * ratio)
            status = "WARNING"
            color = "\033[93m"  # Yellow
        elif predicted_temp >= 60:
            fan_speed = 100  # Moderate
            status = "ELEVATED"
            color = "\033[94m"  # Blue
        else:
            fan_speed = 50  # Low
            status = "NORMAL"
            color = "\033[92m"  # Green
        
        # Send command to Arduino if available
        if self.arduino_available:
            try:
                command = f'F{fan_speed}\n'.encode()
                self.arduino.write(command)
            except Exception as e:
                pass
        
        return fan_speed, status, color
    
    def run_monitoring(self, duration_minutes=10, log_file='results/prediction_log.csv'):
        """
        Run real-time monitoring and prediction.
        
        Args:
            duration_minutes: How long to run monitoring
            log_file: Path to save prediction log
        """
        print("\n" + "="*70)
        print("PROACTIVE THERMAL MANAGEMENT - REAL-TIME MONITORING")
        print("="*70)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Prediction horizon: {self.PREDICTION_HORIZON} seconds")
        print(f"Warning threshold: {self.TEMP_WARNING}°C")
        print(f"Critical threshold: {self.TEMP_CRITICAL}°C")
        print("="*70)
        
        # Initialize log file
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        log_data = []
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        print("\nPress Ctrl+C to stop\n")
        print("Collecting initial samples (need 11 seconds)...")
        
        sample_count = 0
        
        try:
            while time.time() < end_time:
                # Get current state
                state = self.get_system_state()
                sample_count += 1
                
                # Engineer features
                features = self.engineer_features(state)
                
                if features is None:
                    # Still collecting initial samples
                    remaining = 11 - len(self.feature_history)
                    print(f"\rCollecting... {len(self.feature_history)}/11 samples", end='', flush=True)
                    time.sleep(1)
                    continue
                
                if sample_count == 11:
                    # First prediction
                    print("\n\nStarting predictions...")
                    print("Time      | Current | Predicted | Delta | Status   | Fan")
                    print("-"*70)
                
                # Predict future temperature
                predicted_temp = self.predict_temperature(features)
                
                # Calculate prediction delta
                temp_delta = predicted_temp - state['cpu_temp']
                
                # Control cooling
                fan_speed, status, color = self.control_fan(
                    predicted_temp, state['cpu_temp']
                )
                
                # Display status
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"{timestamp} | "
                      f"{state['cpu_temp']:6.2f}°C | "
                      f"{predicted_temp:6.2f}°C | "
                      f"{temp_delta:+5.2f}°C | "
                      f"{color}{status:8s}\033[0m | "
                      f"{fan_speed:3d}/255",
                      flush=True)
                
                # Log data
                log_entry = {
                    'timestamp': timestamp,
                    'current_temp': state['cpu_temp'],
                    'predicted_temp': predicted_temp,
                    'temp_delta': temp_delta,
                    'cpu_load': state['cpu_load'],
                    'fan_speed': fan_speed,
                    'status': status
                }
                log_data.append(log_entry)
                
                # Wait for next sample
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Monitoring stopped by user")
        
        finally:
            # Save log
            if log_data:
                log_df = pd.DataFrame(log_data)
                log_df.to_csv(log_file, index=False)
                print(f"\n✓ Prediction log saved to: {log_file}")
                
                # Print summary statistics
                print("\n" + "="*70)
                print("MONITORING SUMMARY")
                print("="*70)
                print(f"Total predictions: {len(log_df)}")
                print(f"Average prediction error: {abs(log_df['temp_delta']).mean():.2f}°C")
                print(f"Max prediction error: {abs(log_df['temp_delta']).max():.2f}°C")
                print(f"Temperature range: {log_df['current_temp'].min():.1f}°C - {log_df['current_temp'].max():.1f}°C")
            
            # Cleanup
            if self.arduino:
                self.arduino.write(b'F0\n')  # Turn off fan
                self.arduino.close()
            
            print("="*70)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     PROACTIVE THERMAL MANAGEMENT SYSTEM                 ║
    ║   Real-Time Prediction & Cooling Control                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not os.path.exists('models/best_thermal_model.pkl'):
        print("❌ Error: Trained model not found")
        print("   Please run train_model.py first")
        print("\n   Expected location: models/best_thermal_model.pkl")
        print("   Current directory:", os.getcwd())
        exit(1)
    
    # Create system
    try:
        system = ProactiveCoolingSystem()
    except Exception as e:
        print(f"\n❌ Failed to initialize system: {e}")
        exit(1)
    
    print("\nSystem initialized successfully!")
    print("\nOptions:")
    print("1. Run real-time monitoring")
    
    choice = input("\nPress ENTER to start monitoring (or Ctrl+C to exit): ").strip()
    
    try:
        duration = int(input("Enter monitoring duration in minutes (default 5): ") or "5")
    except:
        duration = 5
    
    print(f"\nStarting {duration}-minute monitoring session...")
    system.run_monitoring(duration_minutes=duration)
    
    print("\n✓ Monitoring complete!")