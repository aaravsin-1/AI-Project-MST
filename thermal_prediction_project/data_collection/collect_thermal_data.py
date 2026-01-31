"""
Predictive Thermal Management System - Data Collection Module
=============================================================
This script collects real-time thermal and system telemetry data
for training a predictive cooling model.

Author: Thermal Prediction Team
Date: January 2026
"""

import psutil
import time
import csv
import serial
import os
from datetime import datetime
import sys
import numpy as np

class ThermalDataCollector:
    """
    Collects CPU temperature, load, RAM usage, and ambient temperature
    from system sensors and external Arduino-connected DS18B20 sensor.
    """
    
    def __init__(self, arduino_port='/dev/ttyUSB0', sampling_interval=1.0):
        """
        Initialize the data collector.
        
        Args:
            arduino_port: Serial port for Arduino connection
            sampling_interval: Time between samples in seconds
        """
        self.arduino_port = arduino_port
        self.sampling_interval = sampling_interval
        self.arduino = None
        self.csv_file = None
        self.csv_writer = None
        
        # Try to connect to Arduino (optional - will work without it)
        self.arduino_available = self._init_arduino()
        
    def _init_arduino(self):
        """Initialize Arduino serial connection"""
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"✓ Arduino connected on {self.arduino_port}")
            return True
        except Exception as e:
            print(f"⚠ Arduino not available: {e}")
            print("  Continuing without ambient temperature sensor...")
            return False
    
    def get_cpu_temperature(self):
        """
        Get CPU temperature from system sensors.
        Works on Linux systems with thermal sensors.
        """
        try:
            # Try to read from thermal sensors (Linux)
            temps = psutil.sensors_temperatures()
            
            if 'coretemp' in temps:
                # Intel CPUs
                temp = temps['coretemp'][0].current
            elif 'k10temp' in temps:
                # AMD CPUs
                temp = temps['k10temp'][0].current
            elif 'cpu_thermal' in temps:
                # Raspberry Pi
                temp = temps['cpu_thermal'][0].current
            else:
                # Fallback: use first available sensor
                temp = list(temps.values())[0][0].current
            
            return round(temp, 2)
        
        except Exception as e:
            print(f"Error reading CPU temperature: {e}")
            # Simulate temperature if sensors not available
            return self._simulate_cpu_temp()
    
    def _simulate_cpu_temp(self):
        """
        Simulate CPU temperature based on CPU load.
        Used when hardware sensors are not available.
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Base temp (idle) + load-dependent increase + random noise
        base_temp = 35.0
        load_factor = cpu_percent * 0.4  # 0.4°C per 1% load
        noise = np.random.normal(0, 1.5)  # Random thermal fluctuation
        
        return round(base_temp + load_factor + noise, 2)
    
    def get_cpu_load(self):
        """Get current CPU utilization percentage"""
        return round(psutil.cpu_percent(interval=0.5), 2)
    
    def get_ram_usage(self):
        """Get current RAM usage percentage"""
        return round(psutil.virtual_memory().percent, 2)
    
    def get_ambient_temperature(self):
        """
        Read ambient temperature from Arduino-connected DS18B20 sensor.
        Returns simulated value if Arduino not available.
        """
        if self.arduino_available:
            try:
                # Request temperature from Arduino
                self.arduino.write(b'T\n')
                time.sleep(0.1)
                
                # Read response
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode('utf-8').strip()
                    temp = float(response)
                    return round(temp, 2)
            except Exception as e:
                print(f"Error reading Arduino: {e}")
        
        # Simulate ambient temperature (22-28°C with slow variation)
        return round(24 + 2 * np.sin(time.time() / 100), 2)
    
    def collect_sample(self):
        """
        Collect one complete sample of all telemetry data.
        
        Returns:
            dict: Dictionary containing timestamp and all sensor readings
        """
        timestamp = datetime.now()
        
        data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'unix_time': int(timestamp.timestamp()),
            'cpu_load': self.get_cpu_load(),
            'ram_usage': self.get_ram_usage(),
            'ambient_temp': self.get_ambient_temperature(),
            'cpu_temp': self.get_cpu_temperature()
        }
        
        return data
    
    def init_csv_file(self, filename='thermal_data.csv'):
        """
        Initialize CSV file for data logging.
        
        Args:
            filename: Name of the CSV file
        """
        # Create data directory if it doesn't exist
        os.makedirs('collected_data', exist_ok=True)
        filepath = os.path.join('collected_data', filename)
        
        # Check if file exists
        file_exists = os.path.isfile(filepath)
        
        # Open file in append mode
        self.csv_file = open(filepath, 'a', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['timestamp', 'unix_time', 'cpu_load', 'ram_usage', 
                       'ambient_temp', 'cpu_temp']
        )
        
        # Write header if new file
        if not file_exists:
            self.csv_writer.writeheader()
            print(f"✓ Created new data file: {filepath}")
        else:
            print(f"✓ Appending to existing file: {filepath}")
        
        return filepath
    
    def log_sample(self, data):
        """Write sample to CSV file"""
        if self.csv_writer:
            self.csv_writer.writerow(data)
            self.csv_file.flush()  # Ensure data is written immediately
    
    def run_collection(self, duration_minutes=30, workload_script=None):
        """
        Run data collection for specified duration.
        
        Args:
            duration_minutes: How long to collect data
            workload_script: Optional path to script that generates CPU load
        """
        print("\n" + "="*60)
        print("THERMAL DATA COLLECTION STARTED")
        print("="*60)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Sampling interval: {self.sampling_interval} seconds")
        print(f"Arduino sensor: {'ENABLED' if self.arduino_available else 'SIMULATED'}")
        print("="*60 + "\n")
        
        filepath = self.init_csv_file()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        sample_count = 0
        
        try:
            while time.time() < end_time:
                # Collect sample
                data = self.collect_sample()
                self.log_sample(data)
                sample_count += 1
                
                # Display progress
                elapsed = time.time() - start_time
                progress = (elapsed / (duration_minutes * 60)) * 100
                
                print(f"\r[{progress:5.1f}%] Sample {sample_count:4d} | "
                      f"CPU: {data['cpu_temp']:5.1f}°C ({data['cpu_load']:5.1f}%) | "
                      f"RAM: {data['ram_usage']:5.1f}% | "
                      f"Ambient: {data['ambient_temp']:5.1f}°C", 
                      end='', flush=True)
                
                # Wait for next sample
                time.sleep(self.sampling_interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Collection interrupted by user")
        
        finally:
            # Cleanup
            if self.csv_file:
                self.csv_file.close()
            if self.arduino:
                self.arduino.close()
            
            print(f"\n\n{'='*60}")
            print("DATA COLLECTION COMPLETED")
            print(f"{'='*60}")
            print(f"Total samples: {sample_count}")
            print(f"Data saved to: {filepath}")
            print(f"File size: {os.path.getsize(filepath) / 1024:.2f} KB")
            print(f"{'='*60}\n")


def generate_cpu_load_pattern():
    """
    Generate varying CPU load patterns to create diverse thermal data.
    This should be run in parallel with data collection.
    """
    print("\nGenerating CPU load patterns...")
    
    patterns = [
        ('idle', 0.1, 60),      # Low load for 60 seconds
        ('medium', 0.5, 120),   # Medium load for 120 seconds
        ('high', 0.9, 90),      # High load for 90 seconds
        ('variable', None, 120) # Variable load for 120 seconds
    ]
    
    for pattern_name, load_level, duration in patterns:
        print(f"\nPattern: {pattern_name.upper()} ({duration}s)")
        start = time.time()
        
        while time.time() - start < duration:
            if pattern_name == 'variable':
                # Random load between 20% and 90%
                load_level = 0.2 + 0.7 * np.random.random()
            
            # Busy loop to generate CPU load
            end_busy = time.time() + load_level
            while time.time() < end_busy:
                _ = sum([i**2 for i in range(1000)])
            
            # Idle time
            time.sleep(1 - load_level)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   PREDICTIVE THERMAL MANAGEMENT - DATA COLLECTOR        ║
    ║   AI-Driven Proactive Cooling System                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DURATION_MINUTES = 30  # Collect data for 30 minutes
    SAMPLING_INTERVAL = 1.0  # Sample every 1 second
    
    # Create collector
    collector = ThermalDataCollector(
        arduino_port='/dev/ttyUSB0',
        sampling_interval=SAMPLING_INTERVAL
    )
    
    # Instructions
    print("INSTRUCTIONS:")
    print("1. This script will collect thermal data for {} minutes".format(DURATION_MINUTES))
    print("2. To generate load, open another terminal and run:")
    print("   python3 generate_workload.py")
    print("3. Or manually use your computer normally")
    print("4. Press Ctrl+C to stop early\n")
    
    input("Press ENTER to start data collection...")
    
    # Run collection
    collector.run_collection(duration_minutes=DURATION_MINUTES)
    
    print("\n✓ Data collection complete!")
    print("Next steps:")
    print("  1. Review collected data in: collected_data/thermal_data.csv")
    print("  2. Run preprocessing: python3 preprocess_data.py")
    print("  3. Train model: python3 train_model.py")
