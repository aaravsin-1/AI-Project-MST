
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
