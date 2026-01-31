"""
CPU Workload Generator
=====================
Generates controlled CPU load patterns for thermal data collection.
Run this script in parallel with the data collector.
"""

import time
import multiprocessing
import numpy as np
import sys

def cpu_stress_worker(load_level, duration):
    """
    Worker function that generates CPU load.
    
    Args:
        load_level: Target CPU utilization (0.0 to 1.0)
        duration: How long to maintain this load (seconds)
    """
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Busy period
        busy_start = time.time()
        while time.time() - busy_start < load_level:
            # Perform computationally intensive work
            _ = sum([i**2 for i in range(10000)])
        
        # Idle period
        time.sleep(1.0 - load_level)


def run_workload_pattern(pattern_name, load_level, duration, num_cores=None):
    """
    Execute a workload pattern across multiple CPU cores.
    
    Args:
        pattern_name: Name of the pattern
        load_level: Target CPU load (0.0 to 1.0)
        duration: Duration in seconds
        num_cores: Number of cores to use (None = all)
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    
    print(f"\n{'='*60}")
    print(f"Pattern: {pattern_name}")
    print(f"Target Load: {load_level*100:.0f}%")
    print(f"Duration: {duration}s")
    print(f"CPU Cores: {num_cores}")
    print(f"{'='*60}")
    
    # Create worker processes
    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(
            target=cpu_stress_worker,
            args=(load_level, duration)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        print(f"Core {i+1}/{num_cores} completed", end='\r')
    
    print(f"\n✓ Pattern '{pattern_name}' completed")


def main():
    """
    Execute a series of workload patterns to generate diverse thermal data.
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          CPU WORKLOAD GENERATOR                         ║
    ║   For Thermal Data Collection                            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    num_cores = multiprocessing.cpu_count()
    print(f"System CPU cores: {num_cores}")
    print(f"\nThis script will generate controlled CPU load patterns.")
    print(f"Run the data collector script simultaneously for best results.\n")
    
    # Define workload patterns
    patterns = [
        # (name, load_level, duration_seconds)
        ("PHASE 1: System Idle", 0.05, 60),
        ("PHASE 2: Light Load", 0.25, 90),
        ("PHASE 3: Medium Load", 0.50, 120),
        ("PHASE 4: Heavy Load", 0.75, 90),
        ("PHASE 5: Maximum Load", 0.95, 60),
        ("PHASE 6: Cooling Down", 0.10, 120),
    ]
    
    print("Workload Schedule:")
    total_duration = 0
    for i, (name, load, duration) in enumerate(patterns, 1):
        total_duration += duration
        print(f"  {i}. {name:30s} - {load*100:3.0f}% for {duration:3d}s")
    
    print(f"\nTotal duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")
    
    input("\nPress ENTER to start workload generation...")
    
    try:
        start_time = time.time()
        
        for pattern_name, load_level, duration in patterns:
            run_workload_pattern(pattern_name, load_level, duration, num_cores)
            
            # Brief pause between patterns
            time.sleep(2)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("WORKLOAD GENERATION COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Workload generation interrupted by user")
        sys.exit(0)


def generate_stress_test():
    """
    Alternative: Generate a single sustained high load.
    Useful for thermal stress testing.
    """
    print("Running CPU stress test...")
    duration = 300  # 5 minutes
    load_level = 0.95
    
    run_workload_pattern("STRESS TEST", load_level, duration)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stress":
        generate_stress_test()
    else:
        main()
