"""
System Flowchart Generator
=========================
Creates comprehensive flowcharts for the thermal prediction system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_system_flowchart():
    """
    Create comprehensive system architecture flowchart.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Colors
    color_data = '#2E86AB'
    color_process = '#A23B72'
    color_model = '#F18F01'
    color_deploy = '#06A77D'
    color_hardware = '#6D2E46'
    
    # Title
    ax.text(5, 19.5, 'Predictive Thermal Management System', 
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))
    
    # Layer 1: Data Collection
    y_pos = 18
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5,
                                boxstyle="round,pad=0.1", 
                                facecolor=color_data, edgecolor='black', linewidth=2, alpha=0.3))
    
    ax.text(5, y_pos, 'DATA COLLECTION LAYER', 
            fontsize=13, fontweight='bold', ha='center')
    
    # Data sources
    boxes_y = y_pos - 0.5
    ax.add_patch(FancyBboxPatch((1, boxes_y-0.3), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=color_data, edgecolor='black', linewidth=1.5))
    ax.text(2, boxes_y, 'System Sensors\n(CPU, RAM, Temp)', 
            fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((3.5, boxes_y-0.3), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=color_data, edgecolor='black', linewidth=1.5))
    ax.text(4.5, boxes_y, 'Arduino Sensor\n(DS18B20 Ambient)', 
            fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((6, boxes_y-0.3), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=color_data, edgecolor='black', linewidth=1.5))
    ax.text(7, boxes_y, 'Workload\nGenerator', 
            fontsize=9, ha='center', va='center')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos-1.2), (5, y_pos-1.8),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # Layer 2: Preprocessing
    y_pos = 15.5
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color_process, edgecolor='black', linewidth=2, alpha=0.3))
    
    ax.text(5, y_pos, 'PREPROCESSING & FEATURE ENGINEERING', 
            fontsize=13, fontweight='bold', ha='center')
    
    features_y = y_pos - 0.5
    features = [
        '• Data Cleaning\n• Outlier Removal',
        '• Lag Features\n• Rate Features',
        '• Rolling Stats\n• Interactions'
    ]
    
    for i, feat in enumerate(features):
        x_pos = 1.5 + i * 2.5
        ax.add_patch(FancyBboxPatch((x_pos-0.6, features_y-0.35), 1.8, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_process, edgecolor='black', linewidth=1.5))
        ax.text(x_pos + 0.3, features_y, feat, fontsize=8, ha='center', va='center')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos-1.2), (5, y_pos-1.8),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # Layer 3: Model Training
    y_pos = 13
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-1.3), 9, 2.0,
                                boxstyle="round,pad=0.1",
                                facecolor=color_model, edgecolor='black', linewidth=2, alpha=0.3))
    
    ax.text(5, y_pos+0.3, 'MACHINE LEARNING LAYER', 
            fontsize=13, fontweight='bold', ha='center')
    
    # Model types
    models_y = y_pos - 0.3
    models = [
        'Random\nForest',
        'Gradient\nBoosting',
        'Neural\nNetwork',
        'SVR'
    ]
    
    for i, model in enumerate(models):
        x_pos = 1.2 + i * 2.2
        ax.add_patch(FancyBboxPatch((x_pos-0.5, models_y-0.35), 1.5, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_model, edgecolor='black', linewidth=1.5))
        ax.text(x_pos + 0.25, models_y, model, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Optimization box
    opt_y = y_pos - 0.9
    ax.add_patch(FancyBboxPatch((2, opt_y-0.25), 6, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#FFD700', edgecolor='black', linewidth=1.5))
    ax.text(5, opt_y, '⚙ Hyperparameter Optimization | Cross-Validation | Feature Selection', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos-1.7), (5, y_pos-2.3),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # Layer 4: Evaluation
    y_pos = 10.2
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#90EE90', edgecolor='black', linewidth=2, alpha=0.5))
    
    ax.text(5, y_pos, 'MODEL EVALUATION & COMPARISON', 
            fontsize=13, fontweight='bold', ha='center')
    
    metrics_y = y_pos - 0.5
    metrics = ['RMSE', 'MAE', 'R² Score', 'Feature Importance']
    
    for i, metric in enumerate(metrics):
        x_pos = 1.5 + i * 2
        ax.add_patch(FancyBboxPatch((x_pos-0.5, metrics_y-0.25), 1.5, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor='black', linewidth=1.5))
        ax.text(x_pos + 0.25, metrics_y, metric, fontsize=9, ha='center', va='center')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos-1.2), (5, y_pos-1.8),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # Layer 5: Deployment
    y_pos = 7.9
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color_deploy, edgecolor='black', linewidth=2, alpha=0.3))
    
    ax.text(5, y_pos, 'REAL-TIME INFERENCE LAYER', 
            fontsize=13, fontweight='bold', ha='center')
    
    deploy_y = y_pos - 0.5
    steps = [
        'Collect\nState',
        'Engineer\nFeatures',
        'Predict\nTemp',
        'Make\nDecision'
    ]
    
    for i, step in enumerate(steps):
        x_pos = 1.5 + i * 2
        ax.add_patch(FancyBboxPatch((x_pos-0.5, deploy_y-0.25), 1.5, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_deploy, edgecolor='black', linewidth=1.5))
        ax.text(x_pos + 0.25, deploy_y, step, fontsize=9, ha='center', va='center', fontweight='bold')
        
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_pos+1.2, deploy_y), xytext=(x_pos+0.7, deploy_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos-1.2), (5, y_pos-1.8),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # Layer 6: Hardware Control
    y_pos = 5.6
    
    ax.add_patch(FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color_hardware, edgecolor='black', linewidth=2, alpha=0.3))
    
    ax.text(5, y_pos, 'PHYSICAL ACTUATION LAYER', 
            fontsize=13, fontweight='bold', ha='center', color='white')
    
    hardware_y = y_pos - 0.5
    hardware = ['Arduino\nController', 'PWM\nControl', 'Cooling\nFan']
    
    for i, hw in enumerate(hardware):
        x_pos = 2 + i * 2.5
        ax.add_patch(FancyBboxPatch((x_pos-0.6, hardware_y-0.25), 1.8, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_hardware, edgecolor='white', linewidth=2))
        ax.text(x_pos + 0.3, hardware_y, hw, fontsize=9, ha='center', va='center', 
               color='white', fontweight='bold')
        
        if i < len(hardware) - 1:
            ax.annotate('', xy=(x_pos+1.4, hardware_y), xytext=(x_pos+0.9, hardware_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    
    # Result box
    result_y = y_pos - 1.3
    ax.add_patch(FancyBboxPatch((2, result_y-0.3), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor='#90EE90', edgecolor='black', linewidth=2))
    ax.text(5, result_y, '✓ PROACTIVE COOLING ACHIEVED\nPrevents overheating before it occurs', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Side annotations
    # Innovation highlights
    innovations_x = 9.5
    innovations = [
        (18.5, 'Custom Data\nCollection'),
        (15.5, 'Physics-Based\nFeatures'),
        (13, 'Multi-Model\nComparison'),
        (7.9, 'Real-Time\nPrediction'),
        (5.6, 'Hardware\nIntegration')
    ]
    
    for y, text in innovations:
        ax.add_patch(mpatches.FancyBboxPatch((innovations_x-0.4, y-0.25), 0.8, 0.5,
                                             boxstyle="round,pad=0.03",
                                             facecolor='#FFD700', edgecolor='black', linewidth=1))
        ax.text(innovations_x, y, text, fontsize=7, ha='center', va='center', fontweight='bold')
    
    # Legend
    legend_y = 2.5
    ax.text(5, legend_y+1, 'System Components', fontsize=12, fontweight='bold', ha='center')
    
    legend_items = [
        (color_data, 'Data Collection'),
        (color_process, 'Preprocessing'),
        (color_model, 'ML Training'),
        (color_deploy, 'Deployment'),
        (color_hardware, 'Hardware')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        x = 1.5 + i * 1.7
        ax.add_patch(mpatches.Rectangle((x-0.2, legend_y-0.15), 0.4, 0.3,
                                       facecolor=color, edgecolor='black'))
        ax.text(x, legend_y-0.5, label, fontsize=8, ha='center')
    
    # Performance metrics box
    metrics_box_y = 0.8
    ax.add_patch(FancyBboxPatch((1, metrics_box_y-0.5), 8, 0.9,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F4F8', edgecolor='black', linewidth=2))
    
    ax.text(5, metrics_box_y+0.15, 'Expected Performance', 
            fontsize=11, fontweight='bold', ha='center')
    ax.text(5, metrics_box_y-0.15, 'RMSE: ~1.2-1.5°C  |  R²: ~0.98  |  Real-time: <5ms inference  |  70-80% better than Kaggle data', 
            fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('/home/claude/thermal_prediction_project/documentation/system_flowchart.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ System flowchart saved: documentation/system_flowchart.png")


def create_data_flow_diagram():
    """
    Create detailed data flow diagram.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Data Processing Pipeline', 
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))
    
    # Stage 1: Raw Data
    y = 10
    ax.add_patch(FancyBboxPatch((1, y-0.4), 2.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFB6C1', edgecolor='black', linewidth=2))
    ax.text(2.25, y, 'Raw Telemetry\n1800 samples/30min', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(4.5, y), xytext=(3.7, y),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Stage 2: Cleaning
    ax.add_patch(FancyBboxPatch((4.5, y-0.4), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#98FB98', edgecolor='black', linewidth=2))
    ax.text(5.5, y, 'Data Cleaning\nOutlier Removal', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrow
    ax.annotate('', xy=(7.5, y), xytext=(6.7, y),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Stage 3: Clean Data
    ax.add_patch(FancyBboxPatch((7.5, y-0.4), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFD700', edgecolor='black', linewidth=2))
    ax.text(8.5, y, 'Clean Data\n~1750 samples', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Down arrow
    ax.annotate('', xy=(8.5, y-0.5), xytext=(8.5, y-1.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Feature Engineering
    y = 7.5
    ax.add_patch(FancyBboxPatch((2, y-1.2), 6, 2.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#E6E6FA', edgecolor='black', linewidth=2, alpha=0.7))
    
    ax.text(5, y+0.8, 'Feature Engineering (23 Features Created)', 
            fontsize=11, fontweight='bold', ha='center')
    
    # Feature categories
    features_data = [
        ('Lag Features\n(5)', 2.5, y+0.2),
        ('Rate Features\n(3)', 4.2, y+0.2),
        ('Rolling Stats\n(4)', 5.9, y+0.2),
        ('Interactions\n(3)', 2.5, y-0.5),
        ('Regime Flags\n(3)', 4.2, y-0.5),
        ('Base Features\n(3)', 5.9, y-0.5)
    ]
    
    for label, x, y_pos in features_data:
        ax.add_patch(FancyBboxPatch((x-0.5, y_pos-0.25), 1.4, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor='black', linewidth=1.5))
        ax.text(x + 0.2, y_pos, label, fontsize=8, ha='center', va='center')
    
    # Down arrow
    ax.annotate('', xy=(5, y-1.3), xytext=(5, y-2.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Train/Test Split
    y = 4.5
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 7, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFA07A', edgecolor='black', linewidth=2))
    ax.text(5, y, 'Train/Test Split (Temporal): 80% Train | 20% Test', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Split into two branches
    ax.annotate('', xy=(3, y-0.5), xytext=(4.5, y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, y-0.5), xytext=(5.5, y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Training path
    y = 3
    ax.add_patch(FancyBboxPatch((1, y-0.4), 3.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#87CEEB', edgecolor='black', linewidth=2))
    ax.text(2.75, y, 'Training Set\n~1400 samples\n7 Models Trained', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Testing path
    ax.add_patch(FancyBboxPatch((5.5, y-0.4), 3.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#DDA0DD', edgecolor='black', linewidth=2))
    ax.text(7.25, y, 'Testing Set\n~350 samples\nModel Evaluation', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Final result
    ax.annotate('', xy=(5, y-0.5), xytext=(2.75, y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5, y-0.5), xytext=(7.25, y-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    y = 1.5
    ax.add_patch(FancyBboxPatch((2, y-0.5), 6, 1,
                                boxstyle="round,pad=0.1",
                                facecolor='#90EE90', edgecolor='black', linewidth=3))
    ax.text(5, y+0.2, 'Best Model Selected', 
            fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(5, y-0.15, 'Random Forest: RMSE ~1.2-1.5°C, R² ~0.98', 
            fontsize=10, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('/home/claude/thermal_prediction_project/documentation/data_flow_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Data flow diagram saved: documentation/data_flow_diagram.png")


if __name__ == "__main__":
    print("Generating flowcharts...")
    create_system_flowchart()
    create_data_flow_diagram()
    print("\n✓ All flowcharts generated successfully!")
