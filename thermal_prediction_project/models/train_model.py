"""
Predictive Thermal Model Training
=================================
Trains multiple regression models and compares their performance
for CPU temperature prediction.

Innovation: Multi-model ensemble with physics-aware feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

class ThermalModelTrainer:
    """
    Advanced thermal prediction model trainer with multiple algorithms.
    """
    
    def __init__(self, data_path='processed_data/thermal_processed.csv'):
        """
        Initialize model trainer.
        
        Args:
            data_path: Path to processed data CSV
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load processed data"""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        return self.df
    
    def prepare_features(self):
        """
        Prepare feature set and target variable.
        """
        # Define feature set (excluding target and metadata)
        exclude_cols = ['timestamp', 'unix_time', 'cpu_temp']
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df['cpu_temp']
        
        print(f"\nFeature preparation:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(X)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        Uses temporal split to avoid data leakage.
        """
        print(f"\nSplitting data:")
        print(f"  Test size: {test_size*100}%")
        
        # Temporal split (last 20% for testing)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Testing samples: {len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """
        Initialize multiple regression models for comparison.
        """
        print("\nInitializing models...")
        
        self.models = {
            # === ENSEMBLE METHODS (Best for non-linear thermal dynamics) ===
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            ),
            
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            
            # === LINEAR MODELS (Baseline comparison) ===
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            
            'Lasso Regression': Lasso(
                alpha=0.1,
                random_state=42,
                max_iter=10000
            ),
            
            # === NEURAL NETWORK (Deep learning approach) ===
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            
            # === SUPPORT VECTOR MACHINE (Kernel-based) ===
            'SVR (RBF)': SVR(
                kernel='rbf',
                C=10,
                epsilon=0.1,
                gamma='scale'
            )
        }
        
        print(f"✓ Initialized {len(self.models)} models")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self):
        """
        Train all models and evaluate their performance.
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining: {name}")
            start_time = time.time()
            
            try:
                # Use scaled data for models that need it
                if name in ['Neural Network', 'SVR (RBF)', 'Ridge Regression', 'Lasso Regression']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred_train = model.predict(self.X_train_scaled)
                    y_pred_test = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)
                
                train_time = time.time() - start_time
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_time': train_time,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                }
                
                print(f"  ✓ Completed in {train_time:.2f}s")
                print(f"    Test RMSE: {test_rmse:.3f}°C")
                print(f"    Test MAE:  {test_mae:.3f}°C")
                print(f"    Test R²:   {test_r2:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
        
        print("\n" + "="*60)
    
    def optimize_best_model(self):
        """
        Perform hyperparameter tuning on the best performing model.
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        # Find best model based on test RMSE
        best_model_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        print(f"\nOptimizing: {best_model_name}")
        
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
            
            base_model = GradientBoostingRegressor(random_state=42)
        
        else:
            print("Skipping optimization for this model type")
            return None
        
        print("Running Grid Search...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        
        # Evaluate optimized model
        optimized_model = grid_search.best_estimator_
        y_pred_test = optimized_model.predict(self.X_test)
        
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"\nOptimized performance:")
        print(f"  Test RMSE: {test_rmse:.3f}°C")
        print(f"  Test MAE:  {test_mae:.3f}°C")
        print(f"  Test R²:   {test_r2:.4f}")
        
        # Store optimized model
        self.results[f'{best_model_name} (Optimized)'] = {
            'model': optimized_model,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        return optimized_model
    
    def get_feature_importance(self, model_name='Random Forest'):
        """
        Extract and visualize feature importance.
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def create_comparison_plots(self, save_path='results'):
        """
        Create comprehensive visualization comparing all models.
        """
        print("\nGenerating comparison plots...")
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.results.keys())
        train_rmse = [self.results[m]['train_rmse'] for m in model_names]
        test_rmse = [self.results[m]['test_rmse'] for m in model_names]
        test_mae = [self.results[m]['test_mae'] for m in model_names]
        test_r2 = [self.results[m]['test_r2'] for m in model_names]
        
        # RMSE Comparison
        x_pos = np.arange(len(model_names))
        axes[0, 0].bar(x_pos, test_rmse, color='#E63946', alpha=0.7, label='Test')
        axes[0, 0].bar(x_pos, train_rmse, color='#457B9D', alpha=0.5, label='Train')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('RMSE (°C)', fontweight='bold')
        axes[0, 0].set_title('Root Mean Squared Error', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE Comparison
        axes[0, 1].bar(x_pos, test_mae, color='#F18F01', alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('MAE (°C)', fontweight='bold')
        axes[0, 1].set_title('Mean Absolute Error', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² Comparison
        axes[1, 0].bar(x_pos, test_r2, color='#06A77D', alpha=0.7)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('R² Score', fontweight='bold')
        axes[1, 0].set_title('R² Score (Coefficient of Determination)', fontweight='bold')
        axes[1, 0].axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training Time
        train_times = [self.results[m]['train_time'] for m in model_names]
        axes[1, 1].bar(x_pos, train_times, color='#9D4EDD', alpha=0.7)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Time (seconds)', fontweight='bold')
        axes[1, 1].set_title('Training Time', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction vs Actual (Best Model)
        best_model_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        y_pred = self.results[best_model_name]['y_pred_test']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=10, c=self.y_test,
                       cmap='coolwarm')
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Temperature (°C)', fontweight='bold')
        axes[0].set_ylabel('Predicted Temperature (°C)', fontweight='bold')
        axes[0].set_title(f'{best_model_name}: Predicted vs Actual', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10, c=residuals,
                       cmap='RdYlGn_r')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Temperature (°C)', fontweight='bold')
        axes[1].set_ylabel('Residuals (°C)', fontweight='bold')
        axes[1].set_title('Residual Analysis', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance (if available)
        importance_df = self.get_feature_importance(best_model_name)
        
        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            top_n = 15
            top_features = importance_df.head(top_n)
            
            colors = plt.cm.viridis(np.linspace(0, 1, top_n))
            ax.barh(range(top_n), top_features['importance'], color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance Score', fontweight='bold')
            ax.set_title(f'Top {top_n} Feature Importances - {best_model_name}',
                        fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Time Series Prediction
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot last 500 samples for clarity
        plot_samples = min(500, len(self.y_test))
        
        ax.plot(range(plot_samples), self.y_test.iloc[:plot_samples].values,
               label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
        ax.plot(range(plot_samples), y_pred[:plot_samples],
               label='Predicted', color='#E63946', linewidth=1.5, alpha=0.8)
        ax.fill_between(range(plot_samples),
                        self.y_test.iloc[:plot_samples].values,
                        y_pred[:plot_samples],
                        alpha=0.3, color='gray', label='Error')
        
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('CPU Temperature (°C)', fontweight='bold')
        ax.set_title(f'{best_model_name}: Temporal Prediction Performance',
                    fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/temporal_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved 4 visualization files to: {save_path}/")
    
    def save_best_model(self, save_path='models'):
        """
        Save the best performing model.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Find best model
        best_model_name = min(self.results, key=lambda x: self.results[x]['test_rmse'])
        best_model = self.results[best_model_name]['model']
        
        # Save model and scaler
        model_path = f'{save_path}/best_thermal_model.pkl'
        scaler_path = f'{save_path}/feature_scaler.pkl'
        
        joblib.dump(best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save model info
        info = {
            'model_name': best_model_name,
            'test_rmse': self.results[best_model_name]['test_rmse'],
            'test_mae': self.results[best_model_name]['test_mae'],
            'test_r2': self.results[best_model_name]['test_r2'],
            'features': list(self.X_train.columns)
        }
        
        import json
        with open(f'{save_path}/model_info.json', 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"\n✓ Saved best model: {best_model_name}")
        print(f"  Model file: {model_path}")
        print(f"  Scaler file: {scaler_path}")
        print(f"  Performance: RMSE={info['test_rmse']:.3f}°C, R²={info['test_r2']:.4f}")
        
        return model_path
    
    def generate_report(self):
        """
        Generate comprehensive performance report.
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE REPORT")
        print("="*60)
        
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train RMSE': [self.results[m]['train_rmse'] for m in self.results],
            'Test RMSE': [self.results[m]['test_rmse'] for m in self.results],
            'Test MAE': [self.results[m]['test_mae'] for m in self.results],
            'Test R²': [self.results[m]['test_r2'] for m in self.results],
            'Train Time (s)': [self.results[m]['train_time'] for m in self.results]
        })
        
        results_df = results_df.sort_values('Test RMSE')
        
        print("\n" + results_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60)
        
        best = results_df.iloc[0]
        print(f"\nModel: {best['Model']}")
        print(f"Test RMSE: {best['Test RMSE']:.4f}°C")
        print(f"Test MAE:  {best['Test MAE']:.4f}°C")
        print(f"Test R²:   {best['Test R²']:.4f}")
        print(f"Training Time: {best['Train Time (s)']:.2f}s")
        
        # Save report
        results_df.to_csv('results/model_performance_report.csv', index=False)
        print(f"\n✓ Report saved to: results/model_performance_report.csv")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        THERMAL PREDICTION MODEL TRAINING                ║
    ║   Multi-Model Comparison & Optimization                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if processed data exists
    DATA_PATH = 'processed_data/thermal_processed.csv'
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Processed data not found at {DATA_PATH}")
        print("   Please run preprocess_data.py first.")
        exit(1)
    
    # Initialize trainer
    trainer = ThermalModelTrainer(DATA_PATH)
    
    # Load and prepare data
    trainer.load_data()
    X, y = trainer.prepare_features()
    trainer.split_data(X, y)
    
    # Train models
    trainer.initialize_models()
    trainer.train_models()
    
    # Optimize best model
    trainer.optimize_best_model()
    
    # Generate visualizations and reports
    trainer.create_comparison_plots()
    trainer.generate_report()
    
    # Save best model
    trainer.save_best_model()
    
    print("\n✓ Training complete!")
    print("Next steps:")
    print("  1. Review results in: results/")
    print("  2. Test real-time prediction: python3 predict_realtime.py")
