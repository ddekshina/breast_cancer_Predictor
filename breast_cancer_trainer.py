# breast_cancer_trainer.py
"""
Breast Cancer ML Model - Training Module
Handles model training, evaluation, and saving for production deployment.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BreastCancerTrainer:
    """
    Training module for breast cancer prediction model.
    Handles data loading, model training, evaluation, and persistence.
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.training_metadata = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        """Load and prepare the breast cancer dataset."""
        print("🔬 Loading Wisconsin Breast Cancer Dataset...")
        
        # Load dataset
        data = load_breast_cancer()
        
        # Create DataFrame
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['target'] = data.target
        
        # Store feature names for later use
        self.feature_names = list(data.feature_names)
        
        # Separate features and target
        self.X = self.df[self.feature_names]
        self.y = self.df['target']
        
        print(f"✅ Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"   Classes: {dict(zip(data.target_names, [sum(self.y == i) for i in [0,1]]))}")
        
        return self.X, self.y
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale the data for training."""
        print("📊 Preparing training data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y
        )
        
        # Fit scaler on training data and transform both sets
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data prepared: Train({self.X_train.shape[0]}), Test({self.X_test.shape[0]})")
        
    def train_models(self):
        """Train multiple models and select the best one."""
        print("🤖 Training multiple models...")
        
        # Model configurations
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5, 
                random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000, solver='lbfgs'
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, probability=True, 
                C=1.0, gamma='scale'
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"  Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train, 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            # Test evaluation
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            test_auc = roc_auc_score(self.y_test, y_pred_proba)
            test_accuracy = (y_pred == self.y_test).mean()
            
            results[name] = {
                'model': model,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'test_auc': test_auc,
                'test_accuracy': test_accuracy
            }
            
            print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    Test AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Select best model based on test AUC
        best_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        self.training_results = results
        
        print(f"🏆 Best model: {best_name} (AUC: {results[best_name]['test_auc']:.4f})")
        
    def hyperparameter_tuning(self):
        """Fine-tune the best model."""
        print(f"🔧 Hyperparameter tuning for {self.best_model_name}...")
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            },
            'SVM': {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.1, 1],
                'kernel': ['rbf']
            }
        }
        
        param_grid = param_grids.get(self.best_model_name, {})
        
        if param_grid:
            # Create base model
            if self.best_model_name == 'RandomForest':
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            elif self.best_model_name == 'LogisticRegression':
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            else:  # SVM
                base_model = SVC(random_state=42, probability=True)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"   Improved AUC: {grid_search.best_score_:.4f}")
        
    def evaluate_final_model(self):
        """Comprehensive evaluation of the final model."""
        print("📈 Final model evaluation...")
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        final_auc = roc_auc_score(self.y_test, y_pred_proba)
        final_accuracy = (y_pred == self.y_test).mean()
        
        print(f"🎯 Final Performance:")
        print(f"   AUC Score: {final_auc:.4f}")
        print(f"   Accuracy: {final_accuracy:.4f}")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(
            self.y_test, y_pred, 
            target_names=['Malignant', 'Benign'],
            digits=4
        ))
        
        # Store metadata
        self.training_metadata = {
            'model_name': self.best_model_name,
            'final_auc': final_auc,
            'final_accuracy': final_accuracy,
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_training_samples': len(self.X_train)
        }
        
    def save_model(self, model_name='breast_cancer_model'):
        """Save the trained model, scaler, and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        # Package everything needed for prediction
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': self.training_metadata
        }
        
        # Save the package
        joblib.dump(model_package, filepath)
        
        # Also save as latest model
        latest_path = os.path.join(self.model_dir, 'latest_model.pkl')
        joblib.dump(model_package, latest_path)
        
        print(f"💾 Model saved:")
        print(f"   Timestamped: {filepath}")
        print(f"   Latest: {latest_path}")
        
        return filepath
        
    def train_full_pipeline(self):
        """Execute the complete training pipeline."""
        print("🚀 Starting Breast Cancer Model Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare data
            self.prepare_data()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Hyperparameter tuning
            self.hyperparameter_tuning()
            
            # Step 5: Final evaluation
            self.evaluate_final_model()
            
            # Step 6: Save model
            model_path = self.save_model()
            
            print("\n" + "=" * 60)
            print("🎉 Training completed successfully!")
            print(f"🏆 Best Model: {self.best_model_name}")
            print(f"📊 Performance: AUC={self.training_metadata['final_auc']:.4f}")
            print(f"💾 Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training execution."""
    trainer = BreastCancerTrainer()
    model_path = trainer.train_full_pipeline()
    return model_path

if __name__ == "__main__":
    main()