"""
Breast Cancer ML Model - Prediction Module
Handles model loading and real-time predictions for production deployment.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Tuple, Union
import json
from datetime import datetime

class BreastCancerPredictor:
    """
    Production-ready predictor for breast cancer risk assessment.
    Loads pre-trained model and provides fast predictions with health recommendations.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str = 'models/latest_model.pkl'):
        """Load the pre-trained model and scaler."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"🔄 Loading model from {model_path}...")
            
            # Load the model package
            model_package = joblib.load(model_path)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.metadata = model_package['metadata']
            self.is_loaded = True
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model: {self.metadata['model_name']}")
            print(f"   Performance: AUC={self.metadata['final_auc']:.4f}")
            print(f"   Training Date: {self.metadata['training_date']}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise
    
    def get_feature_template(self) -> Dict[str, float]:
        """Get a template dictionary with all required features."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return {feature: 0.0 for feature in self.feature_names}
    
    def validate_input(self, features: Dict[str, float]) -> pd.DataFrame:
        """Validate and prepare input features for prediction."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {list(missing_features)}")
        
        # Extract features in correct order
        feature_values = [features[name] for name in self.feature_names]
        
        # Create DataFrame
        df = pd.DataFrame([feature_values], columns=self.feature_names)
        
        return df
    
    def predict_single(self, features: Dict[str, float]) -> Dict:
        """
        Make a prediction for a single patient.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results and recommendations
        """
        # Validate input
        df = self.validate_input(features)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Make predictions
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Calculate confidence and risk
        confidence = max(probabilities) * 100
        risk_score = probabilities[0] * 100  # Probability of malignant (class 0)
        
        # Determine result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Benign' if prediction == 1 else 'Malignant',
            'risk_score': round(risk_score, 2),
            'confidence': round(confidence, 2),
            'probabilities': {
                'malignant': round(probabilities[0] * 100, 2),
                'benign': round(probabilities[1] * 100, 2)
            }
        }
        
        # Add recommendations
        result['recommendations'] = self._generate_recommendations(result)
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict]:
        """Make predictions for multiple patients."""
        results = []
        for features in features_list:
            try:
                result = self.predict_single(features)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _generate_recommendations(self, prediction_result: Dict) -> Dict:
        """Generate personalized health recommendations based on prediction."""
        risk_score = prediction_result['risk_score']
        prediction_label = prediction_result['prediction_label']
        
        if prediction_label == 'Malignant' or risk_score > 20:
            urgency = 'HIGH'
            recommendations = {
                'immediate_action': [
                    "⚠️ URGENT: Consult an oncologist immediately",
                    "📋 Schedule comprehensive diagnostic workup",
                    "🏥 Consider seeking a second medical opinion",
                    "🧬 Discuss genetic testing if family history present"
                ],
                'lifestyle': [
                    "🚭 Quit smoking if applicable",
                    "🍎 Adopt anti-inflammatory diet",
                    "💪 Light exercise as tolerated",
                    "😴 Prioritize sleep and stress management"
                ],
                'monitoring': [
                    "📅 Schedule monthly follow-ups",
                    "🔍 Self-examine regularly",
                    "📝 Keep symptom diary"
                ]
            }
        elif risk_score > 10:
            urgency = 'MODERATE'
            recommendations = {
                'immediate_action': [
                    "👩‍⚕️ Schedule appointment with primary care physician",
                    "🔍 Discuss enhanced screening schedule",
                    "📋 Review family history with doctor"
                ],
                'lifestyle': [
                    "🥗 Maintain healthy diet rich in antioxidants",
                    "🏃‍♀️ Regular exercise (150 min/week moderate activity)",
                    "⚖️ Maintain healthy weight",
                    "🍷 Limit alcohol consumption"
                ],
                'monitoring': [
                    "📅 Semi-annual medical checkups",
                    "🔍 Monthly self-examinations",
                    "📊 Track any changes in symptoms"
                ]
            }
        else:
            urgency = 'LOW'
            recommendations = {
                'immediate_action': [
                    "✅ Results suggest low risk - maintain regular screening",
                    "👩‍⚕️ Continue routine annual checkups"
                ],
                'lifestyle': [
                    "🥗 Continue healthy diet with fruits and vegetables",
                    "🏃‍♀️ Maintain regular physical activity",
                    "😴 Keep good sleep hygiene",
                    "🧘‍♀️ Practice stress management techniques"
                ],
                'monitoring': [
                    "📅 Annual medical screenings",
                    "🔍 Monthly self-examinations",
                    "📖 Stay informed about breast health"
                ]
            }
        
        return {
            'urgency_level': urgency,
            **recommendations,
            'disclaimer': "⚠️ This is an AI prediction tool. Always consult healthcare professionals for medical decisions."
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        return {
            'model_name': self.metadata['model_name'],
            'accuracy': self.metadata['final_accuracy'],
            'auc_score': self.metadata['final_auc'],
            'training_date': self.metadata['training_date'],
            'n_features': self.metadata['n_features'],
            'n_training_samples': self.metadata['n_training_samples'],
            'feature_names': self.feature_names[:10]  # First 10 features for brevity
        }

# Utility functions for easy CLI usage
def create_sample_input() -> Dict[str, float]:
    """Create a sample input for testing."""
    # These are approximate values for a low-risk case
    sample_features = {
        'mean radius': 14.0, 'mean texture': 19.0, 'mean perimeter': 91.0,
        'mean area': 654.0, 'mean smoothness': 0.096, 'mean compactness': 0.114,
        'mean concavity': 0.088, 'mean concave points': 0.050, 'mean symmetry': 0.180,
        'mean fractal dimension': 0.062, 'radius error': 0.40, 'texture error': 1.20,
        'perimeter error': 2.85, 'area error': 40.0, 'smoothness error': 0.007,
        'compactness error': 0.025, 'concavity error': 0.032, 'concave points error': 0.012,
        'symmetry error': 0.020, 'fractal dimension error': 0.004, 'worst radius': 16.0,
        'worst texture': 25.0, 'worst perimeter': 108.0, 'worst area': 858.0,
        'worst smoothness': 0.139, 'worst compactness': 0.260, 'worst concavity': 0.200,
        'worst concave points': 0.097, 'worst symmetry': 0.290, 'worst fractal dimension': 0.075
    }
    return sample_features

def predict_from_dict(features: Dict[str, float], model_path: str = 'models/latest_model.pkl'):
    """Quick prediction function."""
    predictor = BreastCancerPredictor(model_path)
    return predictor.predict_single(features)

def main():
    """CLI interface for predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Breast Cancer Risk Prediction')
    parser.add_argument('--model-path', default='models/latest_model.pkl', 
                       help='Path to the trained model')
    parser.add_argument('--sample', action='store_true', 
                       help='Run prediction on sample data')
    parser.add_argument('--info', action='store_true', 
                       help='Show model information')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = BreastCancerPredictor(args.model_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: Make sure to train the model first using breast_cancer_trainer.py")
        return
    
    if args.info:
        info = predictor.get_model_info()
        print("📊 Model Information:")
        print(json.dumps(info, indent=2))
        return
    
    if args.sample:
        print("🧪 Running sample prediction...")
        sample_input = create_sample_input()
        result = predictor.predict_single(sample_input)
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Result: {result['prediction_label']}")
        print(f"   Risk Score: {result['risk_score']}%")
        print(f"   Confidence: {result['confidence']}%")
        print(f"\n💡 Recommendations ({result['recommendations']['urgency_level']} Priority):")
        
        for action in result['recommendations']['immediate_action']:
            print(f"   {action}")
            
        print(f"\n{result['recommendations']['disclaimer']}")
    else:
        print("💡 Use --sample to test with sample data or --info to see model details")

if __name__ == "__main__":
    main()