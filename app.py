# app.py
"""
Flask Web Application for Breast Cancer Risk Prediction
Production-ready web interface with health recommendations
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from breast_cancer_predictor import BreastCancerPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor on app startup."""
    global predictor
    try:
        model_path = 'models/latest_model.pkl'
        if not os.path.exists(model_path):
            print("❌ No trained model found. Please run breast_cancer_trainer.py first.")
            return False
        
        predictor = BreastCancerPredictor(model_path)
        print("✅ Predictor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return False

# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Risk Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: flex;
            min-height: 600px;
        }
        
        .input-section {
            flex: 1;
            padding: 30px;
            border-right: 1px solid #eee;
        }
        
        .results-section {
            flex: 1;
            padding: 30px;
            background: #f8f9fa;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .form-col {
            flex: 1;
            min-width: 200px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 10px 10px 10px 0;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        }
        
        .results-container {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .prediction-result {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .result-benign {
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
        }
        
        .result-malignant {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }
        
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .recommendations {
            margin-top: 20px;
        }
        
        .urgency-high { border-left: 4px solid #e74c3c; }
        .urgency-moderate { border-left: 4px solid #f39c12; }
        .urgency-low { border-left: 4px solid #2ecc71; }
        
        .recommendation-category {
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .recommendation-category h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .recommendation-list {
            list-style: none;
            padding-left: 0;
        }
        
        .recommendation-list li {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        
        .recommendation-list li:last-child {
            border-bottom: none;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c66;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .model-info {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .input-section {
                border-right: none;
                border-bottom: 1px solid #eee;
            }
            
            .form-row {
                flex-direction: column;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Breast Cancer Risk Predictor</h1>
            <p>AI-powered risk assessment with personalized health recommendations</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <div class="model-info" id="modelInfo">
                    <strong>📊 Model Information:</strong><br>
                    <span id="modelDetails">Loading model details...</span>
                </div>
                
                <h2>📝 Patient Information</h2>
                <p style="margin-bottom: 20px; color: #666;">Enter the medical measurements below. All fields are required for accurate prediction.</p>
                
                <form id="predictionForm">
                    <div class="form-group">
                        <h3>🔍 Mean Measurements</h3>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="mean_radius">Radius</label>
                                <input type="number" step="0.01" id="mean_radius" name="mean radius" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_texture">Texture</label>
                                <input type="number" step="0.01" id="mean_texture" name="mean texture" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_perimeter">Perimeter</label>
                                <input type="number" step="0.01" id="mean_perimeter" name="mean perimeter" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="mean_area">Area</label>
                                <input type="number" step="0.01" id="mean_area" name="mean area" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_smoothness">Smoothness</label>
                                <input type="number" step="0.001" id="mean_smoothness" name="mean smoothness" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_compactness">Compactness</label>
                                <input type="number" step="0.001" id="mean_compactness" name="mean compactness" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="mean_concavity">Concavity</label>
                                <input type="number" step="0.001" id="mean_concavity" name="mean concavity" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_concave_points">Concave Points</label>
                                <input type="number" step="0.001" id="mean_concave_points" name="mean concave points" required>
                            </div>
                            <div class="form-col">
                                <label for="mean_symmetry">Symmetry</label>
                                <input type="number" step="0.001" id="mean_symmetry" name="mean symmetry" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="mean_fractal_dimension">Fractal Dimension</label>
                                <input type="number" step="0.001" id="mean_fractal_dimension" name="mean fractal dimension" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <h3>📏 Standard Error Measurements</h3>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="radius_error">Radius Error</label>
                                <input type="number" step="0.001" id="radius_error" name="radius error" required>
                            </div>
                            <div class="form-col">
                                <label for="texture_error">Texture Error</label>
                                <input type="number" step="0.001" id="texture_error" name="texture error" required>
                            </div>
                            <div class="form-col">
                                <label for="perimeter_error">Perimeter Error</label>
                                <input type="number" step="0.001" id="perimeter_error" name="perimeter error" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="area_error">Area Error</label>
                                <input type="number" step="0.01" id="area_error" name="area error" required>
                            </div>
                            <div class="form-col">
                                <label for="smoothness_error">Smoothness Error</label>
                                <input type="number" step="0.0001" id="smoothness_error" name="smoothness error" required>
                            </div>
                            <div class="form-col">
                                <label for="compactness_error">Compactness Error</label>
                                <input type="number" step="0.001" id="compactness_error" name="compactness error" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="concavity_error">Concavity Error</label>
                                <input type="number" step="0.001" id="concavity_error" name="concavity error" required>
                            </div>
                            <div class="form-col">
                                <label for="concave_points_error">Concave Points Error</label>
                                <input type="number" step="0.0001" id="concave_points_error" name="concave points error" required>
                            </div>
                            <div class="form-col">
                                <label for="symmetry_error">Symmetry Error</label>
                                <input type="number" step="0.001" id="symmetry_error" name="symmetry error" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="fractal_dimension_error">Fractal Dimension Error</label>
                                <input type="number" step="0.0001" id="fractal_dimension_error" name="fractal dimension error" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <h3>📊 Worst Measurements</h3>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="worst_radius">Worst Radius</label>
                                <input type="number" step="0.01" id="worst_radius" name="worst radius" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_texture">Worst Texture</label>
                                <input type="number" step="0.01" id="worst_texture" name="worst texture" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_perimeter">Worst Perimeter</label>
                                <input type="number" step="0.01" id="worst_perimeter" name="worst perimeter" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="worst_area">Worst Area</label>
                                <input type="number" step="0.01" id="worst_area" name="worst area" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_smoothness">Worst Smoothness</label>
                                <input type="number" step="0.001" id="worst_smoothness" name="worst smoothness" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_compactness">Worst Compactness</label>
                                <input type="number" step="0.001" id="worst_compactness" name="worst compactness" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="worst_concavity">Worst Concavity</label>
                                <input type="number" step="0.001" id="worst_concavity" name="worst concavity" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_concave_points">Worst Concave Points</label>
                                <input type="number" step="0.001" id="worst_concave_points" name="worst concave points" required>
                            </div>
                            <div class="form-col">
                                <label for="worst_symmetry">Worst Symmetry</label>
                                <input type="number" step="0.001" id="worst_symmetry" name="worst symmetry" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-col">
                                <label for="worst_fractal_dimension">Worst Fractal Dimension</label>
                                <input type="number" step="0.001" id="worst_fractal_dimension" name="worst fractal dimension" required>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <button type="submit" class="btn">🔮 Predict Risk</button>
                        <button type="button" class="btn btn-secondary" onclick="loadSampleData()">📋 Load Sample Data</button>
                        <button type="reset" class="btn btn-secondary">🗑️ Clear Form</button>
                    </div>
                </form>
            </div>
            
            <div class="results-section">
                <h2>📊 Prediction Results</h2>
                <div id="resultsContainer">
                    <div class="loading">
                        <p>📝 Complete the form and click "Predict Risk" to see results</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Load model information on page load
        window.onload = function() {
            fetch('/api/model-info')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('modelDetails').innerHTML = '❌ ' + data.error;
                    } else {
                        document.getElementById('modelDetails').innerHTML = 
                            `Model: ${data.model_name} | ` +
                            `Accuracy: ${(data.accuracy * 100).toFixed(1)}% | ` +
                            `AUC: ${data.auc_score.toFixed(3)} | ` +
                            `Trained: ${new Date(data.training_date).toLocaleDateString()}`;
                    }
                })
                .catch(error => {
                    document.getElementById('modelDetails').innerHTML = '❌ Error loading model info';
                });
        };

        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('resultsContainer').innerHTML = 
                '<div class="loading"><p>🔄 Analyzing data...</p></div>';
            
            // Collect form data
            const formData = new FormData(this);
            const features = {};
            
            for (let [key, value] of formData.entries()) {
                features[key] = parseFloat(value);
            }
            
            // Make prediction request
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features: features })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    displayError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                displayError('Network error: ' + error.message);
            });
        });

        function displayResults(result) {
            const resultClass = result.prediction_label === 'Benign' ? 'result-benign' : 'result-malignant';
            const urgencyClass = `urgency-${result.recommendations.urgency_level.toLowerCase()}`;
            
            let html = `
                <div class="results-container">
                    <div class="prediction-result ${resultClass}">
                        <h3>${result.prediction_label}</h3>
                        <div class="risk-score">${result.risk_score}%</div>
                        <p>Risk Score | Confidence: ${result.confidence}%</p>
                    </div>
                </div>
                
                <div class="results-container recommendations ${urgencyClass}">
                    <h3>💡 Personalized Recommendations</h3>
                    <p><strong>Priority Level: ${result.recommendations.urgency_level}</strong></p>
            `;
            
            // Add immediate actions
            if (result.recommendations.immediate_action) {
                html += `
                    <div class="recommendation-category">
                        <h4>🚨 Immediate Actions</h4>
                        <ul class="recommendation-list">
                `;
                result.recommendations.immediate_action.forEach(action => {
                    html += `<li>${action}</li>`;
                });
                html += '</ul></div>';
            }
            
            // Add lifestyle recommendations
            if (result.recommendations.lifestyle) {
                html += `
                    <div class="recommendation-category">
                        <h4>🌱 Lifestyle Recommendations</h4>
                        <ul class="recommendation-list">
                `;
                result.recommendations.lifestyle.forEach(lifestyle => {
                    html += `<li>${lifestyle}</li>`;
                });
                html += '</ul></div>';
            }
            
            // Add monitoring recommendations
            if (result.recommendations.monitoring) {
                html += `
                    <div class="recommendation-category">
                        <h4>📅 Monitoring & Follow-up</h4>
                        <ul class="recommendation-list">
                `;
                result.recommendations.monitoring.forEach(monitoring => {
                    html += `<li>${monitoring}</li>`;
                });
                html += '</ul></div>';
            }
            
            // Add disclaimer
            html += `
                <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                    <small><strong>${result.recommendations.disclaimer}</strong></small>
                </div>
            </div>`;
            
            document.getElementById('resultsContainer').innerHTML = html;
        }

        function displayError(error) {
            document.getElementById('resultsContainer').innerHTML = `
                <div class="error">
                    <strong>❌ Error:</strong> ${error}
                </div>
            `;
        }

        // Load sample data for testing
        function loadSampleData() {
            fetch('/api/sample-data')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('Error loading sample data: ' + data.error);
                        return;
                    }
                    
                    // Fill form with sample data
                    for (let [key, value] of Object.entries(data.features)) {
                        const input = document.querySelector(`input[name="${key}"]`);
                        if (input) {
                            input.value = value;
                        }
                    }
                })
                .catch(error => {
                    alert('Error loading sample data: ' + error.message);
                });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface."""
    if predictor is None:
        return "❌ Model not loaded. Please run breast_cancer_trainer.py first.", 500
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = data['features']
        result = predictor.predict_single(features)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data for testing."""
    try:
        from breast_cancer_predictor import create_sample_input
        sample_features = create_sample_input()
        return jsonify({"features": sample_features})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy" if predictor is not None else "unhealthy"
    return jsonify({"status": status, "model_loaded": predictor is not None})

if __name__ == '__main__':
    print("🚀 Starting Breast Cancer Risk Predictor Web App")
    print("=" * 50)
    
    # Initialize predictor
    if initialize_predictor():
        print("🌐 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
        print("🔗 API endpoints:")
        print("   GET  /health - Health check")
        print("   GET  /api/model-info - Model information")
        print("   POST /api/predict - Make prediction")
        print("   GET  /api/sample-data - Get sample data")
        print("-" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("💡 Please run the following command first:")
        print("   python breast_cancer_trainer.py")
        print("\nThis will train the model and save it for the web app to use.")