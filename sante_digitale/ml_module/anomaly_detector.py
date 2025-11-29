import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Machine Learning module for medical anomaly detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'anomaly_detector.pkl')
        self.scaler_path = os.path.join(settings.MEDIA_ROOT, 'models', 'scaler.pkl')
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing anomaly detection model")
            else:
                self._create_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_model()
    
    def _create_model(self):
        """Create new anomaly detection model"""
        try:
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            logger.info("Created new anomaly detection model")
        except Exception as e:
            logger.error(f"Error creating model: {e}")
    
    def prepare_features(self, measurements_data):
        """Prepare features for anomaly detection"""
        features = []
        
        for measurement in measurements_data:
            # Create feature vector based on measurement type and value
            feature_vector = self._extract_features(measurement)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_features(self, measurement):
        """Extract features from a single measurement"""
        features = []
        
        # Basic measurement features
        features.append(measurement.get('value', 0))
        
        # Time-based features
        measured_at = measurement.get('measured_at')
        if measured_at:
            features.append(measured_at.hour)
            features.append(measured_at.dayofweek)
            features.append(measured_at.month)
        else:
            features.extend([0, 0, 0])
        
        # Patient demographic features
        patient = measurement.get('patient_profile', {})
        features.append(patient.get('age', 0))
        features.append(1 if patient.get('gender') == 'M' else 0)
        features.append(patient.get('height', 0) or 0)
        features.append(patient.get('weight', 0) or 0)
        
        # Measurement type encoding
        measurement_types = {
            'blood_pressure_systolic': [1, 0, 0, 0, 0],
            'blood_pressure_diastolic': [0, 1, 0, 0, 0],
            'blood_glucose': [0, 0, 1, 0, 0],
            'heart_rate': [0, 0, 0, 1, 0],
            'temperature': [0, 0, 0, 0, 1],
        }
        measurement_type = measurement.get('measurement_type', '')
        type_encoding = measurement_types.get(measurement_type, [0, 0, 0, 0, 0])
        features.extend(type_encoding)
        
        # Historical features (last 7 days average)
        historical_avg = self._get_historical_average(measurement)
        features.append(historical_avg)
        
        return np.array(features)
    
    def _get_historical_average(self, measurement):
        """Get historical average for the same measurement type"""
        try:
            patient_id = measurement.get('patient_id')
            measurement_type = measurement.get('measurement_type')
            
            cache_key = f"hist_avg_{patient_id}_{measurement_type}"
            cached_avg = cache.get(cache_key)
            
            if cached_avg is not None:
                return cached_avg
            
            # This would typically query the database
            # For now, return the current value as fallback
            return measurement.get('value', 0)
        except Exception as e:
            logger.error(f"Error getting historical average: {e}")
            return measurement.get('value', 0)
    
    def detect_anomalies(self, measurements_data):
        """Detect anomalies in medical measurements"""
        try:
            if not measurements_data:
                return []
            
            # Prepare features
            features = self.prepare_features(measurements_data)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Predict anomalies
            predictions = self.model.fit_predict(features_scaled)
            anomaly_scores = self.model.decision_function(features_scaled)
            
            # Format results
            results = []
            for i, measurement in enumerate(measurements_data):
                is_anomaly = predictions[i] == -1
                score = anomaly_scores[i]
                
                result = {
                    'measurement_id': measurement.get('id'),
                    'is_anomaly': is_anomaly,
                    'anomaly_score': float(score),
                    'anomaly_status': self._classify_anomaly(score),
                    'confidence': self._calculate_confidence(score)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _classify_anomaly(self, score):
        """Classify anomaly based on score"""
        if score < -0.5:
            return 'critical'
        elif score < -0.2:
            return 'warning'
        else:
            return 'normal'
    
    def _calculate_confidence(self, score):
        """Calculate confidence score"""
        # Convert anomaly score to confidence (0-1)
        confidence = 1 - abs(score)
        return max(0, min(1, confidence))
    
    def train_model(self, training_data):
        """Train the anomaly detection model"""
        try:
            if not training_data or len(training_data) < 10:
                logger.warning("Insufficient training data")
                return False
            
            # Prepare features
            features = self.prepare_features(training_data)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled)
            
            # Save model and scaler
            self._save_model()
            
            logger.info(f"Model trained with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _save_model(self):
        """Save model and scaler to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_model_info(self):
        """Get information about the current model"""
        info = {
            'model_type': 'IsolationForest',
            'is_trained': self.model is not None,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
        }
        
        if self.model:
            info.update({
                'n_estimators': self.model.n_estimators,
                'contamination': self.model.contamination,
            })
        
        return info


class HealthRiskPredictor:
    """Predict health risks based on patient data and measurements"""
    
    def __init__(self):
        self.risk_models = {}
        self._load_risk_models()
    
    def _load_risk_models(self):
        """Load risk prediction models"""
        # In production, these would be pre-trained models
        # For now, we'll use rule-based predictions
        self.risk_models = {
            'diabetes_risk': self._diabetes_risk_score,
            'cardiovascular_risk': self._cardiovascular_risk_score,
            'hypertension_risk': self._hypertension_risk_score,
        }
    
    def predict_risks(self, patient_data, measurements):
        """Predict various health risks"""
        risks = {}
        
        for risk_type, model_func in self.risk_models.items():
            try:
                risk_score = model_func(patient_data, measurements)
                risks[risk_type] = {
                    'score': risk_score,
                    'level': self._classify_risk_level(risk_score),
                    'recommendations': self._get_recommendations(risk_type, risk_score)
                }
            except Exception as e:
                logger.error(f"Error predicting {risk_type}: {e}")
                risks[risk_type] = {'score': 0, 'level': 'low', 'recommendations': []}
        
        return risks
    
    def _diabetes_risk_score(self, patient_data, measurements):
        """Calculate diabetes risk score"""
        score = 0
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 45:
            score += 0.2
        
        # BMI factor
        height = patient_data.get('height', 0) / 100  # Convert cm to m
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / (height * height)
            if bmi > 30:
                score += 0.3
            elif bmi > 25:
                score += 0.15
        
        # Blood glucose measurements
        glucose_measurements = [m for m in measurements if m.get('measurement_type') == 'blood_glucose']
        if glucose_measurements:
            avg_glucose = np.mean([m.get('value', 0) for m in glucose_measurements])
            if avg_glucose > 126:
                score += 0.4
            elif avg_glucose > 100:
                score += 0.2
        
        return min(1.0, score)
    
    def _cardiovascular_risk_score(self, patient_data, measurements):
        """Calculate cardiovascular risk score"""
        score = 0
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 65:
            score += 0.3
        elif age > 55:
            score += 0.2
        
        # Blood pressure factor
        bp_measurements = [m for m in measurements if 'blood_pressure' in m.get('measurement_type', '')]
        if bp_measurements:
            systolic_values = [m.get('value', 0) for m in bp_measurements if 'systolic' in m.get('measurement_type', '')]
            if systolic_values:
                avg_systolic = np.mean(systolic_values)
                if avg_systolic > 140:
                    score += 0.3
                elif avg_systolic > 130:
                    score += 0.15
        
        # Cholesterol factor
        cholesterol_measurements = [m for m in measurements if m.get('measurement_type') == 'cholesterol']
        if cholesterol_measurements:
            avg_cholesterol = np.mean([m.get('value', 0) for m in cholesterol_measurements])
            if avg_cholesterol > 240:
                score += 0.2
            elif avg_cholesterol > 200:
                score += 0.1
        
        return min(1.0, score)
    
    def _hypertension_risk_score(self, patient_data, measurements):
        """Calculate hypertension risk score"""
        score = 0
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 60:
            score += 0.2
        elif age > 45:
            score += 0.1
        
        # BMI factor
        height = patient_data.get('height', 0) / 100
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / (height * height)
            if bmi > 30:
                score += 0.2
            elif bmi > 25:
                score += 0.1
        
        # Blood pressure measurements
        bp_measurements = [m for m in measurements if 'blood_pressure' in m.get('measurement_type', '')]
        if bp_measurements:
            systolic_values = [m.get('value', 0) for m in bp_measurements if 'systolic' in m.get('measurement_type', '')]
            diastolic_values = [m.get('value', 0) for m in bp_measurements if 'diastolic' in m.get('measurement_type', '')]
            
            if systolic_values:
                avg_systolic = np.mean(systolic_values)
                if avg_systolic > 160:
                    score += 0.4
                elif avg_systolic > 140:
                    score += 0.2
            
            if diastolic_values:
                avg_diastolic = np.mean(diastolic_values)
                if avg_diastolic > 100:
                    score += 0.3
                elif avg_diastolic > 90:
                    score += 0.15
        
        return min(1.0, score)
    
    def _classify_risk_level(self, score):
        """Classify risk level based on score"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendations(self, risk_type, score):
        """Get recommendations based on risk type and score"""
        recommendations = {
            'diabetes_risk': [
                'Monitor blood glucose regularly',
                'Maintain healthy weight',
                'Exercise regularly',
                'Follow a balanced diet'
            ],
            'cardiovascular_risk': [
                'Monitor blood pressure',
                'Reduce sodium intake',
                'Exercise regularly',
                'Quit smoking if applicable'
            ],
            'hypertension_risk': [
                'Monitor blood pressure daily',
                'Reduce stress',
                'Limit alcohol consumption',
                'Follow DASH diet'
            ]
        }
        
        if score >= 0.7:
            recommendations[risk_type].append('Consult with healthcare provider immediately')
        elif score >= 0.4:
            recommendations[risk_type].append('Schedule regular check-ups')
        
        return recommendations.get(risk_type, [])
