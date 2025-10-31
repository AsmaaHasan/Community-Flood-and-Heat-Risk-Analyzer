import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import pickle
import os
from datetime import datetime

class RiskPredictionModel:
    """Random Forest-based risk prediction model for flood and heat risks"""
    
    def __init__(self):
        self.flood_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.heat_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.feature_names = [
            'elevation',
            'current_temperature',
            'temperature_anomaly',
            'humidity',
            'pressure',
            'recent_rainfall_3day',
            'total_weekly_rainfall',
            'rainfall_anomaly',
            'soil_moisture',
            'is_urban',
            'is_vegetation',
            'elevation_risk',
            'rainfall_risk',
            'soil_saturation_risk',
            'drainage_risk',
            'temperature_risk',
            'heat_index_risk',
            'temperature_anomaly_risk',
            'urban_heat_island_factor'
        ]
    
    def generate_training_data(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        
        X = []
        y_flood = []
        y_heat = []
        
        for _ in range(n_samples):
            elevation = np.random.uniform(0, 100)
            temp = np.random.uniform(22, 38)
            temp_anomaly = np.random.uniform(-5, 8)
            humidity = np.random.uniform(50, 95)
            pressure = np.random.uniform(1000, 1020)
            recent_rainfall = np.random.gamma(2, 15)
            total_rainfall = recent_rainfall * np.random.uniform(2, 5)
            rainfall_anomaly = np.random.uniform(-1, 2)
            soil_moisture = np.random.uniform(0.2, 0.7)
            is_urban = np.random.choice([0, 1], p=[0.4, 0.6])
            is_vegetation = 1 - is_urban if np.random.random() > 0.3 else 0
            
            elevation_risk = 1 - min(elevation / 100, 1.0)
            rainfall_risk = min(recent_rainfall / 100, 1.0)
            soil_saturation_risk = soil_moisture
            drainage_risk = (0.8 if is_urban else 0.4) * elevation_risk
            
            temperature_risk = max(0, min((temp - 25) / 15, 1.0))
            heat_index = temp + (0.5 * (humidity - 40) / 10)
            heat_index_risk = max(0, min((heat_index - 32) / 15, 1.0))
            temp_anomaly_risk = max(0, min(temp_anomaly / 5, 1.0))
            uhi_factor = 0.8 if is_urban else 0.3
            
            features = [
                elevation, temp, temp_anomaly, humidity, pressure,
                recent_rainfall, total_rainfall, rainfall_anomaly, soil_moisture,
                is_urban, is_vegetation,
                elevation_risk, rainfall_risk, soil_saturation_risk, drainage_risk,
                temperature_risk, heat_index_risk, temp_anomaly_risk, uhi_factor
            ]
            
            flood_risk_score = (
                elevation_risk * 0.25 +
                rainfall_risk * 0.35 +
                soil_saturation_risk * 0.15 +
                drainage_risk * 0.15 +
                rainfall_anomaly * 0.1
            )
            
            heat_risk_score = (
                temperature_risk * 0.3 +
                heat_index_risk * 0.3 +
                temp_anomaly_risk * 0.2 +
                uhi_factor * 0.2
            )
            
            if flood_risk_score < 0.33:
                flood_label = 0
            elif flood_risk_score < 0.66:
                flood_label = 1
            else:
                flood_label = 2
            
            if heat_risk_score < 0.33:
                heat_label = 0
            elif heat_risk_score < 0.66:
                heat_label = 1
            else:
                heat_label = 2
            
            X.append(features)
            y_flood.append(flood_label)
            y_heat.append(heat_label)
        
        return np.array(X), np.array(y_flood), np.array(y_heat)
    
    def train(self):
        """Train both flood and heat risk models"""
        X, y_flood, y_heat = self.generate_training_data()
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.flood_model.fit(X_scaled, y_flood)
        self.heat_model.fit(X_scaled, y_heat)
        
        self.is_trained = True
        
        flood_score = self.flood_model.score(X_scaled, y_flood)
        heat_score = self.heat_model.score(X_scaled, y_heat)
        
        return {
            'flood_accuracy': flood_score,
            'heat_accuracy': heat_score,
            'training_samples': len(X)
        }
    
    def predict_flood_risk(self, features: np.ndarray, nlp_signal: float = 0.0) -> Dict:
        """Predict flood risk with NLP fusion"""
        if not self.is_trained:
            self.train()
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.flood_model.predict(features_scaled)[0]
        probabilities = self.flood_model.predict_proba(features_scaled)[0]
        
        base_risk_score = probabilities[2] * 1.0 + probabilities[1] * 0.5 + probabilities[0] * 0.1
        
        nlp_weight = 0.2
        fused_risk_score = (base_risk_score * (1 - nlp_weight)) + (nlp_signal * nlp_weight)
        
        if fused_risk_score < 0.33:
            final_level = 'Low'
            final_class = 0
        elif fused_risk_score < 0.66:
            final_level = 'Medium'
            final_class = 1
        else:
            final_level = 'High'
            final_class = 2
        
        feature_importance = self.flood_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_features = [(self.feature_names[i], feature_importance[i]) for i in top_features_idx]
        
        return {
            'risk_level': final_level,
            'risk_class': final_class,
            'risk_score': fused_risk_score,
            'base_ml_score': base_risk_score,
            'nlp_contribution': nlp_signal * nlp_weight,
            'probabilities': {
                'low': probabilities[0],
                'medium': probabilities[1],
                'high': probabilities[2]
            },
            'top_features': top_features
        }
    
    def predict_heat_risk(self, features: np.ndarray, nlp_signal: float = 0.0) -> Dict:
        """Predict heat risk with NLP fusion"""
        if not self.is_trained:
            self.train()
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.heat_model.predict(features_scaled)[0]
        probabilities = self.heat_model.predict_proba(features_scaled)[0]
        
        base_risk_score = probabilities[2] * 1.0 + probabilities[1] * 0.5 + probabilities[0] * 0.1
        
        nlp_weight = 0.2
        fused_risk_score = (base_risk_score * (1 - nlp_weight)) + (nlp_signal * nlp_weight)
        
        if fused_risk_score < 0.33:
            final_level = 'Low'
            final_class = 0
        elif fused_risk_score < 0.66:
            final_level = 'Medium'
            final_class = 1
        else:
            final_level = 'High'
            final_class = 2
        
        feature_importance = self.heat_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_features = [(self.feature_names[i], feature_importance[i]) for i in top_features_idx]
        
        return {
            'risk_level': final_level,
            'risk_class': final_class,
            'risk_score': fused_risk_score,
            'base_ml_score': base_risk_score,
            'nlp_contribution': nlp_signal * nlp_weight,
            'probabilities': {
                'low': probabilities[0],
                'medium': probabilities[1],
                'high': probabilities[2]
            },
            'top_features': top_features
        }
    
    def save_models(self, filepath: str = 'models'):
        """Save trained models to disk"""
        os.makedirs(filepath, exist_ok=True)
        
        with open(f'{filepath}/flood_model.pkl', 'wb') as f:
            pickle.dump(self.flood_model, f)
        
        with open(f'{filepath}/heat_model.pkl', 'wb') as f:
            pickle.dump(self.heat_model, f)
        
        with open(f'{filepath}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_models(self, filepath: str = 'models'):
        """Load trained models from disk"""
        try:
            with open(f'{filepath}/flood_model.pkl', 'rb') as f:
                self.flood_model = pickle.load(f)
            
            with open(f'{filepath}/heat_model.pkl', 'rb') as f:
                self.heat_model = pickle.load(f)
            
            with open(f'{filepath}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
