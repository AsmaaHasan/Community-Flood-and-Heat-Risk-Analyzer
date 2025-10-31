import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from typing import Dict, Tuple, List
import pickle
import os
from datetime import datetime

class EnhancedRiskPredictionModel:
    """Enhanced risk prediction with both Random Forest and XGBoost models"""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize model
        Args:
            model_type: 'random_forest', 'xgboost', or 'ensemble' (default)
        """
        self.model_type = model_type
        
        self.rf_flood_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.rf_heat_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.xgb_flood_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        self.xgb_heat_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.performance_metrics = {}
        
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
    
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    def train(self, test_size: float = 0.2):
        """Train all models and compare performance"""
        X, y_flood, y_heat = self.generate_training_data()
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_flood_train, y_flood_test = y_flood[:split_idx], y_flood[split_idx:]
        y_heat_train, y_heat_test = y_heat[:split_idx], y_heat[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest models...")
        self.rf_flood_model.fit(X_train_scaled, y_flood_train)
        self.rf_heat_model.fit(X_train_scaled, y_heat_train)
        
        rf_flood_pred = self.rf_flood_model.predict(X_test_scaled)
        rf_heat_pred = self.rf_heat_model.predict(X_test_scaled)
        
        print("Training XGBoost models...")
        self.xgb_flood_model.fit(X_train_scaled, y_flood_train)
        self.xgb_heat_model.fit(X_train_scaled, y_heat_train)
        
        xgb_flood_pred = self.xgb_flood_model.predict(X_test_scaled)
        xgb_heat_pred = self.xgb_heat_model.predict(X_test_scaled)
        
        self.performance_metrics = {
            'random_forest': {
                'flood_accuracy': accuracy_score(y_flood_test, rf_flood_pred),
                'heat_accuracy': accuracy_score(y_heat_test, rf_heat_pred),
                'flood_report': classification_report(y_flood_test, rf_flood_pred, output_dict=True),
                'heat_report': classification_report(y_heat_test, rf_heat_pred, output_dict=True)
            },
            'xgboost': {
                'flood_accuracy': accuracy_score(y_flood_test, xgb_flood_pred),
                'heat_accuracy': accuracy_score(y_heat_test, xgb_heat_pred),
                'flood_report': classification_report(y_flood_test, xgb_flood_pred, output_dict=True),
                'heat_report': classification_report(y_heat_test, xgb_heat_pred, output_dict=True)
            },
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.is_trained = True
        
        return self.performance_metrics
    
    def predict_flood_risk(self, features: np.ndarray, nlp_signal: float = 0.0) -> Dict:
        """Predict flood risk using selected model type"""
        if not self.is_trained:
            self.train()
        
        features_scaled = self.scaler.transform(features)
        
        if self.model_type == 'random_forest':
            prediction = self.rf_flood_model.predict(features_scaled)[0]
            probabilities = self.rf_flood_model.predict_proba(features_scaled)[0]
            feature_importance = self.rf_flood_model.feature_importances_
            model_name = 'Random Forest'
        elif self.model_type == 'xgboost':
            prediction = self.xgb_flood_model.predict(features_scaled)[0]
            probabilities = self.xgb_flood_model.predict_proba(features_scaled)[0]
            feature_importance = self.xgb_flood_model.feature_importances_
            model_name = 'XGBoost'
        else:
            rf_prob = self.rf_flood_model.predict_proba(features_scaled)[0]
            xgb_prob = self.xgb_flood_model.predict_proba(features_scaled)[0]
            probabilities = (rf_prob + xgb_prob) / 2
            prediction = np.argmax(probabilities)
            feature_importance = (self.rf_flood_model.feature_importances_ + 
                                self.xgb_flood_model.feature_importances_) / 2
            model_name = 'Ensemble (RF + XGB)'
        
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
        
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_features = [(self.feature_names[i], float(feature_importance[i])) for i in top_features_idx]
        
        return {
            'risk_level': final_level,
            'risk_class': final_class,
            'risk_score': fused_risk_score,
            'base_ml_score': base_risk_score,
            'nlp_contribution': nlp_signal * nlp_weight,
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'top_features': top_features,
            'model_used': model_name
        }
    
    def predict_heat_risk(self, features: np.ndarray, nlp_signal: float = 0.0) -> Dict:
        """Predict heat risk using selected model type"""
        if not self.is_trained:
            self.train()
        
        features_scaled = self.scaler.transform(features)
        
        if self.model_type == 'random_forest':
            prediction = self.rf_heat_model.predict(features_scaled)[0]
            probabilities = self.rf_heat_model.predict_proba(features_scaled)[0]
            feature_importance = self.rf_heat_model.feature_importances_
            model_name = 'Random Forest'
        elif self.model_type == 'xgboost':
            prediction = self.xgb_heat_model.predict(features_scaled)[0]
            probabilities = self.xgb_heat_model.predict_proba(features_scaled)[0]
            feature_importance = self.xgb_heat_model.feature_importances_
            model_name = 'XGBoost'
        else:
            rf_prob = self.rf_heat_model.predict_proba(features_scaled)[0]
            xgb_prob = self.xgb_heat_model.predict_proba(features_scaled)[0]
            probabilities = (rf_prob + xgb_prob) / 2
            prediction = np.argmax(probabilities)
            feature_importance = (self.rf_heat_model.feature_importances_ + 
                                self.xgb_heat_model.feature_importances_) / 2
            model_name = 'Ensemble (RF + XGB)'
        
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
        
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_features = [(self.feature_names[i], float(feature_importance[i])) for i in top_features_idx]
        
        return {
            'risk_level': final_level,
            'risk_class': final_class,
            'risk_score': fused_risk_score,
            'base_ml_score': base_risk_score,
            'nlp_contribution': nlp_signal * nlp_weight,
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'top_features': top_features,
            'model_used': model_name
        }
    
    def get_performance_comparison(self) -> pd.DataFrame:
        """Get comparison dataframe of model performances"""
        if not self.performance_metrics:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model in ['random_forest', 'xgboost']:
            comparison_data.append({
                'Model': 'Random Forest' if model == 'random_forest' else 'XGBoost',
                'Flood Accuracy': self.performance_metrics[model]['flood_accuracy'],
                'Heat Accuracy': self.performance_metrics[model]['heat_accuracy'],
                'Avg Accuracy': (self.performance_metrics[model]['flood_accuracy'] + 
                               self.performance_metrics[model]['heat_accuracy']) / 2
            })
        
        return pd.DataFrame(comparison_data)
