import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from data_collectors import (
    WeatherDataCollector,
    RainfallDataCollector,
    ElevationDataCollector,
    ClimateDataCollector
)

class GeospatialFeatureExtractor:
    """Extract and process geospatial features for risk modeling"""
    
    def __init__(self):
        self.weather_collector = WeatherDataCollector()
        self.rainfall_collector = RainfallDataCollector()
        self.elevation_collector = ElevationDataCollector()
        self.climate_collector = ClimateDataCollector()
    
    def extract_all_features(self, lat: float, lon: float) -> Dict:
        """Extract all geospatial features for a location"""
        current_weather = self.weather_collector.get_current_weather(lat, lon)
        rainfall_data = self.rainfall_collector.get_precipitation_data(lat, lon, days=7)
        elevation = self.elevation_collector.get_elevation(lat, lon)
        soil_moisture = self.climate_collector.get_soil_moisture(lat, lon)
        land_cover = self.climate_collector.get_land_cover_type(lat, lon)
        
        recent_rainfall = rainfall_data['rainfall_mm'].tail(3).mean()
        total_weekly_rainfall = rainfall_data['rainfall_mm'].sum()
        rainfall_trend = rainfall_data['rainfall_mm'].tail(3).mean() - rainfall_data['rainfall_mm'].head(3).mean()
        
        temp = current_weather['temperature']
        historical_mean_temp = 27.0
        temp_anomaly = self.climate_collector.get_temperature_anomaly(temp, historical_mean_temp)
        
        historical_mean_rainfall = 50.0
        rainfall_anomaly = self.rainfall_collector.get_rainfall_anomaly(recent_rainfall, historical_mean_rainfall)
        
        features = {
            'latitude': lat,
            'longitude': lon,
            'elevation': elevation,
            'current_temperature': temp,
            'temperature_anomaly': temp_anomaly,
            'humidity': current_weather['humidity'],
            'pressure': current_weather['pressure'],
            'wind_speed': current_weather['wind_speed'],
            'recent_rainfall_3day': recent_rainfall,
            'total_weekly_rainfall': total_weekly_rainfall,
            'rainfall_trend': rainfall_trend,
            'rainfall_anomaly': rainfall_anomaly,
            'soil_moisture': soil_moisture,
            'land_cover': land_cover,
            'timestamp': datetime.now()
        }
        
        return features
    
    def calculate_flood_risk_factors(self, features: Dict) -> Dict:
        """Calculate specific flood risk factors"""
        elevation_risk = 1 - min(features['elevation'] / 100, 1.0)
        
        rainfall_risk = min(features['recent_rainfall_3day'] / 100, 1.0)
        
        soil_saturation_risk = features['soil_moisture']
        
        if features['land_cover'] == 'urban':
            runoff_coefficient = 0.8
        elif features['land_cover'] == 'vegetation':
            runoff_coefficient = 0.3
        else:
            runoff_coefficient = 0.5
        
        drainage_risk = runoff_coefficient * (1 - min(features['elevation'] / 50, 1.0))
        
        flood_factors = {
            'elevation_risk': elevation_risk,
            'rainfall_risk': rainfall_risk,
            'soil_saturation_risk': soil_saturation_risk,
            'drainage_risk': drainage_risk,
            'composite_flood_risk': (
                elevation_risk * 0.3 +
                rainfall_risk * 0.4 +
                soil_saturation_risk * 0.15 +
                drainage_risk * 0.15
            )
        }
        
        return flood_factors
    
    def calculate_heat_risk_factors(self, features: Dict) -> Dict:
        """Calculate specific heat risk factors"""
        temp_threshold = 35.0
        temp_risk = max(0, min((features['current_temperature'] - temp_threshold + 10) / 15, 1.0))
        
        heat_index = features['current_temperature'] + (0.5 * (features['humidity'] - 40) / 10)
        heat_index_risk = max(0, min((heat_index - 32) / 15, 1.0))
        
        temp_anomaly_risk = max(0, min(features['temperature_anomaly'] / 5, 1.0))
        
        if features['land_cover'] == 'urban':
            uhi_factor = 0.8
        elif features['land_cover'] == 'vegetation':
            uhi_factor = 0.2
        else:
            uhi_factor = 0.5
        
        heat_factors = {
            'temperature_risk': temp_risk,
            'heat_index_risk': heat_index_risk,
            'temperature_anomaly_risk': temp_anomaly_risk,
            'urban_heat_island_factor': uhi_factor,
            'composite_heat_risk': (
                temp_risk * 0.35 +
                heat_index_risk * 0.35 +
                temp_anomaly_risk * 0.15 +
                uhi_factor * 0.15
            )
        }
        
        return heat_factors
    
    def prepare_ml_features(self, features: Dict, flood_factors: Dict, heat_factors: Dict) -> np.ndarray:
        """Prepare feature vector for ML model"""
        feature_vector = [
            features['elevation'],
            features['current_temperature'],
            features['temperature_anomaly'],
            features['humidity'],
            features['pressure'],
            features['recent_rainfall_3day'],
            features['total_weekly_rainfall'],
            features['rainfall_anomaly'],
            features['soil_moisture'],
            1 if features['land_cover'] == 'urban' else 0,
            1 if features['land_cover'] == 'vegetation' else 0,
            flood_factors['elevation_risk'],
            flood_factors['rainfall_risk'],
            flood_factors['soil_saturation_risk'],
            flood_factors['drainage_risk'],
            heat_factors['temperature_risk'],
            heat_factors['heat_index_risk'],
            heat_factors['temperature_anomaly_risk'],
            heat_factors['urban_heat_island_factor']
        ]
        
        return np.array(feature_vector).reshape(1, -1)
    
    def get_historical_data(self, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """Generate historical data for training or visualization"""
        historical_records = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            temp_base = 27 + 3 * np.sin(2 * np.pi * i / 365)
            temp = temp_base + np.random.normal(0, 2)
            
            rainfall = max(0, np.random.gamma(2, 12))
            
            historical_records.append({
                'date': date,
                'temperature': temp,
                'rainfall': rainfall,
                'humidity': 75 + np.random.normal(0, 8),
                'flood_risk_score': min(rainfall / 80, 1.0) * 0.7 + np.random.uniform(0, 0.3),
                'heat_risk_score': max(0, (temp - 30) / 10) * 0.7 + np.random.uniform(0, 0.3)
            })
        
        return pd.DataFrame(historical_records)
