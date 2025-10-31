import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import json

class WeatherDataCollector:
    """Collects meteorological data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY', '')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Fetch current weather data for a location"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'timestamp': datetime.now()
                }
            else:
                return self._get_mock_weather(lat, lon)
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_mock_weather(lat, lon)
    
    def get_forecast(self, lat: float, lon: float, days: int = 5) -> pd.DataFrame:
        """Fetch weather forecast"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                forecasts = []
                for item in data['list'][:days*8]:
                    forecasts.append({
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'rain': item.get('rain', {}).get('3h', 0),
                        'description': item['weather'][0]['description']
                    })
                return pd.DataFrame(forecasts)
            else:
                return self._get_mock_forecast(lat, lon, days)
        except Exception as e:
            print(f"Forecast API error: {e}")
            return self._get_mock_forecast(lat, lon, days)
    
    def _get_mock_weather(self, lat: float, lon: float) -> Dict:
        """Generate realistic mock weather data for Manila"""
        base_temp = 28 + np.random.normal(0, 3)
        return {
            'temperature': base_temp,
            'humidity': 75 + np.random.normal(0, 10),
            'pressure': 1010 + np.random.normal(0, 5),
            'description': np.random.choice(['partly cloudy', 'humid', 'hot']),
            'wind_speed': 3 + np.random.normal(0, 2),
            'timestamp': datetime.now()
        }
    
    def _get_mock_forecast(self, lat: float, lon: float, days: int) -> pd.DataFrame:
        """Generate mock forecast data"""
        forecasts = []
        base_time = datetime.now()
        for i in range(days * 8):
            forecasts.append({
                'datetime': base_time + timedelta(hours=i*3),
                'temperature': 28 + np.random.normal(0, 4),
                'humidity': 75 + np.random.normal(0, 12),
                'rain': max(0, np.random.exponential(2)),
                'description': np.random.choice(['partly cloudy', 'humid', 'scattered clouds'])
            })
        return pd.DataFrame(forecasts)


class RainfallDataCollector:
    """Collects rainfall data from NASA GPM (Global Precipitation Measurement)"""
    
    def __init__(self):
        self.gpm_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3"
        
    def get_precipitation_data(self, lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """
        Fetch precipitation data. Uses mock data for MVP.
        In production, would integrate with NASA EarthData API.
        """
        rainfall_data = []
        base_time = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_time + timedelta(days=i)
            if lat > 10 and lat < 20 and lon > 120 and lon < 125:
                rainfall = max(0, np.random.gamma(2, 15))
            else:
                rainfall = max(0, np.random.gamma(2, 10))
            
            rainfall_data.append({
                'date': date,
                'rainfall_mm': rainfall,
                'location': f"{lat:.2f},{lon:.2f}"
            })
        
        return pd.DataFrame(rainfall_data)
    
    def get_rainfall_anomaly(self, current_rainfall: float, historical_mean: float = 50) -> float:
        """Calculate rainfall anomaly (deviation from historical average)"""
        return (current_rainfall - historical_mean) / historical_mean if historical_mean > 0 else 0


class ElevationDataCollector:
    """Collects elevation data for flood risk assessment"""
    
    def __init__(self):
        self.srtm_api = "https://api.open-elevation.com/api/v1/lookup"
        
    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation data from Open Elevation API"""
        try:
            params = {
                'locations': f"{lat},{lon}"
            }
            response = requests.get(self.srtm_api, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data['results'][0]['elevation']
            else:
                return self._estimate_elevation(lat, lon)
        except Exception as e:
            print(f"Elevation API error: {e}")
            return self._estimate_elevation(lat, lon)
    
    def _estimate_elevation(self, lat: float, lon: float) -> float:
        """Estimate elevation based on location (Manila area)"""
        if lat > 14.4 and lat < 14.7 and lon > 120.9 and lon < 121.1:
            return np.random.uniform(5, 25)
        return np.random.uniform(10, 100)


class NewsDataCollector:
    """Collects news articles from NewsAPI and GDELT"""
    
    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY', '')
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
    def get_news_articles(self, query: str, location: str = "Manila", days: int = 7) -> List[Dict]:
        """Fetch news articles from NewsAPI"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            params = {
                'q': f"{query} {location}",
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', [])[:10]:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    })
                return articles
            else:
                return self._get_mock_news(query, location)
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return self._get_mock_news(query, location)
    
    def get_gdelt_data(self, query: str, location: str = "Manila") -> List[Dict]:
        """Fetch articles from GDELT"""
        try:
            params = {
                'query': f"{query} {location}",
                'mode': 'artlist',
                'maxrecords': 10,
                'format': 'json'
            }
            
            response = requests.get(self.gdelt_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', [])[:10]:
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'seendate': article.get('seendate', ''),
                        'socialimage': article.get('socialimage', ''),
                        'domain': article.get('domain', '')
                    })
                return articles
            else:
                return []
        except Exception as e:
            print(f"GDELT API error: {e}")
            return []
    
    def _get_mock_news(self, query: str, location: str) -> List[Dict]:
        """Generate mock news articles for demonstration"""
        mock_articles = []
        
        if 'flood' in query.lower():
            templates = [
                {
                    'title': f'Heavy rainfall triggers flood warnings in {location}',
                    'description': 'Meteorological department issues flood alert as heavy monsoon rains continue to affect low-lying areas.',
                    'content': 'Residents in flood-prone areas have been advised to stay vigilant as water levels continue to rise.',
                    'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                    'source': 'Manila Bulletin'
                },
                {
                    'title': f'{location} prepares for potential flooding',
                    'description': 'Local authorities activate emergency response teams amid weather warnings.',
                    'content': 'Emergency shelters have been set up in preparation for severe weather conditions.',
                    'publishedAt': (datetime.now() - timedelta(days=2)).isoformat(),
                    'source': 'Philippine Daily Inquirer'
                },
                {
                    'title': f'Flood control measures implemented in {location}',
                    'description': 'City government strengthens drainage systems ahead of rainy season.',
                    'content': 'Infrastructure improvements aim to reduce flood risks in vulnerable communities.',
                    'publishedAt': (datetime.now() - timedelta(days=3)).isoformat(),
                    'source': 'ABS-CBN News'
                }
            ]
        elif 'heat' in query.lower():
            templates = [
                {
                    'title': f'Heat index reaches dangerous levels in {location}',
                    'description': 'Health officials warn of heat-related illnesses as temperatures soar.',
                    'content': 'Public advised to stay hydrated and avoid outdoor activities during peak hours.',
                    'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                    'source': 'Manila Times'
                },
                {
                    'title': f'Extreme heat wave affects {location} region',
                    'description': 'Record-breaking temperatures prompt health warnings.',
                    'content': 'Cooling centers opened to provide relief for vulnerable populations.',
                    'publishedAt': (datetime.now() - timedelta(days=2)).isoformat(),
                    'source': 'Rappler'
                },
                {
                    'title': f'Schools adjust schedules due to extreme heat in {location}',
                    'description': 'Educational institutions implement heat safety protocols.',
                    'content': 'Classes shortened and outdoor activities suspended during high heat index.',
                    'publishedAt': (datetime.now() - timedelta(days=4)).isoformat(),
                    'source': 'GMA News'
                }
            ]
        else:
            templates = [
                {
                    'title': f'Weather update for {location}',
                    'description': 'Latest meteorological conditions and forecasts.',
                    'content': 'Residents advised to monitor weather updates regularly.',
                    'publishedAt': datetime.now().isoformat(),
                    'source': 'Weather Channel'
                }
            ]
        
        return templates[:np.random.randint(2, len(templates)+1)]


class ClimateDataCollector:
    """Collects climate data for temperature and soil moisture analysis"""
    
    def get_soil_moisture(self, lat: float, lon: float) -> float:
        """
        Get soil moisture data. Mock implementation for MVP.
        In production, would use Sentinel-1 or NASA SMAP data.
        """
        base_moisture = 0.35
        variation = np.random.normal(0, 0.1)
        return max(0.1, min(0.8, base_moisture + variation))
    
    def get_temperature_anomaly(self, current_temp: float, historical_mean: float = 27) -> float:
        """Calculate temperature anomaly"""
        return current_temp - historical_mean
    
    def get_land_cover_type(self, lat: float, lon: float) -> str:
        """
        Determine land cover type. Mock implementation for MVP.
        In production, would use ESA CCI Land Cover or similar.
        """
        if lat > 14.5 and lat < 14.65 and lon > 120.95 and lon < 121.05:
            return 'urban'
        elif np.random.random() > 0.7:
            return 'vegetation'
        else:
            return 'mixed'
