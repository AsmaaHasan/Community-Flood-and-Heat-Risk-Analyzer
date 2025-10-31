# 👨‍💻 Developer Guide

Technical documentation for developers working on the Community Flood and Heat Risk Analyzer.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Key Components](#key-components)
4. [Development Setup](#development-setup)
5. [Testing](#testing)
6. [Contributing](#contributing)
7. [API Documentation](#api-documentation)

## Architecture Overview

### System Design

The application follows a layered architecture:

```
┌─────────────────────────────────────┐
│     Presentation Layer (app.py)     │
│   Streamlit UI + Interactive Maps   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│      Analysis & ML Layer            │
│  - Enhanced NLP (TF-IDF + NER)      │
│  - Ensemble ML (RF + XGBoost)       │
│  - Feature Engineering              │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│     Data Collection Layer           │
│  - Weather APIs                     │
│  - News APIs                        │
│  - Geospatial Services              │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│     Data Persistence Layer          │
│  - PostgreSQL (SQLAlchemy ORM)      │
│  - User accounts, locations, alerts │
└─────────────────────────────────────┘
```

### Technology Stack

**Frontend**:
- Streamlit 1.51+ (Web framework)
- Folium (Interactive maps)
- Plotly (Visualizations)

**ML & NLP**:
- Scikit-learn (Random Forest, TF-IDF, Naive Bayes)
- XGBoost (Gradient boosting)
- NLTK (NER, tokenization)
- VADER (Sentiment analysis)

**Data Processing**:
- Pandas (Data manipulation)
- NumPy (Numerical operations)
- GeoPandas (Geospatial data)

**Database**:
- PostgreSQL 14+
- SQLAlchemy 2.0+ (ORM)

**APIs**:
- OpenWeatherMap (Weather data)
- NewsAPI (News articles)
- GDELT (Global events)
- Open Elevation (Topography)

## Code Structure

```
.
├── app.py                      # Main Streamlit application
├── enhanced_risk_model.py      # Ensemble ML models (RF + XGBoost)
├── risk_model.py              # Legacy Random Forest model
├── enhanced_nlp_analyzer.py   # Multi-method NLP analysis
├── nlp_analyzer.py            # Basic VADER sentiment
├── data_collectors.py         # API data collection
├── geospatial_features.py     # Feature extraction
├── map_visualizer.py          # Folium map generation
├── database.py                # SQLAlchemy ORM models
├── init_db.py                 # Database initialization
├── pyproject.toml             # Dependencies
├── README.md                  # Project overview
├── QUICKSTART.md              # Quick start guide
├── DEPLOYMENT_GUIDE.md        # Deployment instructions
└── DEVELOPER_GUIDE.md         # This file
```

## Key Components

### 1. Data Collection (`data_collectors.py`)

**Classes**:
- `WeatherDataCollector`: OpenWeatherMap API integration
- `RainfallDataCollector`: NASA GPM satellite data
- `ElevationDataCollector`: Topographic data
- `NewsDataCollector`: NewsAPI + GDELT integration

**Design Pattern**: Collector pattern with graceful fallback to mock data

**Example**:
```python
from data_collectors import WeatherDataCollector

collector = WeatherDataCollector()
weather = collector.get_current_weather(14.5995, 120.9842)
# Returns real data if API key available, otherwise realistic mock data
```

### 2. Feature Engineering (`geospatial_features.py`)

**Class**: `GeospatialFeatureExtractor`

**Key Methods**:
- `extract_all_features(lat, lon)`: Get 19 features from multiple sources
- `calculate_flood_risk_factors()`: Compute flood-specific indicators
- `calculate_heat_risk_factors()`: Compute heat-specific indicators
- `prepare_ml_features()`: Format for ML model input

**Features Extracted** (19 total):
1. Temperature (current)
2. Temperature anomaly
3. Humidity
4. Pressure
5. Wind speed
6. Rainfall (24h)
7. Rainfall (3-day average)
8. Rainfall (weekly total)
9. Rainfall trend
10. Rainfall anomaly
11. Elevation
12. Soil moisture
13. Soil saturation
14. Land cover type
15. Drainage capacity
16. Heat index
17. Heat index anomaly
18. Urban heat island effect
19. Vegetation health

### 3. Enhanced NLP (`enhanced_nlp_analyzer.py`)

**Class**: `EnhancedNewsTextAnalyzer`

**Multi-Method Analysis**:

1. **TF-IDF Classification** (25% weight)
   - Naive Bayes classifier
   - Random Forest classifier
   - Trained on synthetic flood/heat/neutral samples

2. **Urgency Detection** (25% weight)
   - Temporal pattern matching
   - Urgency keyword scoring
   - Weighted by context

3. **Weighted Keywords** (30% weight)
   - Context-aware matching
   - Severity-based weights
   - "catastrophic flood" > "rainfall"

4. **VADER Sentiment** (20% weight)
   - Baseline sentiment analysis
   - Negative sentiment as risk indicator

5. **Named Entity Recognition**
   - Extract locations, organizations, persons
   - Provides context for risk assessment

**Example**:
```python
from enhanced_nlp_analyzer import EnhancedNewsTextAnalyzer

analyzer = EnhancedNewsTextAnalyzer()
article_text = "Flash flood emergency declared in Manila..."

# Full analysis
risk_score = analyzer.calculate_enhanced_risk_score(article_text, 'flood')
# Returns: {
#   'total_risk_score': 0.85,
#   'keyword_score': 0.9,
#   'urgency_score': 1.0,
#   'sentiment_score': 0.7,
#   'ml_score': 0.8
# }

# Entity extraction
entities = analyzer.extract_named_entities(article_text)
# Returns: {'locations': ['Manila'], 'organizations': [], 'persons': []}
```

### 4. Ensemble ML Models (`enhanced_risk_model.py`)

**Class**: `EnhancedRiskPredictionModel`

**Model Types**:
- `random_forest`: Baseline Random Forest
- `xgboost`: XGBoost gradient boosting
- `ensemble`: Average of both models

**Architecture**:
```python
# Flood Risk Models
- Random Forest: 100 trees, max_depth=10
- XGBoost: 100 estimators, max_depth=6, lr=0.1, objective='multi:softprob'

# Heat Risk Models
- Random Forest: 100 trees, max_depth=10
- XGBoost: 100 estimators, max_depth=6, lr=0.1, objective='multi:softprob'
```

**Training**:
- Synthetic data generation (800 samples)
- Train/test split: 80/20
- Validation accuracy: >85%

**Example**:
```python
from enhanced_risk_model import EnhancedRiskPredictionModel

# Train model
model = EnhancedRiskPredictionModel(model_type='ensemble')
results = model.train()

# Make predictions
features = [28.5, 0.8, 75, 1012, 15, ...]  # 19 features
nlp_score = 0.65

prediction = model.predict_flood_risk(features, nlp_score)
# Returns: {
#   'risk_level': 'High',
#   'risk_score': 0.78,
#   'probabilities': {'Low': 0.05, 'Medium': 0.17, 'High': 0.78},
#   'nlp_contribution': 0.13,
#   'model_used': 'ensemble'
# }
```

### 5. Database Layer (`database.py`)

**SQLAlchemy ORM Models**:

```python
# User accounts
class User(Base):
    id, username, email, password_hash, created_at, last_login, is_active

# Saved monitoring locations
class SavedLocation(Base):
    id, user_id, name, latitude, longitude, description, created_at, is_active

# Alert thresholds
class AlertThreshold(Base):
    id, user_id, risk_type, threshold_level, min_risk_score, 
    notify_email, notify_sms, is_active

# Historical risk data
class RiskHistory(Base):
    id, location_id, timestamp, flood_risk_level, flood_risk_score,
    heat_risk_level, heat_risk_score, temperature, humidity, 
    rainfall_24h, elevation, model_used, nlp_mode

# Alert logs
class AlertLog(Base):
    id, user_id, location_id, risk_type, risk_level, risk_score,
    message, sent_at, notification_method, was_successful
```

**Helper Functions**:
```python
# User management
create_user(username, email, password_hash)
get_user_by_username(username)
get_user_by_email(email)

# Location management
create_saved_location(user_id, name, lat, lon, description)
get_user_locations(user_id, active_only=True)

# Alert management
create_alert_threshold(user_id, risk_type, threshold_level, min_risk_score)
get_user_alert_thresholds(user_id, active_only=True)

# History tracking
save_risk_history(location_id, flood_pred, heat_pred, features, model, nlp_mode)
get_location_risk_history(location_id, days=30)

# Alert logging
log_alert(user_id, location_id, risk_type, risk_level, risk_score, message, method)
get_user_alert_logs(user_id, days=30)
```

### 6. Map Visualization (`map_visualizer.py`)

**Class**: `RiskMapVisualizer`

**Features**:
- Multi-layer tile support (OSM, Google Satellite)
- Color-coded risk markers (green/yellow/red)
- Pop-up details with risk metrics
- Heatmap overlays
- Layer controls

**Example**:
```python
from map_visualizer import RiskMapVisualizer

visualizer = RiskMapVisualizer()
risk_map = visualizer.create_risk_map(
    center_lat=14.5995,
    center_lon=120.9842,
    flood_risk='High',
    heat_risk='Medium',
    features={'temperature': 32.5, 'humidity': 78, ...}
)
```

## Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd community-flood-heat-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and database URL
```

### 2. Database Setup

```bash
# Initialize database
python init_db.py

# Verify tables
python -c "from database import get_database_engine; from sqlalchemy import inspect; print(inspect(get_database_engine()).get_table_names())"
```

### 3. Run Development Server

```bash
streamlit run app.py --server.port 5000
```

Access at: http://localhost:5000

### 4. Development Tools

**Code Formatting**:
```bash
# Install black
pip install black

# Format code
black *.py
```

**Type Checking**:
```bash
# Install mypy
pip install mypy

# Run type checker
mypy *.py
```

**Linting**:
```bash
# Install pylint
pip install pylint

# Lint code
pylint *.py
```

## Testing

### Unit Tests

Create `tests/test_features.py`:
```python
import unittest
from geospatial_features import GeospatialFeatureExtractor

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = GeospatialFeatureExtractor()
    
    def test_feature_count(self):
        features = self.extractor.extract_all_features(14.5995, 120.9842)
        self.assertEqual(len(features), 19)
    
    def test_feature_types(self):
        features = self.extractor.extract_all_features(14.5995, 120.9842)
        for feature in features.values():
            self.assertIsInstance(feature, (int, float))

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Test end-to-end workflows:
```python
def test_full_prediction_pipeline():
    # Extract features
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_all_features(14.5995, 120.9842)
    
    # Get NLP scores
    analyzer = EnhancedNewsTextAnalyzer()
    nlp_score = 0.5  # Mock score
    
    # Train model
    model = EnhancedRiskPredictionModel(model_type='ensemble')
    model.train()
    
    # Make prediction
    flood_factors = extractor.calculate_flood_risk_factors(features)
    heat_factors = extractor.calculate_heat_risk_factors(features)
    ml_features = extractor.prepare_ml_features(features, flood_factors, heat_factors)
    
    prediction = model.predict_flood_risk(ml_features, nlp_score)
    
    assert prediction['risk_level'] in ['Low', 'Medium', 'High']
    assert 0 <= prediction['risk_score'] <= 1
```

### Database Tests

```python
def test_user_creation():
    from database import create_user, get_user_by_username
    
    user = create_user('testuser', 'test@example.com', 'hashed_password')
    assert user.username == 'testuser'
    
    retrieved = get_user_by_username('testuser')
    assert retrieved.email == 'test@example.com'
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Pull request process
- Issue reporting
- Feature requests

## API Documentation

### Data Collectors API

```python
# Weather Data
collector = WeatherDataCollector()
current = collector.get_current_weather(lat, lon)
forecast = collector.get_forecast(lat, lon, days=5)

# News Data
news = NewsDataCollector()
articles = news.get_news_articles(query='flood', location='Manila', days=7)
gdelt_data = news.get_gdelt_data(query='heat', location='Philippines')
```

### Feature Extraction API

```python
extractor = GeospatialFeatureExtractor()

# Get all features
features = extractor.extract_all_features(lat, lon)

# Calculate risk factors
flood_factors = extractor.calculate_flood_risk_factors(features)
heat_factors = extractor.calculate_heat_risk_factors(features)

# Prepare for ML
ml_features = extractor.prepare_ml_features(features, flood_factors, heat_factors)
```

### NLP Analysis API

```python
analyzer = EnhancedNewsTextAnalyzer()

# Analyze single text
risk_score = analyzer.calculate_enhanced_risk_score(text, risk_type='flood')

# Analyze articles
df = analyzer.analyze_articles_enhanced(articles, risk_type='heat')

# Get aggregate signal
signal = analyzer.get_aggregate_risk_signal_enhanced(articles, risk_type='flood')

# Extract entities
entities = analyzer.extract_named_entities(text)

# Detect urgency
urgency = analyzer.detect_urgency(text)
```

### ML Model API

```python
model = EnhancedRiskPredictionModel(model_type='ensemble')

# Train model
results = model.train()

# Make predictions
flood_pred = model.predict_flood_risk(features, nlp_score)
heat_pred = model.predict_heat_risk(features, nlp_score)

# Get feature importance
importance_flood = model.get_feature_importance('flood')
importance_heat = model.get_feature_importance('heat')

# Compare models
comparison = model.get_performance_comparison()
```

### Database API

```python
# Session management
session = get_db_session()

# User operations
user = create_user('john', 'john@example.com', 'hash')
user = get_user_by_username('john')

# Location operations
location = create_saved_location(user.id, 'Home', 14.5995, 120.9842)
locations = get_user_locations(user.id)

# Alert operations
threshold = create_alert_threshold(user.id, 'flood', 'High', 0.7)
thresholds = get_user_alert_thresholds(user.id)

# History tracking
history = save_risk_history(location.id, flood_pred, heat_pred, features, 'ensemble', 'enhanced')
past_data = get_location_risk_history(location.id, days=30)
```

## Performance Optimization Tips

1. **Caching**: Use `@st.cache_data` for data fetching, `@st.cache_resource` for models
2. **Database**: Index frequently queried columns, use connection pooling
3. **ML Models**: Use XGBoost only (faster than ensemble) for real-time applications
4. **NLP**: Disable Enhanced NLP for faster processing if accuracy requirements are lower
5. **API Calls**: Batch requests, implement rate limiting, cache responses

## Debugging Tips

1. **Enable SQL Logging**: Set `echo=True` in `create_engine()` call
2. **Streamlit Debug Mode**: Run with `streamlit run app.py --logger.level debug`
3. **Check Logs**: Monitor `~/.streamlit/logs/` for application logs
4. **Network Issues**: Use `requests.get(..., timeout=10)` to avoid hanging
5. **Memory Leaks**: Monitor session state size, clear unused cache entries

## Next Steps

- Implement comprehensive unit tests
- Add API documentation with Swagger/OpenAPI
- Set up CI/CD pipeline
- Implement user authentication UI
- Add alert monitoring system
- Integrate Sentinel satellite data

For questions or contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).
