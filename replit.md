# Community Flood and Heat Risk Analyzer

## Overview

This is an AI-driven early warning system that helps communities assess and prepare for climate hazards by combining satellite geospatial data, meteorological inputs, and NLP-based text analysis. The application provides real-time flood and heat risk predictions using machine learning models, visualizes risk zones on interactive maps, and analyzes news sentiment to detect early warning signals.

The system is designed to work with any geographic coordinates but is optimized for demonstration purposes using Manila, Philippines as the default location. It uses 100% free data sources and open-source libraries, making it accessible for community-level deployment.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Data Collection Architecture

**Multi-Source Data Integration**: The application implements a modular data collection layer that integrates multiple external APIs and services:

- **Weather Data**: OpenWeatherMap API provides current conditions and 5-day forecasts (temperature, humidity, pressure, wind speed)
- **Rainfall Data**: NASA GPM (Global Precipitation Measurement) satellite data for precipitation monitoring
- **Elevation Data**: Topographic services to determine flood susceptibility based on terrain
- **Climate Indicators**: Soil moisture and land cover classification data
- **News Data**: NewsAPI and GDELT for real-time news article collection

**Design Pattern**: The data collection layer uses a collector pattern where each data source has a dedicated collector class (`WeatherDataCollector`, `RainfallDataCollector`, etc.). This provides clean separation of concerns and makes it easy to swap or add data sources. Each collector implements fallback mechanisms with mock data generation to ensure the system continues functioning even when API keys are unavailable or rate limits are reached.

### Feature Engineering Architecture

**Geospatial Feature Extraction**: The `GeospatialFeatureExtractor` class orchestrates data collection from multiple sources and transforms raw data into 19 standardized features for machine learning models:

- Basic meteorological features (temperature, humidity, pressure, wind speed)
- Derived risk indicators (temperature anomaly, rainfall anomaly, soil saturation)
- Geographic characteristics (elevation, land cover type)
- Temporal aggregations (3-day rainfall average, weekly totals, trends)

**Feature Fusion Strategy**: Combines numeric geospatial features with NLP-derived sentiment scores to create a comprehensive risk assessment. This hybrid approach captures both objective physical measurements and subjective signals from news reporting.

### NLP Analysis Architecture

**Sentiment-Based Risk Detection**: Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis to process news articles:

- Keyword extraction for flood-specific and heat-specific terms
- Risk modifier detection (warning, alert, severe, emergency)
- Sentiment scoring with emphasis on negative sentiment as risk indicator
- Text preprocessing with NLTK tokenization and stopword removal

**Rationale**: VADER is chosen over transformer-based models for its speed, zero training requirements, and proven effectiveness on social media/news text. This keeps the system lightweight and responsive.

### Machine Learning Architecture

**Dual Random Forest Classifier Design**: Implements two separate Random Forest models:

1. **Flood Risk Model**: Trained to classify flood risk as Low/Medium/High based on rainfall, elevation, soil moisture, and drainage indicators
2. **Heat Risk Model**: Trained to classify heat risk based on temperature, humidity, heat index, and urban heat island factors

**Training Strategy**: Models are trained on synthetic/demonstration data that reflects realistic climate patterns. The system is designed to be retrained with real historical incident data when available.

**Feature Importance**: Models expose feature importance rankings to provide explainable AI capabilities, helping users understand which factors are driving risk predictions.

### Visualization Architecture

**Interactive Mapping**: Uses Folium library to create web-based interactive maps with:

- Multi-layer tile support (OpenStreetMap, Google Satellite)
- Color-coded risk zones (green/yellow/red for Low/Medium/High)
- Pop-up markers with detailed risk metrics
- Heatmap overlays for risk intensity visualization
- Layer controls for user customization

**Dashboard Framework**: Streamlit provides the web application framework with:

- Real-time data refresh with 10-minute caching intervals
- Responsive multi-column layout
- Plotly-based charts for historical trend analysis
- Metric cards for at-a-glance risk assessment
- Custom CSS styling for risk-level color coding

### Data Flow Pattern

1. User requests analysis for a location (latitude/longitude)
2. Geospatial features are extracted from multiple APIs
3. News articles are collected and analyzed for sentiment
4. Features are standardized and fed to ML models
5. Predictions are generated and combined with NLP scores
6. Results are visualized on interactive map and dashboard
7. Historical data is stored for 30-day trend analysis

### Caching and Performance

**Streamlit Caching**: The `@st.cache_data` decorator is used with TTL (time-to-live) settings to:

- Cache API responses for 10 minutes to reduce external calls
- Cache model predictions to improve dashboard responsiveness
- Balance freshness with performance

## External Dependencies

### Third-Party APIs

**OpenWeatherMap API** (Optional, free tier)
- Purpose: Real-time weather data and forecasts
- Fallback: Mock data generator provides demonstration values
- Rate limits: Handled with graceful degradation

**NewsAPI** (Optional, free tier)
- Purpose: News article collection for sentiment analysis
- Fallback: System continues with geospatial-only predictions
- Alternative: GDELT API can be used as backup source

**NASA GPM API**
- Purpose: Satellite-based precipitation measurements
- Access: Public data, no authentication required

**Elevation Services**
- Purpose: Topographic data for flood risk calculation
- Access: Public elevation APIs

### Python Libraries

**Core Data Science Stack**:
- `pandas`, `numpy`: Data manipulation and numerical operations
- `scikit-learn`: Random Forest models, StandardScaler preprocessing
- `pickle`: Model serialization for persistence

**Geospatial Libraries**:
- `geopandas`: Geographic data structures
- `rasterio`: Raster data processing for satellite imagery
- `folium`: Interactive map generation

**NLP Libraries**:
- `nltk`: Text tokenization and preprocessing
- `vaderSentiment`: Sentiment analysis engine

**Visualization**:
- `streamlit`: Web application framework
- `streamlit-folium`: Folium map integration
- `plotly`: Interactive charts and graphs

**HTTP Requests**:
- `requests`: API communication with timeout handling

### Environment Variables

The application uses optional environment variables for API keys:
- `OPENWEATHER_API_KEY`: OpenWeatherMap authentication
- `NEWS_API_KEY`: NewsAPI authentication

The system is designed to function without these keys using mock data, making it ideal for demonstration and development.