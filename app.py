import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os

from data_collectors import NewsDataCollector
from nlp_analyzer import NewsTextAnalyzer
from geospatial_features import GeospatialFeatureExtractor
from risk_model import RiskPredictionModel
from map_visualizer import RiskMapVisualizer

st.set_page_config(
    page_title="Community Flood & Heat Risk Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load or train the risk prediction model"""
    model = RiskPredictionModel()
    training_results = model.train()
    return model, training_results

@st.cache_data(ttl=600)
def fetch_news_data(location, query_type):
    """Fetch and cache news data"""
    news_collector = NewsDataCollector()
    nlp_analyzer = NewsTextAnalyzer()
    
    flood_articles = news_collector.get_news_articles('flood', location, days=7)
    heat_articles = news_collector.get_news_articles('heat heatwave', location, days=7)
    
    gdelt_flood = news_collector.get_gdelt_data('flood', location)
    gdelt_heat = news_collector.get_gdelt_data('heat', location)
    
    all_flood_articles = flood_articles + gdelt_flood
    all_heat_articles = heat_articles + gdelt_heat
    
    flood_signal = nlp_analyzer.get_aggregate_risk_signal(all_flood_articles, 'flood')
    heat_signal = nlp_analyzer.get_aggregate_risk_signal(all_heat_articles, 'heat')
    
    flood_df = nlp_analyzer.analyze_articles(all_flood_articles, 'flood')
    heat_df = nlp_analyzer.analyze_articles(all_heat_articles, 'heat')
    
    return {
        'flood_signal': flood_signal,
        'heat_signal': heat_signal,
        'flood_articles': flood_df,
        'heat_articles': heat_df
    }

def main():
    st.markdown('<h1 class="main-header">🌍 Community Flood & Heat Risk Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Driven Early Warning System for Climate Hazards**")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📍 Location Settings")
        location_preset = st.selectbox(
            "Select Location",
            ["Manila, Philippines", "Custom Coordinates"]
        )
        
        if location_preset == "Custom Coordinates":
            lat = st.number_input("Latitude", value=14.5995, min_value=-90.0, max_value=90.0, format="%.4f")
            lon = st.number_input("Longitude", value=120.9842, min_value=-180.0, max_value=180.0, format="%.4f")
            location_name = "Custom Location"
        else:
            lat = 14.5995
            lon = 120.9842
            location_name = "Manila"
        
        st.subheader("🔑 API Configuration")
        with st.expander("API Keys (Optional)"):
            st.info("API keys are optional. The system will use demonstration data if not provided.")
            openweather_key = st.text_input("OpenWeatherMap API Key", type="password", value="")
            newsapi_key = st.text_input("NewsAPI Key", type="password", value="")
            
            if openweather_key:
                os.environ['OPENWEATHER_API_KEY'] = openweather_key
            if newsapi_key:
                os.environ['NEWSAPI_KEY'] = newsapi_key
        
        st.subheader("📊 Analysis Options")
        show_historical = st.checkbox("Show Historical Trends", value=True)
        show_nlp_details = st.checkbox("Show NLP Analysis Details", value=True)
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        
        st.markdown("---")
        analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    if analyze_button or st.session_state.analyzed:
        st.session_state.analyzed = True
        
        with st.spinner("Loading risk prediction model..."):
            model, training_results = load_model()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model: Flood Accuracy", f"{training_results['flood_accuracy']:.1%}")
        with col2:
            st.metric("Model: Heat Accuracy", f"{training_results['heat_accuracy']:.1%}")
        with col3:
            st.metric("Training Samples", f"{training_results['training_samples']}")
        
        st.markdown("---")
        
        with st.spinner(f"Collecting data for {location_name}..."):
            feature_extractor = GeospatialFeatureExtractor()
            features = feature_extractor.extract_all_features(lat, lon)
            flood_factors = feature_extractor.calculate_flood_risk_factors(features)
            heat_factors = feature_extractor.calculate_heat_risk_factors(features)
            ml_features = feature_extractor.prepare_ml_features(features, flood_factors, heat_factors)
        
        with st.spinner("Analyzing news articles and social signals..."):
            news_data = fetch_news_data(location_name, 'both')
            flood_nlp_score = news_data['flood_signal']['avg_risk_score']
            heat_nlp_score = news_data['heat_signal']['avg_risk_score']
        
        with st.spinner("Predicting risk levels..."):
            flood_prediction = model.predict_flood_risk(ml_features, flood_nlp_score)
            heat_prediction = model.predict_heat_risk(ml_features, heat_nlp_score)
        
        st.header("🚨 Current Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💧 Flood Risk")
            risk_class = flood_prediction['risk_level']
            risk_color = 'risk-high' if risk_class == 'High' else 'risk-medium' if risk_class == 'Medium' else 'risk-low'
            st.markdown(f'<p class="{risk_color}" style="font-size: 2rem;">{risk_class}</p>', unsafe_allow_html=True)
            
            st.metric("Risk Score", f"{flood_prediction['risk_score']:.2%}")
            st.metric("ML Confidence", f"{max(flood_prediction['probabilities'].values()):.1%}")
            st.metric("NLP Contribution", f"{flood_prediction['nlp_contribution']:.3f}")
            
            st.progress(flood_prediction['risk_score'])
            
            if flood_prediction['risk_level'] == 'High':
                st.error("⚠️ HIGH FLOOD RISK: Immediate precautions recommended!")
            elif flood_prediction['risk_level'] == 'Medium':
                st.warning("⚡ MODERATE FLOOD RISK: Stay alert and monitor updates.")
            else:
                st.success("✅ LOW FLOOD RISK: Conditions are currently favorable.")
        
        with col2:
            st.subheader("🌡️ Heat Risk")
            risk_class = heat_prediction['risk_level']
            risk_color = 'risk-high' if risk_class == 'High' else 'risk-medium' if risk_class == 'Medium' else 'risk-low'
            st.markdown(f'<p class="{risk_color}" style="font-size: 2rem;">{risk_class}</p>', unsafe_allow_html=True)
            
            st.metric("Risk Score", f"{heat_prediction['risk_score']:.2%}")
            st.metric("ML Confidence", f"{max(heat_prediction['probabilities'].values()):.1%}")
            st.metric("NLP Contribution", f"{heat_prediction['nlp_contribution']:.3f}")
            
            st.progress(heat_prediction['risk_score'])
            
            if heat_prediction['risk_level'] == 'High':
                st.error("⚠️ EXTREME HEAT WARNING: Avoid outdoor activities!")
            elif heat_prediction['risk_level'] == 'Medium':
                st.warning("⚡ MODERATE HEAT RISK: Stay hydrated and limit sun exposure.")
            else:
                st.success("✅ LOW HEAT RISK: Conditions are comfortable.")
        
        st.markdown("---")
        
        st.header("🗺️ Interactive Risk Map")
        
        map_viz = RiskMapVisualizer(lat, lon)
        risk_map = map_viz.create_single_location_map(
            lat, lon, flood_prediction, heat_prediction, features
        )
        
        st_folium(risk_map, width=1200, height=600)
        
        st.markdown("---")
        
        st.header("📊 Environmental Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Temperature", f"{features['current_temperature']:.1f}°C")
            st.metric("💧 Humidity", f"{features['humidity']:.0f}%")
        
        with col2:
            st.metric("🌧️ Recent Rainfall (3d)", f"{features['recent_rainfall_3day']:.1f} mm")
            st.metric("📅 Weekly Rainfall", f"{features['total_weekly_rainfall']:.1f} mm")
        
        with col3:
            st.metric("⛰️ Elevation", f"{features['elevation']:.1f} m")
            st.metric("💦 Soil Moisture", f"{features['soil_moisture']:.2f}")
        
        with col4:
            st.metric("🏙️ Land Cover", features['land_cover'].title())
            st.metric("🌡️ Temp Anomaly", f"{features['temperature_anomaly']:+.1f}°C")
        
        if show_feature_importance:
            st.markdown("---")
            st.header("🔍 Feature Importance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Flood Risk Factors")
                flood_features = pd.DataFrame(
                    flood_prediction['top_features'],
                    columns=['Feature', 'Importance']
                )
                fig_flood = px.bar(
                    flood_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Most Important Features for Flood Prediction',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_flood, use_container_width=True)
            
            with col2:
                st.subheader("Top Heat Risk Factors")
                heat_features = pd.DataFrame(
                    heat_prediction['top_features'],
                    columns=['Feature', 'Importance']
                )
                fig_heat = px.bar(
                    heat_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Most Important Features for Heat Prediction',
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        
        if show_historical:
            st.markdown("---")
            st.header("📈 Historical Trends")
            
            historical_data = feature_extractor.get_historical_data(lat, lon, days=30)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['flood_risk_score'],
                name='Flood Risk',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['heat_risk_score'],
                name='Heat Risk',
                line=dict(color='#ff7f0e', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.2)'
            ))
            
            fig.update_layout(
                title='30-Day Risk Score History',
                xaxis_title='Date',
                yaxis_title='Risk Score',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_temp = px.line(
                    historical_data,
                    x='date',
                    y='temperature',
                    title='Temperature Trend',
                    labels={'temperature': 'Temperature (°C)', 'date': 'Date'}
                )
                fig_temp.update_traces(line_color='#dc3545')
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                fig_rain = px.bar(
                    historical_data,
                    x='date',
                    y='rainfall',
                    title='Rainfall History',
                    labels={'rainfall': 'Rainfall (mm)', 'date': 'Date'}
                )
                fig_rain.update_traces(marker_color='#1f77b4')
                st.plotly_chart(fig_rain, use_container_width=True)
        
        if show_nlp_details:
            st.markdown("---")
            st.header("📰 NLP Analysis - News Sentiment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Flood-Related News")
                st.metric("Total Articles Analyzed", news_data['flood_signal']['total_articles'])
                st.metric("Avg Risk Score", f"{news_data['flood_signal']['avg_risk_score']:.2%}")
                st.metric("High Risk Articles", news_data['flood_signal']['high_risk_articles'])
                
                if len(news_data['flood_articles']) > 0:
                    st.dataframe(
                        news_data['flood_articles'][['title', 'risk_score', 'sentiment_compound', 'source']].head(5),
                        use_container_width=True
                    )
            
            with col2:
                st.subheader("Heat-Related News")
                st.metric("Total Articles Analyzed", news_data['heat_signal']['total_articles'])
                st.metric("Avg Risk Score", f"{news_data['heat_signal']['avg_risk_score']:.2%}")
                st.metric("High Risk Articles", news_data['heat_signal']['high_risk_articles'])
                
                if len(news_data['heat_articles']) > 0:
                    st.dataframe(
                        news_data['heat_articles'][['title', 'risk_score', 'sentiment_compound', 'source']].head(5),
                        use_container_width=True
                    )
        
        st.markdown("---")
        
        st.header("ℹ️ Data Sources & Transparency")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Satellite & Weather Data**")
            st.write("• OpenWeatherMap API")
            st.write("• NASA GPM (Rainfall)")
            st.write("• Open Elevation API")
        
        with col2:
            st.markdown("**News & Sentiment Data**")
            st.write("• NewsAPI")
            st.write("• GDELT Project")
            st.write("• VADER Sentiment Analysis")
        
        with col3:
            st.markdown("**Model Information**")
            st.write("• Random Forest Classifier")
            st.write("• 19 Geospatial Features")
            st.write("• NLP Feature Fusion (20%)")
        
        st.info(f"**Last Updated:** {features['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | **Location:** {lat:.4f}, {lon:.4f}")
    
    else:
        st.info("👈 Configure your settings in the sidebar and click 'Analyze Risk' to begin.")
        
        st.markdown("""
        ### About This System
        
        The **Community Flood and Heat Risk Analyzer** is an AI-driven early warning system that combines:
        
        - 🛰️ **Satellite & Meteorological Data**: Real-time weather, rainfall, elevation, and climate indicators
        - 📰 **News & Social Intelligence**: NLP analysis of local news articles and bulletins
        - 🤖 **Machine Learning**: Random Forest models trained on multi-source environmental data
        - 🗺️ **Interactive Visualization**: Color-coded risk zones on interactive maps
        
        ### Key Features
        
        - ✅ Real-time flood and heat risk prediction
        - ✅ Explainable AI with feature importance rankings
        - ✅ Historical trend analysis (30-day history)
        - ✅ News sentiment integration using VADER NLP
        - ✅ Interactive Folium maps with risk zones
        - ✅ Support for any geographic location
        
        ### Demo Location: Manila, Philippines
        
        Manila was chosen as the demo location because it frequently experiences both floods (monsoons, typhoons) 
        and extreme heat events, making it ideal for demonstrating the system's capabilities.
        
        ---
        
        **Get Started:** Use the sidebar to configure your analysis and explore climate risks in your community.
        """)

if __name__ == "__main__":
    main()
