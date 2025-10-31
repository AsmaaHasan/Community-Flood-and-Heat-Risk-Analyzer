import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class RiskMapVisualizer:
    """Create interactive Folium maps with risk zones and alerts"""
    
    def __init__(self, center_lat: float = 14.5995, center_lon: float = 120.9842):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.manila_bounds = {
            'north': 14.7642,
            'south': 14.3487,
            'east': 121.1501,
            'west': 120.8940
        }
    
    def create_base_map(self, zoom_start: int = 11) -> folium.Map:
        """Create base map centered on location"""
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer('OpenStreetMap').add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def get_risk_color(self, risk_level: str) -> str:
        """Get color code for risk level"""
        colors = {
            'Low': '#28a745',
            'Medium': '#ffc107', 
            'High': '#dc3545'
        }
        return colors.get(risk_level, '#6c757d')
    
    def get_risk_icon(self, risk_type: str) -> str:
        """Get icon for risk type"""
        icons = {
            'flood': 'tint',
            'heat': 'thermometer-full'
        }
        return icons.get(risk_type, 'exclamation-triangle')
    
    def add_risk_marker(self, m: folium.Map, lat: float, lon: float, 
                       risk_type: str, risk_level: str, risk_score: float,
                       details: Dict = None) -> folium.Map:
        """Add risk marker to map"""
        color = self.get_risk_color(risk_level)
        icon = self.get_risk_icon(risk_type)
        
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: {color};">
                {risk_type.title()} Risk: {risk_level}
            </h4>
            <hr style="margin: 5px 0;">
            <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
            <p><strong>Location:</strong> {lat:.4f}, {lon:.4f}</p>
        """
        
        if details:
            if 'temperature' in details:
                popup_html += f"<p><strong>Temperature:</strong> {details['temperature']:.1f}°C</p>"
            if 'rainfall' in details:
                popup_html += f"<p><strong>Recent Rainfall:</strong> {details['rainfall']:.1f}mm</p>"
            if 'elevation' in details:
                popup_html += f"<p><strong>Elevation:</strong> {details['elevation']:.1f}m</p>"
        
        popup_html += "</div>"
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{risk_type.title()} Risk: {risk_level}",
            icon=folium.Icon(color=color.replace('#', '').lower() if color == '#28a745' else 
                           ('green' if color == '#28a745' else 'orange' if color == '#ffc107' else 'red'),
                           icon=icon)
        ).add_to(m)
        
        return m
    
    def add_risk_circle(self, m: folium.Map, lat: float, lon: float,
                       risk_level: str, risk_score: float, 
                       radius: float = 2000) -> folium.Map:
        """Add colored circle showing risk zone"""
        color = self.get_risk_color(risk_level)
        
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.3,
            opacity=0.6,
            popup=f"Risk Level: {risk_level}<br>Score: {risk_score:.2f}",
            tooltip=f"{risk_level} Risk Zone"
        ).add_to(m)
        
        return m
    
    def add_heatmap(self, m: folium.Map, locations: List[Tuple[float, float, float]]) -> folium.Map:
        """Add heatmap layer showing risk intensity"""
        if locations:
            plugins.HeatMap(
                locations,
                min_opacity=0.3,
                max_opacity=0.8,
                radius=25,
                blur=20,
                gradient={0.4: 'blue', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
            ).add_to(m)
        
        return m
    
    def add_legend(self, m: folium.Map) -> folium.Map:
        """Add custom legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto;
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
            <p style="margin: 0 0 10px 0; font-weight: bold;">Risk Levels</p>
            <p style="margin: 5px 0;"><i style="background:#28a745; width: 20px; height: 20px; 
               display: inline-block; border-radius: 3px;"></i> Low Risk</p>
            <p style="margin: 5px 0;"><i style="background:#ffc107; width: 20px; height: 20px; 
               display: inline-block; border-radius: 3px;"></i> Medium Risk</p>
            <p style="margin: 5px 0;"><i style="background:#dc3545; width: 20px; height: 20px; 
               display: inline-block; border-radius: 3px;"></i> High Risk</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_multi_point_risk_map(self, risk_points: List[Dict]) -> folium.Map:
        """Create map with multiple risk assessment points"""
        m = self.create_base_map()
        
        heatmap_data = []
        
        for point in risk_points:
            lat = point['lat']
            lon = point['lon']
            risk_type = point['risk_type']
            risk_level = point['risk_level']
            risk_score = point['risk_score']
            details = point.get('details', {})
            
            self.add_risk_marker(m, lat, lon, risk_type, risk_level, risk_score, details)
            self.add_risk_circle(m, lat, lon, risk_level, risk_score, radius=1500)
            
            heatmap_data.append([lat, lon, risk_score])
        
        if heatmap_data:
            self.add_heatmap(m, heatmap_data)
        
        self.add_legend(m)
        
        return m
    
    def create_single_location_map(self, lat: float, lon: float,
                                   flood_risk: Dict, heat_risk: Dict,
                                   features: Dict = None) -> folium.Map:
        """Create detailed map for single location with both risk types"""
        m = self.create_base_map(zoom_start=12)
        
        flood_details = {
            'temperature': features.get('current_temperature', 0) if features else 0,
            'rainfall': features.get('recent_rainfall_3day', 0) if features else 0,
            'elevation': features.get('elevation', 0) if features else 0
        }
        
        heat_details = {
            'temperature': features.get('current_temperature', 0) if features else 0,
            'humidity': features.get('humidity', 0) if features else 0,
            'heat_index': flood_details['temperature'] + (0.5 * (heat_details.get('humidity', 70) - 40) / 10)
        }
        
        self.add_risk_marker(
            m, lat - 0.01, lon - 0.01, 'flood', 
            flood_risk['risk_level'], flood_risk['risk_score'],
            flood_details
        )
        
        self.add_risk_marker(
            m, lat + 0.01, lon + 0.01, 'heat',
            heat_risk['risk_level'], heat_risk['risk_score'],
            heat_details
        )
        
        self.add_risk_circle(
            m, lat, lon, flood_risk['risk_level'], 
            flood_risk['risk_score'], radius=3000
        )
        
        folium.Marker(
            location=[lat, lon],
            popup="Analysis Center Point",
            tooltip="Center",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        self.add_legend(m)
        
        return m
