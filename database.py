"""
Database module for Community Flood and Heat Risk Analyzer
Uses SQLAlchemy ORM for PostgreSQL database operations
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional
import streamlit as st

# Create declarative base
Base = declarative_base()

# User model for authentication and preferences
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    locations = relationship("SavedLocation", back_populates="user", cascade="all, delete-orphan")
    alert_settings = relationship("AlertThreshold", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

# Saved locations for monitoring
class SavedLocation(Base):
    __tablename__ = 'saved_locations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationship
    user = relationship("User", back_populates="locations")
    risk_history = relationship("RiskHistory", back_populates="location", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SavedLocation(name='{self.name}', lat={self.latitude}, lon={self.longitude})>"

# Alert thresholds for user-defined risk levels
class AlertThreshold(Base):
    __tablename__ = 'alert_thresholds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    risk_type = Column(String(20), nullable=False)  # 'flood' or 'heat'
    threshold_level = Column(String(20), nullable=False)  # 'Low', 'Medium', 'High'
    min_risk_score = Column(Float, nullable=False)  # Minimum risk score to trigger alert (0.0-1.0)
    notify_email = Column(Boolean, default=True)
    notify_sms = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="alert_settings")
    
    def __repr__(self):
        return f"<AlertThreshold(type='{self.risk_type}', threshold='{self.threshold_level}', score={self.min_risk_score})>"

# Historical risk data for trend analysis
class RiskHistory(Base):
    __tablename__ = 'risk_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('saved_locations.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Risk predictions
    flood_risk_level = Column(String(20))  # 'Low', 'Medium', 'High'
    flood_risk_score = Column(Float)
    heat_risk_level = Column(String(20))
    heat_risk_score = Column(Float)
    
    # Contributing factors
    temperature = Column(Float)
    humidity = Column(Float)
    rainfall_24h = Column(Float)
    elevation = Column(Float)
    
    # Model metadata
    model_used = Column(String(50))  # 'ensemble', 'random_forest', 'xgboost'
    nlp_mode = Column(String(50))  # 'enhanced', 'basic'
    
    # Relationship
    location = relationship("SavedLocation", back_populates="risk_history")
    
    def __repr__(self):
        return f"<RiskHistory(location_id={self.location_id}, flood={self.flood_risk_level}, heat={self.heat_risk_level})>"

# Alert notifications log
class AlertLog(Base):
    __tablename__ = 'alert_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    location_id = Column(Integer, ForeignKey('saved_locations.id'), nullable=False)
    risk_type = Column(String(20), nullable=False)
    risk_level = Column(String(20), nullable=False)
    risk_score = Column(Float, nullable=False)
    message = Column(Text)
    sent_at = Column(DateTime, default=datetime.utcnow, index=True)
    notification_method = Column(String(20))  # 'email', 'sms'
    was_successful = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<AlertLog(user_id={self.user_id}, type='{self.risk_type}', level='{self.risk_level}')>"

# Database connection and session management
@st.cache_resource
def get_database_engine():
    """Create and cache database engine"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,
        max_overflow=10,
        echo=False  # Set to True for SQL logging during development
    )
    return engine

def init_database():
    """Initialize database tables"""
    engine = get_database_engine()
    Base.metadata.create_all(engine)
    return engine

@st.cache_resource
def get_session_maker():
    """Create and cache session maker"""
    engine = get_database_engine()
    Session = sessionmaker(bind=engine)
    return Session

def get_db_session():
    """Get a new database session"""
    Session = get_session_maker()
    return Session()

# Helper functions for common database operations
def create_user(username: str, email: str, password_hash: str) -> User:
    """Create a new user"""
    session = get_db_session()
    try:
        user = User(username=username, email=email, password_hash=password_hash)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_by_username(username: str) -> User:
    """Get user by username"""
    session = get_db_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        return user
    finally:
        session.close()

def get_user_by_email(email: str) -> User:
    """Get user by email"""
    session = get_db_session()
    try:
        user = session.query(User).filter(User.email == email).first()
        return user
    finally:
        session.close()

def create_saved_location(user_id: int, name: str, latitude: float, longitude: float, description: Optional[str] = None) -> SavedLocation:
    """Create a new saved location for a user"""
    session = get_db_session()
    try:
        location = SavedLocation(
            user_id=user_id,
            name=name,
            latitude=latitude,
            longitude=longitude,
            description=description
        )
        session.add(location)
        session.commit()
        session.refresh(location)
        return location
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_locations(user_id: int, active_only: bool = True):
    """Get all saved locations for a user"""
    session = get_db_session()
    try:
        query = session.query(SavedLocation).filter(SavedLocation.user_id == user_id)
        if active_only:
            query = query.filter(SavedLocation.is_active == True)
        locations = query.all()
        return locations
    finally:
        session.close()

def create_alert_threshold(user_id: int, risk_type: str, threshold_level: str, min_risk_score: float, 
                          notify_email: bool = True, notify_sms: bool = False) -> AlertThreshold:
    """Create a new alert threshold for a user"""
    session = get_db_session()
    try:
        threshold = AlertThreshold(
            user_id=user_id,
            risk_type=risk_type,
            threshold_level=threshold_level,
            min_risk_score=min_risk_score,
            notify_email=notify_email,
            notify_sms=notify_sms
        )
        session.add(threshold)
        session.commit()
        session.refresh(threshold)
        return threshold
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_alert_thresholds(user_id: int, active_only: bool = True):
    """Get all alert thresholds for a user"""
    session = get_db_session()
    try:
        query = session.query(AlertThreshold).filter(AlertThreshold.user_id == user_id)
        if active_only:
            query = query.filter(AlertThreshold.is_active == True)
        thresholds = query.all()
        return thresholds
    finally:
        session.close()

def save_risk_history(location_id: int, flood_prediction: dict, heat_prediction: dict, 
                     features: dict, model_used: str, nlp_mode: str) -> RiskHistory:
    """Save risk prediction to history"""
    session = get_db_session()
    try:
        history = RiskHistory(
            location_id=location_id,
            flood_risk_level=flood_prediction['risk_level'],
            flood_risk_score=flood_prediction['risk_score'],
            heat_risk_level=heat_prediction['risk_level'],
            heat_risk_score=heat_prediction['risk_score'],
            temperature=features.get('temperature'),
            humidity=features.get('humidity'),
            rainfall_24h=features.get('rainfall_24h'),
            elevation=features.get('elevation'),
            model_used=model_used,
            nlp_mode=nlp_mode
        )
        session.add(history)
        session.commit()
        session.refresh(history)
        return history
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_location_risk_history(location_id: int, days: int = 30):
    """Get risk history for a location (last N days)"""
    session = get_db_session()
    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        history = session.query(RiskHistory).filter(
            RiskHistory.location_id == location_id,
            RiskHistory.timestamp >= cutoff_date
        ).order_by(RiskHistory.timestamp.desc()).all()
        
        return history
    finally:
        session.close()

def log_alert(user_id: int, location_id: int, risk_type: str, risk_level: str, 
             risk_score: float, message: str, notification_method: str, 
             was_successful: bool = True) -> AlertLog:
    """Log an alert notification"""
    session = get_db_session()
    try:
        log = AlertLog(
            user_id=user_id,
            location_id=location_id,
            risk_type=risk_type,
            risk_level=risk_level,
            risk_score=risk_score,
            message=message,
            notification_method=notification_method,
            was_successful=was_successful
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return log
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_user_alert_logs(user_id: int, days: int = 30):
    """Get alert logs for a user (last N days)"""
    session = get_db_session()
    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        logs = session.query(AlertLog).filter(
            AlertLog.user_id == user_id,
            AlertLog.sent_at >= cutoff_date
        ).order_by(AlertLog.sent_at.desc()).all()
        
        return logs
    finally:
        session.close()
