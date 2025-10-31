import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re

class NewsTextAnalyzer:
    """NLP-based sentiment and risk analysis for news articles"""
    
    def __init__(self):
        self._download_nltk_data()
        self.vader = SentimentIntensityAnalyzer()
        
        self.flood_keywords = [
            'flood', 'flooding', 'inundation', 'overflow', 'deluge',
            'rainfall', 'monsoon', 'storm', 'typhoon', 'downpour',
            'waterlogged', 'submerged', 'evacuate', 'evacuation',
            'drainage', 'water level', 'rising water', 'heavy rain'
        ]
        
        self.heat_keywords = [
            'heat', 'hot', 'heatwave', 'temperature', 'scorching',
            'sweltering', 'humid', 'humidity', 'heat index',
            'extreme heat', 'heat stroke', 'dehydration',
            'high temperature', 'heat advisory', 'heat warning'
        ]
        
        self.risk_modifiers = [
            'warning', 'alert', 'danger', 'severe', 'extreme',
            'critical', 'emergency', 'urgent', 'high risk',
            'catastrophic', 'devastating', 'threatening'
        ]
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'compound': 0.0
            }
        
        scores = self.vader.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg'],
            'compound': scores['compound']
        }
    
    def extract_keywords(self, text: str, risk_type: str = 'flood') -> List[str]:
        """Extract relevant keywords from text"""
        if not text:
            return []
        
        text_lower = text.lower()
        
        if risk_type == 'flood':
            keywords = self.flood_keywords
        else:
            keywords = self.heat_keywords
        
        found_keywords = []
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def calculate_risk_score(self, text: str, risk_type: str = 'flood') -> float:
        """
        Calculate risk score based on keyword frequency and sentiment
        Returns a score between 0 and 1
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        if risk_type == 'flood':
            keywords = self.flood_keywords
        else:
            keywords = self.heat_keywords
        
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        modifier_count = sum(1 for mod in self.risk_modifiers if mod in text_lower)
        
        sentiment = self.analyze_sentiment(text)
        negative_sentiment = abs(min(sentiment['compound'], 0))
        
        keyword_score = min(keyword_count / 5, 1.0) * 0.5
        modifier_score = min(modifier_count / 3, 1.0) * 0.3
        sentiment_score = negative_sentiment * 0.2
        
        total_score = keyword_score + modifier_score + sentiment_score
        
        return min(total_score, 1.0)
    
    def analyze_articles(self, articles: List[Dict], risk_type: str = 'flood') -> pd.DataFrame:
        """Analyze multiple news articles and return structured results"""
        if not articles:
            return pd.DataFrame({
                'title': [],
                'sentiment_compound': [],
                'risk_score': [],
                'keyword_count': [],
                'source': []
            })
        
        results = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            full_text = f"{title} {description} {content}"
            
            sentiment = self.analyze_sentiment(full_text)
            risk_score = self.calculate_risk_score(full_text, risk_type)
            keywords = self.extract_keywords(full_text, risk_type)
            
            results.append({
                'title': title,
                'sentiment_compound': sentiment['compound'],
                'sentiment_negative': sentiment['negative'],
                'sentiment_positive': sentiment['positive'],
                'risk_score': risk_score,
                'keyword_count': len(keywords),
                'keywords': ', '.join(keywords[:5]),
                'source': article.get('source', 'Unknown'),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return pd.DataFrame(results)
    
    def get_aggregate_risk_signal(self, articles: List[Dict], risk_type: str = 'flood') -> Dict[str, float]:
        """Get aggregate risk signal from all articles"""
        if not articles:
            return {
                'avg_risk_score': 0.0,
                'avg_sentiment': 0.0,
                'total_articles': 0,
                'high_risk_articles': 0
            }
        
        df = self.analyze_articles(articles, risk_type)
        
        if len(df) == 0:
            return {
                'avg_risk_score': 0.0,
                'avg_sentiment': 0.0,
                'total_articles': 0,
                'high_risk_articles': 0
            }
        
        return {
            'avg_risk_score': float(df['risk_score'].mean()),
            'avg_sentiment': float(df['sentiment_compound'].mean()),
            'total_articles': len(df),
            'high_risk_articles': int(len(df[df['risk_score'] > 0.6])),
            'keyword_density': float(df['keyword_count'].mean())
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
