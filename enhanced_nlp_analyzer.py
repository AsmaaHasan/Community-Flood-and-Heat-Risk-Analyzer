"""
Enhanced NLP Analyzer with TF-IDF classification and advanced text analysis
Extends the basic VADER sentiment analysis with machine learning-based classification
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as RFClassifier
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from datetime import datetime, timedelta

class EnhancedNewsTextAnalyzer:
    """
    Advanced NLP analyzer combining VADER sentiment with TF-IDF classification,
    named entity recognition, and urgency detection
    """
    
    def __init__(self):
        self._download_nltk_data()
        self.vader = SentimentIntensityAnalyzer()
        
        # Enhanced keyword sets with weights
        self.flood_keywords = {
            # High severity (weight 1.0)
            'catastrophic flood': 1.0, 'devastating flood': 1.0, 'flash flood': 1.0,
            'flood emergency': 1.0, 'severe flooding': 1.0,
            # Medium severity (weight 0.7)
            'flood': 0.7, 'flooding': 0.7, 'inundation': 0.7, 'overflow': 0.7,
            'deluge': 0.7, 'waterlogged': 0.7, 'submerged': 0.7,
            # Low severity (weight 0.4)
            'rainfall': 0.4, 'monsoon': 0.4, 'storm': 0.4, 'typhoon': 0.4,
            'downpour': 0.4, 'heavy rain': 0.4, 'rising water': 0.4,
            'water level': 0.3, 'drainage': 0.3
        }
        
        self.heat_keywords = {
            # High severity (weight 1.0)
            'heat emergency': 1.0, 'extreme heatwave': 1.0, 'deadly heat': 1.0,
            'heat stroke outbreak': 1.0,
            # Medium severity (weight 0.7)
            'heatwave': 0.7, 'heat wave': 0.7, 'scorching': 0.7, 'sweltering': 0.7,
            'extreme heat': 0.7, 'heat advisory': 0.7, 'heat warning': 0.7,
            # Low severity (weight 0.4)
            'heat': 0.4, 'hot': 0.3, 'temperature': 0.3, 'humid': 0.3,
            'humidity': 0.3, 'heat index': 0.4, 'high temperature': 0.4
        }
        
        self.urgency_indicators = {
            'immediate': 1.0, 'urgent': 0.9, 'emergency': 1.0, 'now': 0.8,
            'imminent': 0.9, 'critical': 0.9, 'severe': 0.8, 'extreme': 0.9,
            'warning': 0.7, 'alert': 0.7, 'breaking': 0.8, 'evacuate': 1.0,
            'evacuation': 1.0, 'shelter': 0.7, 'danger': 0.8, 'threatening': 0.7
        }
        
        self.temporal_patterns = [
            r'within\s+(\d+)\s+(hour|minute)',
            r'in\s+the\s+next\s+(\d+)\s+(hour|day)',
            r'expected\s+(today|tonight|tomorrow)',
            r'currently',
            r'right now',
            r'at this moment'
        ]
        
        # TF-IDF vectorizer for text classification
        self.tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )
        
        # Classifiers (will be trained on synthetic data)
        self.flood_classifier = MultinomialNB()
        self.heat_classifier = MultinomialNB()
        self.risk_level_classifier = RFClassifier(n_estimators=50, random_state=42)
        
        self.is_trained = False
        self._train_classifiers()
    
    def _download_nltk_data(self):
        """Download required NLTK data including NER tags"""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('corpora/words', 'words')
        ]
        
        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                except:
                    pass  # Continue if download fails
    
    def _train_classifiers(self):
        """Train TF-IDF classifiers on synthetic training data"""
        # Generate synthetic training samples
        flood_samples = [
            "Heavy rainfall causes severe flooding in downtown area",
            "Flash flood warning issued for low-lying areas near river",
            "Streets submerged as water levels rise rapidly",
            "Emergency evacuation ordered due to catastrophic flooding",
            "Monsoon rains trigger widespread inundation across region",
            "Drainage system overwhelmed by intense downpour",
            "Residents trapped by rising floodwaters seek shelter",
            "Storm surge brings devastating flood damage to coastal communities",
            "Overflowing rivers threaten homes in flood-prone zones",
            "Waterlogged streets make travel dangerous after heavy rains"
        ]
        
        heat_samples = [
            "Extreme heatwave grips city with record temperatures",
            "Heat advisory issued as temperatures soar above 40 degrees",
            "Elderly vulnerable to heat stroke in scorching conditions",
            "Sweltering humidity makes heat index unbearable",
            "Heat warning in effect with dangerous temperature levels",
            "Emergency cooling centers open amid deadly heat",
            "High temperatures create health crisis in urban areas",
            "Heatwave continues with no relief in sight",
            "Record-breaking heat affects thousands of residents",
            "Extreme weather brings dangerous heat conditions"
        ]
        
        neutral_samples = [
            "Weather forecast shows partly cloudy skies",
            "Temperature remains normal for this time of year",
            "Mild weather conditions expected throughout the week",
            "Seasonal patterns continue without significant changes",
            "Comfortable weather brings people outdoors",
            "Clear skies and moderate temperatures forecasted",
            "Weather remains stable with no alerts issued",
            "Typical spring conditions across the region",
            "Pleasant weather continues into the weekend",
            "No significant weather events expected"
        ]
        
        # Create training data
        all_samples = flood_samples + heat_samples + neutral_samples
        flood_labels = [1] * len(flood_samples) + [0] * (len(heat_samples) + len(neutral_samples))
        heat_labels = [0] * len(flood_samples) + [1] * len(heat_samples) + [0] * len(neutral_samples)
        
        # Risk levels: 0=Low, 1=Medium, 2=High
        risk_labels = (
            [2] * 5 + [1] * 5 +  # Flood: 5 high, 5 medium
            [2] * 5 + [1] * 5 +  # Heat: 5 high, 5 medium
            [0] * len(neutral_samples)  # Neutral: all low
        )
        
        try:
            # Fit TF-IDF vectorizer
            X = self.tfidf.fit_transform(all_samples)
            
            # Train classifiers
            self.flood_classifier.fit(X, flood_labels)
            self.heat_classifier.fit(X, heat_labels)
            self.risk_level_classifier.fit(X.toarray(), risk_labels)
            
            self.is_trained = True
        except Exception as e:
            print(f"Warning: Classifier training failed: {e}")
            self.is_trained = False
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (locations, organizations) from text"""
        if not text:
            return {'locations': [], 'organizations': [], 'persons': []}
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags, binary=False)
            
            locations = []
            organizations = []
            persons = []
            
            for subtree in named_entities:
                if hasattr(subtree, 'label'):
                    entity_text = ' '.join(word for word, tag in subtree.leaves())
                    if subtree.label() == 'GPE':  # Geo-political entity (location)
                        locations.append(entity_text)
                    elif subtree.label() == 'ORGANIZATION':
                        organizations.append(entity_text)
                    elif subtree.label() == 'PERSON':
                        persons.append(entity_text)
            
            return {
                'locations': list(set(locations)),
                'organizations': list(set(organizations)),
                'persons': list(set(persons))
            }
        except Exception as e:
            return {'locations': [], 'organizations': [], 'persons': []}
    
    def detect_urgency(self, text: str) -> Dict[str, float]:
        """Detect urgency level and temporal immediacy in text"""
        if not text:
            return {'urgency_score': 0.0, 'has_temporal_pattern': False}
        
        text_lower = text.lower()
        
        # Calculate urgency score from keywords
        urgency_score = 0.0
        urgency_count = 0
        
        for indicator, weight in self.urgency_indicators.items():
            if indicator in text_lower:
                urgency_score += weight
                urgency_count += 1
        
        # Normalize score
        if urgency_count > 0:
            urgency_score = min(urgency_score / urgency_count, 1.0)
        
        # Check for temporal patterns indicating immediacy
        has_temporal = False
        for pattern in self.temporal_patterns:
            if re.search(pattern, text_lower):
                has_temporal = True
                urgency_score = min(urgency_score + 0.2, 1.0)
                break
        
        return {
            'urgency_score': urgency_score,
            'has_temporal_pattern': has_temporal,
            'urgency_indicators_found': urgency_count
        }
    
    def extract_weighted_keywords(self, text: str, risk_type: str = 'flood') -> Dict[str, float]:
        """Extract keywords with their importance weights"""
        if not text:
            return {}
        
        text_lower = text.lower()
        keywords = self.flood_keywords if risk_type == 'flood' else self.heat_keywords
        
        found_keywords = {}
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                # Count occurrences and multiply by weight
                count = text_lower.count(keyword)
                found_keywords[keyword] = weight * count
        
        return found_keywords
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Use TF-IDF classifiers to determine flood/heat relevance and risk level"""
        if not text or not self.is_trained:
            return {
                'is_flood_related': False,
                'is_heat_related': False,
                'predicted_risk_level': 'Low',
                'flood_probability': 0.0,
                'heat_probability': 0.0
            }
        
        try:
            # Transform text to TF-IDF features
            X = self.tfidf.transform([text])
            
            # Get predictions
            flood_prob = self.flood_classifier.predict_proba(X)[0][1]
            heat_prob = self.heat_classifier.predict_proba(X)[0][1]
            risk_level = self.risk_level_classifier.predict(X.toarray())[0]
            
            risk_level_names = ['Low', 'Medium', 'High']
            
            return {
                'is_flood_related': flood_prob > 0.5,
                'is_heat_related': heat_prob > 0.5,
                'predicted_risk_level': risk_level_names[risk_level],
                'flood_probability': float(flood_prob),
                'heat_probability': float(heat_prob),
                'ml_confidence': max(flood_prob, heat_prob)
            }
        except Exception as e:
            return {
                'is_flood_related': False,
                'is_heat_related': False,
                'predicted_risk_level': 'Low',
                'flood_probability': 0.0,
                'heat_probability': 0.0
            }
    
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
    
    def calculate_enhanced_risk_score(self, text: str, risk_type: str = 'flood') -> Dict[str, float]:
        """
        Calculate comprehensive risk score combining multiple NLP techniques
        Returns detailed breakdown of risk components
        """
        if not text:
            return {
                'total_risk_score': 0.0,
                'keyword_score': 0.0,
                'urgency_score': 0.0,
                'sentiment_score': 0.0,
                'ml_score': 0.0
            }
        
        # 1. Weighted keyword analysis (30% weight)
        keywords = self.extract_weighted_keywords(text, risk_type)
        keyword_score = min(sum(keywords.values()) / 3.0, 1.0) * 0.30
        
        # 2. Urgency detection (25% weight)
        urgency = self.detect_urgency(text)
        urgency_score = urgency['urgency_score'] * 0.25
        
        # 3. Sentiment analysis (20% weight)
        sentiment = self.analyze_sentiment(text)
        negative_sentiment = abs(min(sentiment['compound'], 0))
        sentiment_score = negative_sentiment * 0.20
        
        # 4. ML classification (25% weight)
        ml_classification = self.classify_text(text)
        if risk_type == 'flood':
            ml_score = ml_classification['flood_probability'] * 0.25
        else:
            ml_score = ml_classification['heat_probability'] * 0.25
        
        total_score = keyword_score + urgency_score + sentiment_score + ml_score
        
        return {
            'total_risk_score': min(total_score, 1.0),
            'keyword_score': keyword_score / 0.30,  # Normalized to 0-1
            'urgency_score': urgency_score / 0.25,
            'sentiment_score': sentiment_score / 0.20,
            'ml_score': ml_score / 0.25,
            'ml_predicted_level': ml_classification.get('predicted_risk_level', 'Unknown')
        }
    
    def analyze_articles_enhanced(self, articles: List[Dict], risk_type: str = 'flood') -> pd.DataFrame:
        """Enhanced article analysis with all NLP features"""
        if not articles:
            return pd.DataFrame()
        
        results = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            full_text = f"{title} {description} {content}"
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(full_text)
            
            # Enhanced risk scoring
            risk_breakdown = self.calculate_enhanced_risk_score(full_text, risk_type)
            
            # Named entity extraction
            entities = self.extract_named_entities(full_text)
            
            # Urgency detection
            urgency = self.detect_urgency(full_text)
            
            # ML classification
            ml_class = self.classify_text(full_text)
            
            # Weighted keywords
            keywords = self.extract_weighted_keywords(full_text, risk_type)
            top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results.append({
                'title': title,
                'sentiment_compound': sentiment['compound'],
                'sentiment_negative': sentiment['negative'],
                'total_risk_score': risk_breakdown['total_risk_score'],
                'keyword_score': risk_breakdown['keyword_score'],
                'urgency_score': urgency['urgency_score'],
                'ml_probability': ml_class['flood_probability'] if risk_type == 'flood' else ml_class['heat_probability'],
                'ml_predicted_level': ml_class['predicted_risk_level'],
                'has_temporal_urgency': urgency['has_temporal_pattern'],
                'locations_mentioned': ', '.join(entities['locations'][:3]),
                'top_keywords': ', '.join([k for k, v in top_keywords]),
                'source': article.get('source', 'Unknown'),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return pd.DataFrame(results)
    
    def get_aggregate_risk_signal_enhanced(self, articles: List[Dict], risk_type: str = 'flood') -> Dict[str, float]:
        """Get enhanced aggregate risk signal from all articles"""
        if not articles:
            return {
                'avg_risk_score': 0.0,
                'avg_urgency': 0.0,
                'avg_ml_probability': 0.0,
                'total_articles': 0,
                'high_risk_articles': 0,
                'urgent_articles': 0,
                'ml_confidence': 0.0
            }
        
        df = self.analyze_articles_enhanced(articles, risk_type)
        
        if len(df) == 0:
            return {
                'avg_risk_score': 0.0,
                'avg_urgency': 0.0,
                'avg_ml_probability': 0.0,
                'total_articles': 0,
                'high_risk_articles': 0,
                'urgent_articles': 0,
                'ml_confidence': 0.0
            }
        
        return {
            'avg_risk_score': float(df['total_risk_score'].mean()),
            'avg_urgency': float(df['urgency_score'].mean()),
            'avg_ml_probability': float(df['ml_probability'].mean()),
            'total_articles': len(df),
            'high_risk_articles': int(len(df[df['total_risk_score'] > 0.6])),
            'urgent_articles': int(len(df[df['urgency_score'] > 0.7])),
            'ml_confidence': float(df['ml_probability'].mean()),
            'avg_sentiment': float(df['sentiment_compound'].mean())
        }
