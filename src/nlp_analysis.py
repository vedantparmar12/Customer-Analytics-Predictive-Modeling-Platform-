import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import re
from collections import Counter
from .logger import get_logger
from .custom_exception import CustomException
import os
import joblib

logger = get_logger(__name__)

class CustomerReviewAnalyzer:
    def __init__(self, data_path, output_path="artifacts/nlp"):
        self.data_path = data_path
        self.output_path = output_path
        self.reviews_df = None
        self.nlp = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Customer Review Analyzer initialized")
    
    def load_nlp_model(self):
        """Load SpaCy model"""
        try:
            logger.info("Loading SpaCy model...")
            # Try to load the model, if not available, download it
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.info("Downloading SpaCy model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("SpaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SpaCy model: {e}")
            raise CustomException("Failed to load SpaCy model", e)
    
    def load_reviews(self):
        """Load customer reviews data"""
        try:
            logger.info("Loading customer reviews...")
            self.reviews_df = pd.read_csv(os.path.join(self.data_path, "olist_order_reviews_dataset.csv"))
            
            # Remove null reviews
            self.reviews_df = self.reviews_df.dropna(subset=['review_comment_message'])
            
            logger.info(f"Loaded {len(self.reviews_df)} reviews with comments")
        except Exception as e:
            logger.error(f"Error loading reviews: {e}")
            raise CustomException("Failed to load reviews", e)
    
    def analyze_sentiment(self):
        """Analyze sentiment of reviews"""
        try:
            logger.info("Analyzing review sentiment...")
            
            sentiments = []
            polarities = []
            subjectivities = []
            
            for review in self.reviews_df['review_comment_message'].fillna(''):
                if review:
                    blob = TextBlob(str(review))
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Classify sentiment
                    if polarity > 0.1:
                        sentiment = 'positive'
                    elif polarity < -0.1:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    sentiments.append(sentiment)
                    polarities.append(polarity)
                    subjectivities.append(subjectivity)
                else:
                    sentiments.append('neutral')
                    polarities.append(0)
                    subjectivities.append(0)
            
            self.reviews_df['sentiment'] = sentiments
            self.reviews_df['polarity'] = polarities
            self.reviews_df['subjectivity'] = subjectivities
            
            # Aggregate sentiment by rating
            sentiment_by_rating = self.reviews_df.groupby(['review_score', 'sentiment']).size().unstack(fill_value=0)
            sentiment_by_rating.to_csv(os.path.join(self.output_path, 'sentiment_by_rating.csv'))
            
            logger.info("Sentiment analysis completed")
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise CustomException("Failed to analyze sentiment", e)
    
    def extract_topics(self, n_topics=10):
        """Extract common topics from reviews using SpaCy"""
        try:
            logger.info("Extracting topics from reviews...")
            
            # Combine all reviews
            all_reviews = ' '.join(self.reviews_df['review_comment_message'].fillna('').astype(str))
            
            # Process with SpaCy
            doc = self.nlp(all_reviews[:1000000])  # Limit to 1M chars for performance
            
            # Extract entities
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['PRODUCT', 'ORG', 'PERSON', 'FAC']]
            entity_counts = Counter(entities).most_common(20)
            
            # Extract noun phrases
            noun_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Max 3 words
                    noun_phrases.append(chunk.text.lower())
            
            phrase_counts = Counter(noun_phrases).most_common(n_topics)
            
            # Save topic analysis
            topics_df = pd.DataFrame({
                'entities': [e[0] for e in entity_counts[:n_topics]],
                'entity_count': [e[1] for e in entity_counts[:n_topics]],
                'phrases': [p[0] for p in phrase_counts],
                'phrase_count': [p[1] for p in phrase_counts]
            })
            
            topics_df.to_csv(os.path.join(self.output_path, 'review_topics.csv'), index=False)
            
            logger.info(f"Extracted {len(topics_df)} topics")
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            raise CustomException("Failed to extract topics", e)
    
    def analyze_keywords_by_rating(self):
        """Extract keywords for different rating levels"""
        try:
            logger.info("Analyzing keywords by rating...")
            
            keywords_by_rating = {}
            
            for rating in [1, 2, 3, 4, 5]:
                rating_reviews = self.reviews_df[
                    self.reviews_df['review_score'] == rating
                ]['review_comment_message'].fillna('')
                
                if len(rating_reviews) > 0:
                    # Combine reviews for this rating
                    combined_text = ' '.join(rating_reviews.astype(str))[:500000]
                    
                    # Process with SpaCy
                    doc = self.nlp(combined_text)
                    
                    # Extract adjectives and nouns
                    words = []
                    for token in doc:
                        if token.pos_ in ['ADJ', 'NOUN'] and len(token.text) > 3:
                            words.append(token.lemma_.lower())
                    
                    # Get most common words
                    word_counts = Counter(words).most_common(20)
                    keywords_by_rating[rating] = word_counts
            
            # Save keyword analysis
            keywords_df = pd.DataFrame(keywords_by_rating)
            keywords_df.to_csv(os.path.join(self.output_path, 'keywords_by_rating.csv'))
            
            logger.info("Keyword analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing keywords: {e}")
            raise CustomException("Failed to analyze keywords", e)
    
    def create_review_features(self):
        """Create features from reviews for the main customer dataset"""
        try:
            logger.info("Creating review-based features...")
            
            # Aggregate review features by customer
            orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            
            # Merge reviews with orders
            review_orders = self.reviews_df.merge(
                orders_df[['order_id', 'customer_id']], 
                on='order_id'
            )
            
            # Customer-level review features
            customer_review_features = review_orders.groupby('customer_id').agg({
                'review_score': ['mean', 'std', 'min'],
                'sentiment': lambda x: (x == 'positive').mean(),  # Positive sentiment ratio
                'polarity': 'mean',
                'subjectivity': 'mean'
            }).reset_index()
            
            # Flatten columns
            customer_review_features.columns = ['customer_id', 'avg_review_score', 
                                               'review_score_std', 'min_review_score',
                                               'positive_sentiment_ratio', 'avg_polarity',
                                               'avg_subjectivity']
            
            # Save features
            customer_review_features.to_csv(
                os.path.join(self.output_path, 'customer_review_features.csv'), 
                index=False
            )
            
            logger.info(f"Created review features for {len(customer_review_features)} customers")
            
            return customer_review_features
            
        except Exception as e:
            logger.error(f"Error creating review features: {e}")
            raise CustomException("Failed to create review features", e)
    
    def run(self):
        """Run complete NLP analysis pipeline"""
        self.load_nlp_model()
        self.load_reviews()
        self.analyze_sentiment()
        self.extract_topics()
        self.analyze_keywords_by_rating()
        review_features = self.create_review_features()
        
        logger.info("NLP analysis pipeline completed successfully")
        return review_features

if __name__ == "__main__":
    analyzer = CustomerReviewAnalyzer("data")
    analyzer.run()