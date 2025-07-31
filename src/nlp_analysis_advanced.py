"""
Advanced NLP Analysis with Embeddings and Corpus
"""
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .logger import get_logger
from .custom_exception import CustomException
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

class AdvancedNLPAnalyzer:
    def __init__(self, data_path, output_path="artifacts/nlp_advanced"):
        self.data_path = data_path
        self.output_path = output_path
        self.reviews_df = None
        self.nlp = None
        self.corpus = None
        self.word2vec_model = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Advanced NLP Analyzer initialized")
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
    
    def load_nlp_models(self):
        """Load SpaCy and prepare for advanced NLP"""
        try:
            logger.info("Loading NLP models...")
            
            # Load SpaCy with more components
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom entity patterns if needed
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['product', 'order', 'delivery', 'arrived', 'received'])
            
            logger.info("NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            raise CustomException("Failed to load NLP models", e)
    
    def build_corpus(self):
        """Build a clean corpus from reviews"""
        try:
            logger.info("Building review corpus...")
            
            # Load reviews
            self.reviews_df = pd.read_csv(os.path.join(self.data_path, "olist_order_reviews_dataset.csv"))
            self.reviews_df = self.reviews_df.dropna(subset=['review_comment_message'])
            
            # Clean and preprocess text
            def clean_text(text):
                if pd.isna(text):
                    return ""
                
                # Convert to lowercase
                text = str(text).lower()
                
                # Remove special characters but keep spaces
                text = re.sub(r'[^a-z0-9\s]', ' ', text)
                
                # Remove extra whitespace
                text = ' '.join(text.split())
                
                return text
            
            self.reviews_df['clean_text'] = self.reviews_df['review_comment_message'].apply(clean_text)
            
            # Tokenize
            def tokenize_text(text):
                tokens = word_tokenize(text)
                # Remove stopwords and short tokens
                tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
                return tokens
            
            self.reviews_df['tokens'] = self.reviews_df['clean_text'].apply(tokenize_text)
            
            # Build corpus (list of tokenized documents)
            self.corpus = self.reviews_df['tokens'].tolist()
            
            logger.info(f"Built corpus with {len(self.corpus)} documents")
            
            # Save preprocessed data
            self.reviews_df[['order_id', 'review_score', 'clean_text']].to_csv(
                os.path.join(self.output_path, 'preprocessed_reviews.csv'), 
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error building corpus: {e}")
            raise CustomException("Failed to build corpus", e)
    
    def create_word_embeddings(self):
        """Create Word2Vec embeddings from the corpus"""
        try:
            logger.info("Creating word embeddings...")
            
            # Detect bigrams and trigrams
            bigram = Phrases(self.corpus, min_count=10, threshold=50)
            trigram = Phrases(bigram[self.corpus], threshold=50)
            
            # Create phraser for faster processing
            bigram_mod = Phraser(bigram)
            trigram_mod = Phraser(trigram)
            
            # Apply to corpus
            corpus_with_phrases = [trigram_mod[bigram_mod[doc]] for doc in self.corpus]
            
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(
                sentences=corpus_with_phrases,
                vector_size=100,  # Embedding dimension
                window=5,         # Context window
                min_count=5,      # Minimum word frequency
                workers=4,        # Parallel processing
                sg=1,            # Skip-gram (1) or CBOW (0)
                epochs=10
            )
            
            # Save the model
            self.word2vec_model.save(os.path.join(self.output_path, 'word2vec_model.bin'))
            
            # Extract vocabulary and embeddings
            vocab = list(self.word2vec_model.wv.index_to_key)
            embeddings = np.array([self.word2vec_model.wv[word] for word in vocab])
            
            logger.info(f"Created embeddings for {len(vocab)} words")
            
            # Save vocabulary and embeddings
            vocab_df = pd.DataFrame({
                'word': vocab,
                'frequency': [self.word2vec_model.wv.get_vecattr(word, 'count') for word in vocab]
            })
            vocab_df.to_csv(os.path.join(self.output_path, 'vocabulary.csv'), index=False)
            
            np.save(os.path.join(self.output_path, 'embeddings.npy'), embeddings)
            
            # Find similar words examples
            similarity_examples = {}
            test_words = ['good', 'bad', 'fast', 'slow', 'quality', 'price', 'delivery']
            
            for word in test_words:
                if word in self.word2vec_model.wv:
                    similar = self.word2vec_model.wv.most_similar(word, topn=5)
                    similarity_examples[word] = similar
            
            # Save similarity examples
            pd.DataFrame(similarity_examples).to_csv(
                os.path.join(self.output_path, 'word_similarities.csv')
            )
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise CustomException("Failed to create embeddings", e)
    
    def extract_entities_advanced(self):
        """Advanced entity extraction with custom patterns"""
        try:
            logger.info("Performing advanced entity extraction...")
            
            # Sample of reviews for entity extraction (limit for performance)
            sample_reviews = self.reviews_df.sample(min(5000, len(self.reviews_df)))
            
            entities_data = []
            
            for idx, row in sample_reviews.iterrows():
                if pd.notna(row['review_comment_message']):
                    doc = self.nlp(str(row['review_comment_message'])[:1000])
                    
                    # Extract standard entities
                    for ent in doc.ents:
                        entities_data.append({
                            'review_id': idx,
                            'entity': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
                    
                    # Extract custom patterns (e.g., product features)
                    # Pattern for size/color mentions
                    size_pattern = r'\b(small|medium|large|xl|xxl|size)\b'
                    color_pattern = r'\b(red|blue|green|black|white|yellow|color)\b'
                    
                    for match in re.finditer(size_pattern, row['clean_text'], re.I):
                        entities_data.append({
                            'review_id': idx,
                            'entity': match.group(),
                            'label': 'SIZE',
                            'start': match.start(),
                            'end': match.end()
                        })
                    
                    for match in re.finditer(color_pattern, row['clean_text'], re.I):
                        entities_data.append({
                            'review_id': idx,
                            'entity': match.group(),
                            'label': 'COLOR',
                            'start': match.start(),
                            'end': match.end()
                        })
            
            # Save entities
            entities_df = pd.DataFrame(entities_data)
            entities_df.to_csv(os.path.join(self.output_path, 'extracted_entities.csv'), index=False)
            
            # Entity statistics
            entity_stats = entities_df.groupby(['label', 'entity']).size().reset_index(name='count')
            entity_stats = entity_stats.sort_values(['label', 'count'], ascending=[True, False])
            entity_stats.to_csv(os.path.join(self.output_path, 'entity_statistics.csv'), index=False)
            
            logger.info(f"Extracted {len(entities_df)} entities")
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            raise CustomException("Failed to extract entities", e)
    
    def topic_modeling_lda(self, n_topics=10):
        """Perform topic modeling using LDA"""
        try:
            logger.info("Performing LDA topic modeling...")
            
            # Create document strings
            documents = [' '.join(tokens) for tokens in self.corpus if len(tokens) > 3]
            
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # LDA Model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method='batch',
                max_iter=10
            )
            
            lda_topics = lda_model.fit_transform(tfidf_matrix)
            
            # Extract top words for each topic
            topics_data = []
            n_top_words = 10
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_word_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                top_scores = [topic[i] for i in top_word_indices]
                
                topics_data.append({
                    'topic_id': topic_idx,
                    'top_words': ', '.join(top_words),
                    'word_scores': top_scores
                })
            
            topics_df = pd.DataFrame(topics_data)
            topics_df.to_csv(os.path.join(self.output_path, 'lda_topics.csv'), index=False)
            
            # Save topic distributions for documents
            np.save(os.path.join(self.output_path, 'document_topics.npy'), lda_topics)
            
            logger.info(f"Extracted {n_topics} topics using LDA")
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            raise CustomException("Failed to perform topic modeling", e)
    
    def sentiment_analysis_advanced(self):
        """Advanced sentiment analysis with aspects"""
        try:
            logger.info("Performing advanced sentiment analysis...")
            
            # Aspect keywords
            aspects = {
                'delivery': ['delivery', 'shipping', 'arrived', 'fast', 'slow', 'delay'],
                'quality': ['quality', 'good', 'bad', 'excellent', 'poor', 'defect'],
                'price': ['price', 'expensive', 'cheap', 'value', 'worth', 'cost'],
                'packaging': ['package', 'packaging', 'box', 'wrapped', 'damaged'],
                'service': ['service', 'support', 'help', 'response', 'customer']
            }
            
            # Initialize aspect sentiments
            for aspect in aspects:
                self.reviews_df[f'{aspect}_mentioned'] = 0
                self.reviews_df[f'{aspect}_sentiment'] = 0
            
            # Analyze each review
            for idx, row in self.reviews_df.iterrows():
                if pd.notna(row['clean_text']):
                    text = row['clean_text']
                    
                    # Check each aspect
                    for aspect, keywords in aspects.items():
                        # Check if aspect is mentioned
                        mentioned = any(keyword in text for keyword in keywords)
                        self.reviews_df.at[idx, f'{aspect}_mentioned'] = int(mentioned)
                        
                        if mentioned:
                            # Extract sentences mentioning the aspect
                            sentences = text.split('.')
                            aspect_sentences = [s for s in sentences 
                                              if any(k in s for k in keywords)]
                            
                            if aspect_sentences:
                                # Get sentiment for aspect sentences
                                aspect_text = ' '.join(aspect_sentences)
                                blob = TextBlob(aspect_text)
                                self.reviews_df.at[idx, f'{aspect}_sentiment'] = blob.sentiment.polarity
            
            # Overall sentiment with TextBlob
            self.reviews_df['overall_polarity'] = self.reviews_df['review_comment_message'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
            )
            
            self.reviews_df['overall_subjectivity'] = self.reviews_df['review_comment_message'].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
            )
            
            # Sentiment categories
            def categorize_sentiment(polarity):
                if polarity >= 0.3:
                    return 'very_positive'
                elif polarity >= 0.1:
                    return 'positive'
                elif polarity >= -0.1:
                    return 'neutral'
                elif polarity >= -0.3:
                    return 'negative'
                else:
                    return 'very_negative'
            
            self.reviews_df['sentiment_category'] = self.reviews_df['overall_polarity'].apply(categorize_sentiment)
            
            # Save aspect sentiment analysis
            aspect_columns = ['order_id', 'review_score', 'overall_polarity', 'sentiment_category']
            for aspect in aspects:
                aspect_columns.extend([f'{aspect}_mentioned', f'{aspect}_sentiment'])
            
            sentiment_df = self.reviews_df[aspect_columns]
            sentiment_df.to_csv(os.path.join(self.output_path, 'aspect_sentiments.csv'), index=False)
            
            # Aggregate aspect sentiments
            aspect_summary = {}
            for aspect in aspects:
                mentioned = self.reviews_df[f'{aspect}_mentioned'].sum()
                if mentioned > 0:
                    avg_sentiment = self.reviews_df[
                        self.reviews_df[f'{aspect}_mentioned'] == 1
                    ][f'{aspect}_sentiment'].mean()
                    aspect_summary[aspect] = {
                        'mentions': mentioned,
                        'avg_sentiment': avg_sentiment
                    }
            
            pd.DataFrame(aspect_summary).T.to_csv(
                os.path.join(self.output_path, 'aspect_summary.csv')
            )
            
            logger.info("Advanced sentiment analysis completed")
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise CustomException("Failed to perform sentiment analysis", e)
    
    def create_review_embeddings(self):
        """Create review-level embeddings using Word2Vec"""
        try:
            logger.info("Creating review embeddings...")
            
            # Method 1: Average word embeddings for each review
            review_embeddings = []
            
            for tokens in self.corpus:
                if len(tokens) > 0:
                    # Get embeddings for words in vocabulary
                    word_vecs = []
                    for token in tokens:
                        if token in self.word2vec_model.wv:
                            word_vecs.append(self.word2vec_model.wv[token])
                    
                    if word_vecs:
                        # Average the word vectors
                        review_embedding = np.mean(word_vecs, axis=0)
                    else:
                        # Zero vector if no words in vocabulary
                        review_embedding = np.zeros(self.word2vec_model.wv.vector_size)
                else:
                    review_embedding = np.zeros(self.word2vec_model.wv.vector_size)
                
                review_embeddings.append(review_embedding)
            
            review_embeddings = np.array(review_embeddings)
            
            # Save review embeddings
            np.save(os.path.join(self.output_path, 'review_embeddings.npy'), review_embeddings)
            
            # Add embedding features to reviews
            self.reviews_df['has_embedding'] = [1 if np.any(emb) else 0 for emb in review_embeddings]
            self.reviews_df['embedding_norm'] = [np.linalg.norm(emb) for emb in review_embeddings]
            
            logger.info(f"Created embeddings for {len(review_embeddings)} reviews")
            
        except Exception as e:
            logger.error(f"Error creating review embeddings: {e}")
            raise CustomException("Failed to create review embeddings", e)
    
    def create_customer_nlp_features(self):
        """Aggregate NLP features at customer level"""
        try:
            logger.info("Creating customer-level NLP features...")
            
            # Load order data to map reviews to customers
            orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            
            # Merge reviews with customer IDs
            review_customers = self.reviews_df.merge(
                orders_df[['order_id', 'customer_id']], 
                on='order_id', 
                how='left'
            )
            
            # Aggregate features by customer
            customer_features = review_customers.groupby('customer_id').agg({
                # Basic review stats
                'review_score': ['mean', 'std', 'min', 'max'],
                'overall_polarity': ['mean', 'std'],
                'overall_subjectivity': 'mean',
                
                # Sentiment distribution
                'sentiment_category': lambda x: (x == 'very_positive').mean(),
                
                # Aspect mentions
                'delivery_mentioned': 'sum',
                'quality_mentioned': 'sum',
                'price_mentioned': 'sum',
                'packaging_mentioned': 'sum',
                'service_mentioned': 'sum',
                
                # Aspect sentiments (average when mentioned)
                'delivery_sentiment': lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0,
                'quality_sentiment': lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0,
                'price_sentiment': lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0,
                
                # Text statistics
                'clean_text': lambda x: np.mean([len(str(t).split()) for t in x]),  # Avg review length
                'has_embedding': 'mean'
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = [
                'customer_id', 
                'avg_review_score', 'review_score_std', 'min_review_score', 'max_review_score',
                'avg_sentiment_polarity', 'sentiment_polarity_std', 'avg_subjectivity',
                'very_positive_ratio',
                'delivery_mentions', 'quality_mentions', 'price_mentions', 
                'packaging_mentions', 'service_mentions',
                'avg_delivery_sentiment', 'avg_quality_sentiment', 'avg_price_sentiment',
                'avg_review_length', 'embedding_coverage'
            ]
            
            # Save customer NLP features
            customer_features.to_csv(
                os.path.join(self.output_path, 'customer_nlp_features.csv'), 
                index=False
            )
            
            logger.info(f"Created NLP features for {len(customer_features)} customers")
            
            return customer_features
            
        except Exception as e:
            logger.error(f"Error creating customer features: {e}")
            raise CustomException("Failed to create customer features", e)
    
    def generate_nlp_insights(self):
        """Generate insights from NLP analysis"""
        try:
            logger.info("Generating NLP insights...")
            
            insights = {
                'total_reviews': len(self.reviews_df),
                'reviews_with_text': len(self.reviews_df[self.reviews_df['clean_text'] != '']),
                'vocabulary_size': len(self.word2vec_model.wv.index_to_key),
                'avg_review_length': self.reviews_df['clean_text'].str.split().str.len().mean(),
                'sentiment_distribution': self.reviews_df['sentiment_category'].value_counts().to_dict(),
                'most_mentioned_aspects': {
                    aspect: self.reviews_df[f'{aspect}_mentioned'].sum() 
                    for aspect in ['delivery', 'quality', 'price', 'packaging', 'service']
                }
            }
            
            # Save insights
            pd.DataFrame([insights]).to_csv(
                os.path.join(self.output_path, 'nlp_insights.csv'), 
                index=False
            )
            
            logger.info("NLP insights generated")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise CustomException("Failed to generate insights", e)
    
    def run(self):
        """Run complete advanced NLP pipeline"""
        self.load_nlp_models()
        self.build_corpus()
        self.create_word_embeddings()
        self.extract_entities_advanced()
        self.topic_modeling_lda()
        self.sentiment_analysis_advanced()
        self.create_review_embeddings()
        customer_features = self.create_customer_nlp_features()
        self.generate_nlp_insights()
        
        logger.info("Advanced NLP analysis pipeline completed successfully")
        return customer_features

if __name__ == "__main__":
    analyzer = AdvancedNLPAnalyzer("data")
    analyzer.run()