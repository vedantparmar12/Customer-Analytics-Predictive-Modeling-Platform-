import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
import os
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class RecommendationEngine:
    def __init__(self, data_path, output_path="artifacts/models"):
        self.data_path = data_path
        self.output_path = output_path
        self.orders_df = None
        self.order_items_df = None
        self.products_df = None
        self.customers_df = None
        self.user_item_matrix = None
        self.product_similarity_matrix = None
        self.user_similarity_matrix = None
        self.svd_model = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_list = None
        self.item_list = None
        self.top_items_idx = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Recommendation Engine initialized")
    
    def load_data(self):
        """Load necessary data for recommendations"""
        try:
            logger.info("Loading data for recommendation engine...")
            self.orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            self.order_items_df = pd.read_csv(os.path.join(self.data_path, "olist_order_items_dataset.csv"))
            self.products_df = pd.read_csv(os.path.join(self.data_path, "olist_products_dataset.csv"))
            self.customers_df = pd.read_csv(os.path.join(self.data_path, "olist_customers_dataset.csv"))
            
            # Get unique customers
            self.orders_df = self.orders_df.merge(
                self.customers_df[['customer_id', 'customer_unique_id']], 
                on='customer_id'
            )
            
            logger.info("Data loaded successfully for recommendations")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data for recommendations", e)
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix using sparse matrices"""
        try:
            logger.info("Creating user-item matrix...")
            
            # Merge order data with items
            order_products = self.orders_df[['order_id', 'customer_unique_id']].merge(
                self.order_items_df[['order_id', 'product_id', 'price']], 
                on='order_id'
            )
            
            # Sample data if too large (for memory efficiency)
            unique_users = order_products['customer_unique_id'].nunique()
            unique_products = order_products['product_id'].nunique()
            logger.info(f"Total users: {unique_users}, Total products: {unique_products}")
            
            # If matrix would be too large, sample the data
            if unique_users * unique_products > 100_000_000:  # 100M cells threshold
                logger.info("Dataset too large, sampling users and products...")
                # Keep most active users
                top_users = order_products['customer_unique_id'].value_counts().head(20000).index
                # Keep most popular products
                top_products = order_products['product_id'].value_counts().head(5000).index
                
                order_products = order_products[
                    order_products['customer_unique_id'].isin(top_users) & 
                    order_products['product_id'].isin(top_products)
                ]
                logger.info(f"Sampled to {order_products['customer_unique_id'].nunique()} users and {order_products['product_id'].nunique()} products")
            
            # Create implicit ratings based on purchase frequency and price
            user_product_interactions = order_products.groupby(['customer_unique_id', 'product_id']).agg({
                'order_id': 'count',  # Purchase frequency
                'price': 'mean'       # Average price paid
            }).reset_index()
            
            user_product_interactions.columns = ['customer_unique_id', 'product_id', 'purchase_count', 'avg_price']
            
            # Create implicit rating (combination of frequency and normalized price)
            price_norm = user_product_interactions['avg_price'] / user_product_interactions['avg_price'].max()
            user_product_interactions['rating'] = (
                user_product_interactions['purchase_count'] * 0.7 + 
                price_norm * 0.3 * 5  # Scale to 0-5 range
            )
            
            # Create user and item mappings
            unique_users = user_product_interactions['customer_unique_id'].unique()
            unique_items = user_product_interactions['product_id'].unique()
            
            self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
            
            # Create sparse matrix directly
            row_indices = [self.user_mapping[user] for user in user_product_interactions['customer_unique_id']]
            col_indices = [self.item_mapping[item] for item in user_product_interactions['product_id']]
            data = user_product_interactions['rating'].values
            
            self.user_item_matrix = csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(len(unique_users), len(unique_items))
            )
            
            logger.info(f"Created sparse user-item matrix: {self.user_item_matrix.shape}")
            logger.info(f"Matrix density: {self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4%}")
            
            # Save mappings and user/item lists for later use
            self.user_list = list(unique_users)
            self.item_list = list(unique_items)
            
            joblib.dump(self.user_mapping, os.path.join(self.output_path, "user_mapping.pkl"))
            joblib.dump(self.item_mapping, os.path.join(self.output_path, "item_mapping.pkl"))
            joblib.dump(self.user_list, os.path.join(self.output_path, "user_list.pkl"))
            joblib.dump(self.item_list, os.path.join(self.output_path, "item_list.pkl"))
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {e}")
            raise CustomException("Failed to create user-item matrix", e)
    
    def build_collaborative_filtering(self):
        """Build collaborative filtering models"""
        try:
            logger.info("Building collaborative filtering models...")
            
            # Use the sparse matrix directly (it's already sparse)
            user_item_sparse = self.user_item_matrix
            
            # For large matrices, compute similarity only for a subset
            n_items = user_item_sparse.shape[1]
            n_users = user_item_sparse.shape[0]
            
            # Item-based collaborative filtering
            if n_items > 10000:
                logger.info(f"Large item set ({n_items} items), computing similarity for top items only...")
                # Compute item popularity
                item_popularity = np.array(user_item_sparse.sum(axis=0)).flatten()
                top_items_idx = np.argsort(item_popularity)[-5000:]  # Top 5000 items
                
                # Compute similarity only for top items
                item_subset = user_item_sparse[:, top_items_idx]
                self.product_similarity_matrix = cosine_similarity(item_subset.T)
                self.top_items_idx = top_items_idx
            else:
                logger.info("Computing item similarity matrix...")
                self.product_similarity_matrix = cosine_similarity(user_item_sparse.T)
                self.top_items_idx = None
            
            # User-based collaborative filtering
            if n_users > 20000:
                logger.info(f"Large user set ({n_users} users), skipping user similarity matrix...")
                self.user_similarity_matrix = None  # Skip for memory efficiency
            else:
                logger.info("Computing user similarity matrix...")
                self.user_similarity_matrix = cosine_similarity(user_item_sparse)
            
            # Matrix factorization using SVD
            logger.info("Training SVD model...")
            n_components = min(50, min(n_users, n_items) - 1)  # Ensure n_components is valid
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = self.svd_model.fit_transform(user_item_sparse)
            item_factors = self.svd_model.components_.T
            
            # Save models
            joblib.dump(self.product_similarity_matrix, 
                       os.path.join(self.output_path, "product_similarity_matrix.pkl"))
            if self.user_similarity_matrix is not None:
                joblib.dump(self.user_similarity_matrix, 
                           os.path.join(self.output_path, "user_similarity_matrix.pkl"))
            joblib.dump(self.svd_model, 
                       os.path.join(self.output_path, "svd_model.pkl"))
            joblib.dump(user_factors, 
                       os.path.join(self.output_path, "user_factors.pkl"))
            joblib.dump(item_factors, 
                       os.path.join(self.output_path, "item_factors.pkl"))
            if self.top_items_idx is not None:
                joblib.dump(self.top_items_idx, 
                           os.path.join(self.output_path, "top_items_idx.pkl"))
            
            logger.info("Collaborative filtering models built successfully")
            
        except Exception as e:
            logger.error(f"Error building collaborative filtering: {e}")
            raise CustomException("Failed to build collaborative filtering", e)
    
    def get_item_recommendations(self, product_id, n_recommendations=10):
        """Get similar product recommendations"""
        try:
            if product_id not in self.item_mapping:
                logger.warning(f"Product {product_id} not found in training data")
                return []
            
            # Get product index
            product_idx = self.item_mapping[product_id]
            
            # Handle case where we only computed similarity for top items
            if hasattr(self, 'top_items_idx') and self.top_items_idx is not None:
                if product_idx not in self.top_items_idx:
                    logger.warning(f"Product {product_id} not in top items, returning popular products")
                    return self.get_popular_products(n_recommendations)
                
                # Map to reduced index
                reduced_idx = np.where(self.top_items_idx == product_idx)[0][0]
                sim_scores = self.product_similarity_matrix[reduced_idx]
                
                # Get top N similar products
                similar_indices = np.argsort(sim_scores)[::-1][1:n_recommendations+1]
                similar_products = [self.item_list[self.top_items_idx[i]] for i in similar_indices]
            else:
                # Get similarity scores
                sim_scores = self.product_similarity_matrix[product_idx]
                
                # Get top N similar products
                similar_indices = np.argsort(sim_scores)[::-1][1:n_recommendations+1]
                similar_products = [self.item_list[i] for i in similar_indices]
            
            # Get product details
            recommendations = []
            for prod_id in similar_products:
                if prod_id in self.products_df['product_id'].values:
                    prod_info = self.products_df[self.products_df['product_id'] == prod_id].iloc[0]
                    recommendations.append({
                        'product_id': prod_id,
                        'category': prod_info.get('product_category_name', 'Unknown'),
                        'similarity_score': float(sim_scores[similar_indices[len(recommendations)]])
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting item recommendations: {e}")
            raise CustomException("Failed to get item recommendations", e)
    
    def get_user_recommendations(self, customer_id, n_recommendations=10):
        """Get personalized recommendations for a user"""
        try:
            if customer_id not in self.user_mapping:
                logger.warning(f"Customer {customer_id} not found in training data")
                return self.get_popular_products(n_recommendations)
            
            # Get user index
            user_idx = self.user_mapping[customer_id]
            
            # Get user's purchase history (sparse matrix row)
            user_purchases = self.user_item_matrix.getrow(user_idx).toarray().flatten()
            purchased_items_idx = np.where(user_purchases > 0)[0]
            purchased_items = [self.item_list[idx] for idx in purchased_items_idx]
            
            # If user similarity matrix exists, use collaborative filtering
            if hasattr(self, 'user_similarity_matrix') and self.user_similarity_matrix is not None:
                # User-based collaborative filtering
                user_similarities = self.user_similarity_matrix[user_idx]
                similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
                
                # Aggregate ratings from similar users
                recommendations_scores = {}
                for similar_user_idx in similar_users:
                    similarity_weight = user_similarities[similar_user_idx]
                    similar_user_ratings = self.user_item_matrix.getrow(similar_user_idx).toarray().flatten()
                    
                    for item_idx in np.where(similar_user_ratings > 0)[0]:
                        product_id = self.item_list[item_idx]
                        if product_id not in purchased_items:
                            if product_id not in recommendations_scores:
                                recommendations_scores[product_id] = 0
                            recommendations_scores[product_id] += similar_user_ratings[item_idx] * similarity_weight
            else:
                # Use matrix factorization (SVD) for recommendations
                logger.info("Using SVD-based recommendations...")
                # Load user factors
                user_factors = joblib.load(os.path.join(self.output_path, "user_factors.pkl"))
                item_factors = joblib.load(os.path.join(self.output_path, "item_factors.pkl"))
                
                # Compute scores for all items
                user_vector = user_factors[user_idx]
                scores = np.dot(item_factors, user_vector)
                
                # Create recommendations excluding purchased items
                recommendations_scores = {}
                for item_idx, score in enumerate(scores):
                    product_id = self.item_list[item_idx]
                    if product_id not in purchased_items:
                        recommendations_scores[product_id] = score
            
            # Sort and get top recommendations
            sorted_recommendations = sorted(recommendations_scores.items(), 
                                         key=lambda x: x[1], reverse=True)[:n_recommendations]
            
            # Format recommendations
            recommendations = []
            for prod_id, score in sorted_recommendations:
                if prod_id in self.products_df['product_id'].values:
                    prod_info = self.products_df[self.products_df['product_id'] == prod_id].iloc[0]
                    recommendations.append({
                        'product_id': prod_id,
                        'category': prod_info.get('product_category_name', 'Unknown'),
                        'recommendation_score': float(score),
                        'method': 'collaborative_filtering' if hasattr(self, 'user_similarity_matrix') else 'svd'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {e}")
            raise CustomException("Failed to get user recommendations", e)
    
    def get_popular_products(self, n_recommendations=10):
        """Get popular products for cold start users"""
        try:
            # Calculate product popularity
            product_popularity = self.order_items_df.groupby('product_id').agg({
                'order_id': 'count',
                'price': 'mean'
            }).reset_index()
            
            product_popularity.columns = ['product_id', 'order_count', 'avg_price']
            product_popularity['popularity_score'] = (
                product_popularity['order_count'] / product_popularity['order_count'].max()
            )
            
            # Get top products
            top_products = product_popularity.nlargest(n_recommendations, 'popularity_score')
            
            # Format recommendations
            recommendations = []
            for _, row in top_products.iterrows():
                if row['product_id'] in self.products_df['product_id'].values:
                    prod_info = self.products_df[self.products_df['product_id'] == row['product_id']].iloc[0]
                    recommendations.append({
                        'product_id': row['product_id'],
                        'category': prod_info.get('product_category_name', 'Unknown'),
                        'popularity_score': row['popularity_score'],
                        'order_count': int(row['order_count']),
                        'method': 'popularity_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            raise CustomException("Failed to get popular products", e)
    
    def evaluate_recommendations(self):
        """Evaluate recommendation system performance"""
        try:
            logger.info("Evaluating recommendation system...")
            
            # Split data for evaluation
            n_users = self.user_item_matrix.shape[0]
            test_user_indices = np.random.choice(n_users, 
                                               size=min(1000, n_users // 10), 
                                               replace=False)
            
            hits = 0
            total_recommendations = 0
            
            for user_idx in test_user_indices:
                user = self.user_list[user_idx]
                
                # Get user's actual purchases from sparse matrix
                user_purchases = self.user_item_matrix.getrow(user_idx).toarray().flatten()
                purchased_items_idx = np.where(user_purchases > 0)[0]
                purchased_items = [self.item_list[idx] for idx in purchased_items_idx]
                
                if len(purchased_items) > 5:
                    # Hide some items for testing
                    test_items = np.random.choice(purchased_items, 
                                                size=min(3, len(purchased_items) // 2), 
                                                replace=False)
                    
                    # Get recommendations
                    recommendations = self.get_user_recommendations(user, n_recommendations=10)
                    recommended_ids = [rec['product_id'] for rec in recommendations]
                    
                    # Check hits
                    for test_item in test_items:
                        if test_item in recommended_ids:
                            hits += 1
                    total_recommendations += len(test_items)
            
            # Calculate metrics
            if total_recommendations > 0:
                hit_rate = hits / total_recommendations
                logger.info(f"Recommendation Hit Rate: {hit_rate:.2%}")
                
                # Save evaluation results
                with open(os.path.join(self.output_path, "recommendation_evaluation.txt"), 'w') as f:
                    f.write(f"Recommendation System Evaluation\n")
                    f.write(f"================================\n")
                    f.write(f"Test Users: {len(test_users)}\n")
                    f.write(f"Total Recommendations Tested: {total_recommendations}\n")
                    f.write(f"Hits: {hits}\n")
                    f.write(f"Hit Rate: {hit_rate:.2%}\n")
            
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {e}")
            raise CustomException("Failed to evaluate recommendations", e)
    
    def run(self):
        """Run complete recommendation engine pipeline"""
        self.load_data()
        self.create_user_item_matrix()
        self.build_collaborative_filtering()
        self.evaluate_recommendations()
        
        # Save the user-item matrix for later use
        self.user_item_matrix.to_csv(
            os.path.join(self.output_path, "user_item_matrix.csv")
        )
        
        logger.info("Recommendation engine pipeline completed successfully")

if __name__ == "__main__":
    engine = RecommendationEngine("data")
    engine.run()