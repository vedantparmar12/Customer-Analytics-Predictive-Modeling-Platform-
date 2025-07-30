import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, KNNWithMeans
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import accuracy
import networkx as nx
from pyvis.network import Network
import joblib
import os
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class AdvancedRecommendationEngine:
    def __init__(self, data_path, output_path="artifacts/models"):
        self.data_path = data_path
        self.output_path = output_path
        self.orders_df = None
        self.order_items_df = None
        self.products_df = None
        self.ratings_data = None
        self.best_model = None
        self.model_performances = {}
        
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "network_viz"), exist_ok=True)
        
        logger.info("Advanced Recommendation Engine initialized")
    
    def load_and_prepare_data(self):
        """Load data and create ratings dataset"""
        try:
            logger.info("Loading data for recommendations...")
            
            # Load datasets
            self.orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            self.order_items_df = pd.read_csv(os.path.join(self.data_path, "olist_order_items_dataset.csv"))
            self.products_df = pd.read_csv(os.path.join(self.data_path, "olist_products_dataset.csv"))
            customers_df = pd.read_csv(os.path.join(self.data_path, "olist_customers_dataset.csv"))
            reviews_df = pd.read_csv(os.path.join(self.data_path, "olist_order_reviews_dataset.csv"))
            
            # Merge to get customer-product interactions with ratings
            order_products = self.orders_df.merge(
                customers_df[['customer_id', 'customer_unique_id']], 
                on='customer_id'
            ).merge(
                self.order_items_df[['order_id', 'product_id', 'price']], 
                on='order_id'
            ).merge(
                reviews_df[['order_id', 'review_score']], 
                on='order_id', 
                how='left'
            )
            
            # Create implicit ratings for orders without reviews
            order_products['rating'] = order_products['review_score'].fillna(3.5)
            
            # Aggregate ratings by customer-product pairs
            self.ratings_data = order_products.groupby(
                ['customer_unique_id', 'product_id']
            )['rating'].mean().reset_index()
            
            # Filter users and items with minimum interactions
            user_counts = self.ratings_data['customer_unique_id'].value_counts()
            item_counts = self.ratings_data['product_id'].value_counts()
            
            # Keep users with at least 5 ratings and items with at least 10 ratings
            # This ensures more robust patterns
            active_users = user_counts[user_counts >= 5].index
            popular_items = item_counts[item_counts >= 10].index
            
            self.ratings_data = self.ratings_data[
                (self.ratings_data['customer_unique_id'].isin(active_users)) &
                (self.ratings_data['product_id'].isin(popular_items))
            ]
            
            # Add slight noise to ratings to prevent perfect memorization
            np.random.seed(42)
            noise = np.random.normal(0, 0.1, len(self.ratings_data))
            self.ratings_data['rating'] = np.clip(
                self.ratings_data['rating'] + noise, 1, 5
            )
            
            logger.info(f"Prepared {len(self.ratings_data)} ratings from {len(self.ratings_data['customer_unique_id'].unique())} users")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load recommendation data", e)
    
    def train_multiple_algorithms(self):
        """Train multiple recommendation algorithms"""
        try:
            logger.info("Training multiple recommendation algorithms...")
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.ratings_data[['customer_unique_id', 'product_id', 'rating']], 
                reader
            )
            
            # Define algorithms with regularization
            algorithms = {
                'SVD': SVD(
                    n_factors=50,  # Reduced factors
                    n_epochs=20, 
                    random_state=42,
                    lr_all=0.005,  # Lower learning rate
                    reg_all=0.1  # Higher regularization
                ),
                'SVD_tuned': SVD(),  # Will be tuned
                'NMF': NMF(
                    n_factors=30,  # Reduced factors
                    n_epochs=20, 
                    random_state=42,
                    reg_pu=0.1,  # User regularization
                    reg_qi=0.1  # Item regularization
                ),
                'KNN_Basic': KNNBasic(
                    k=20,  # Reduced neighbors
                    min_k=5,  # Minimum neighbors
                    sim_options={'name': 'cosine', 'shrinkage': 100}  # Shrinkage for regularization
                ),
                'KNN_Means': KNNWithMeans(
                    k=20,  # Reduced neighbors
                    min_k=5,  # Minimum neighbors
                    sim_options={'name': 'pearson', 'shrinkage': 100}  # Shrinkage for regularization
                )
            }
            
            # Train and evaluate each algorithm
            for name, algo in algorithms.items():
                logger.info(f"Training {name}...")
                
                if name == 'SVD_tuned':
                    # Hyperparameter tuning for SVD with regularization focus
                    param_grid = {
                        'n_factors': [30, 50, 70],  # Fewer factors
                        'n_epochs': [15, 20],  # Fewer epochs
                        'lr_all': [0.002, 0.005],  # Lower learning rates
                        'reg_all': [0.1, 0.2, 0.3]  # Much higher regularization
                    }
                    
                    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
                    gs.fit(data)
                    
                    algo = gs.best_estimator['rmse']
                    logger.info(f"Best parameters for SVD: {gs.best_params['rmse']}")
                
                # Cross-validation
                cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
                
                self.model_performances[name] = {
                    'rmse': np.mean(cv_results['test_rmse']),
                    'mae': np.mean(cv_results['test_mae']),
                    'fit_time': np.mean(cv_results['fit_time']),
                    'test_time': np.mean(cv_results['test_time'])
                }
                
                logger.info(f"{name} - RMSE: {self.model_performances[name]['rmse']:.4f}, MAE: {self.model_performances[name]['mae']:.4f}")
                
                # Train on full dataset for production
                trainset = data.build_full_trainset()
                algo.fit(trainset)
                
                # Save model
                joblib.dump(algo, os.path.join(self.output_path, f"surprise_{name.lower()}_model.pkl"))
            
            # Select best model based on RMSE
            best_model_name = min(self.model_performances, key=lambda x: self.model_performances[x]['rmse'])
            self.best_model = joblib.load(os.path.join(self.output_path, f"surprise_{best_model_name.lower()}_model.pkl"))
            
            logger.info(f"Best model: {best_model_name}")
            
        except Exception as e:
            logger.error(f"Error training algorithms: {e}")
            raise CustomException("Failed to train recommendation algorithms", e)
    
    def create_product_network(self):
        """Create network visualization of product relationships"""
        try:
            logger.info("Creating product network visualization...")
            
            # Create co-purchase network
            co_purchase = self.order_items_df.merge(
                self.order_items_df, 
                on='order_id', 
                suffixes=('_1', '_2')
            )
            
            # Remove self-loops
            co_purchase = co_purchase[co_purchase['product_id_1'] != co_purchase['product_id_2']]
            
            # Count co-occurrences
            edge_weights = co_purchase.groupby(
                ['product_id_1', 'product_id_2']
            ).size().reset_index(name='weight')
            
            # Filter strong connections
            edge_weights = edge_weights[edge_weights['weight'] >= 5]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add edges with weights
            for _, row in edge_weights.iterrows():
                G.add_edge(row['product_id_1'], row['product_id_2'], weight=row['weight'])
            
            # Get product categories
            product_categories = self.products_df.set_index('product_id')['product_category_name'].to_dict()
            
            # Add node attributes
            for node in G.nodes():
                G.nodes[node]['category'] = product_categories.get(node, 'unknown')
                G.nodes[node]['degree'] = G.degree(node)
            
            # Identify communities
            communities = nx.community.louvain_communities(G)
            
            # Create interactive visualization with PyVis
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            
            # Add nodes and edges
            for node in G.nodes():
                net.add_node(
                    node, 
                    label=str(node)[:8], 
                    title=f"Product: {node}\nCategory: {G.nodes[node]['category']}\nConnections: {G.nodes[node]['degree']}",
                    size=min(G.nodes[node]['degree'] * 2, 50)
                )
            
            for edge in G.edges(data=True):
                net.add_edge(edge[0], edge[1], value=edge[2]['weight'])
            
            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
              }
            }
            """)
            
            # Save visualization
            net.save_graph(os.path.join(self.output_path, "network_viz", "product_network.html"))
            
            # Save network statistics
            network_stats = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                'density': nx.density(G),
                'num_communities': len(communities)
            }
            
            pd.DataFrame([network_stats]).to_csv(
                os.path.join(self.output_path, "network_viz", "network_statistics.csv"),
                index=False
            )
            
            logger.info(f"Created product network with {network_stats['num_nodes']} nodes and {network_stats['num_edges']} edges")
            
        except Exception as e:
            logger.error(f"Error creating product network: {e}")
            raise CustomException("Failed to create product network", e)
    
    def create_customer_similarity_network(self):
        """Create customer similarity network based on purchase patterns"""
        try:
            logger.info("Creating customer similarity network...")
            
            # Sample customers for visualization (too many for full network)
            sample_customers = self.ratings_data['customer_unique_id'].value_counts().head(100).index
            
            # Create customer-product matrix
            customer_product_matrix = self.ratings_data[
                self.ratings_data['customer_unique_id'].isin(sample_customers)
            ].pivot_table(
                index='customer_unique_id',
                columns='product_id',
                values='rating',
                fill_value=0
            )
            
            # Calculate customer similarity
            from sklearn.metrics.pairwise import cosine_similarity
            customer_similarity = cosine_similarity(customer_product_matrix)
            
            # Create network
            G = nx.Graph()
            
            # Add nodes (customers)
            for customer in customer_product_matrix.index:
                G.add_node(customer, label=customer[:8])
            
            # Add edges (similarities above threshold)
            threshold = 0.3
            for i, customer1 in enumerate(customer_product_matrix.index):
                for j, customer2 in enumerate(customer_product_matrix.index):
                    if i < j and customer_similarity[i, j] > threshold:
                        G.add_edge(customer1, customer2, weight=customer_similarity[i, j])
            
            # Save as GraphML for further analysis
            nx.write_graphml(G, os.path.join(self.output_path, "network_viz", "customer_similarity_network.graphml"))
            
            logger.info(f"Created customer network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error creating customer network: {e}")
            raise CustomException("Failed to create customer network", e)
    
    def generate_business_insights(self):
        """Generate business insights from recommendation system"""
        try:
            logger.info("Generating business insights...")
            
            # Product affinity analysis
            product_pairs = self.order_items_df.merge(
                self.order_items_df, 
                on='order_id', 
                suffixes=('_a', '_b')
            )
            product_pairs = product_pairs[product_pairs['product_id_a'] < product_pairs['product_id_b']]
            
            # Top product combinations
            top_combinations = product_pairs.groupby(
                ['product_id_a', 'product_id_b']
            ).size().reset_index(name='frequency').nlargest(20, 'frequency')
            
            # Add product categories
            top_combinations = top_combinations.merge(
                self.products_df[['product_id', 'product_category_name']], 
                left_on='product_id_a', 
                right_on='product_id',
                how='left'
            ).merge(
                self.products_df[['product_id', 'product_category_name']], 
                left_on='product_id_b', 
                right_on='product_id',
                how='left',
                suffixes=('_a', '_b')
            )
            
            top_combinations.to_csv(
                os.path.join(self.output_path, "top_product_combinations.csv"),
                index=False
            )
            
            # Cross-category recommendations
            cross_category = top_combinations[
                top_combinations['product_category_name_a'] != top_combinations['product_category_name_b']
            ].head(10)
            
            cross_category.to_csv(
                os.path.join(self.output_path, "cross_category_opportunities.csv"),
                index=False
            )
            
            # Model performance summary
            perf_df = pd.DataFrame(self.model_performances).T
            perf_df.to_csv(
                os.path.join(self.output_path, "recommendation_model_performance.csv")
            )
            
            logger.info("Business insights generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise CustomException("Failed to generate business insights", e)
    
    def run(self):
        """Run complete advanced recommendation pipeline"""
        self.load_and_prepare_data()
        self.train_multiple_algorithms()
        self.create_product_network()
        self.create_customer_similarity_network()
        self.generate_business_insights()
        
        logger.info("Advanced recommendation engine pipeline completed successfully")

if __name__ == "__main__":
    engine = AdvancedRecommendationEngine("data")
    engine.run()