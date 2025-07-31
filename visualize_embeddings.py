"""
Visualize and Explore Word Embeddings
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from gensim.models import Word2Vec
import os

def load_embeddings(nlp_path="artifacts/nlp_advanced"):
    """Load saved embeddings and vocabulary"""
    try:
        # Load vocabulary
        vocab_df = pd.read_csv(os.path.join(nlp_path, 'vocabulary.csv'))
        
        # Load embeddings
        embeddings = np.load(os.path.join(nlp_path, 'embeddings.npy'))
        
        # Load Word2Vec model
        model = Word2Vec.load(os.path.join(nlp_path, 'word2vec_model.bin'))
        
        return vocab_df, embeddings, model
    except:
        print("Embeddings not found. Run nlp_analysis_advanced.py first.")
        return None, None, None

def visualize_embeddings_2d(embeddings, vocab_df, n_words=100, method='tsne'):
    """Visualize word embeddings in 2D using t-SNE or PCA"""
    
    # Select top N words by frequency
    top_words = vocab_df.nlargest(n_words, 'frequency')
    top_indices = top_words.index.tolist()
    top_embeddings = embeddings[top_indices]
    
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(top_embeddings)
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers+text',
        text=top_words['word'].values,
        textposition='top center',
        marker=dict(
            size=np.log(top_words['frequency'].values) * 2,
            color=top_words['frequency'].values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Word Frequency")
        ),
        hovertemplate='Word: %{text}<br>Frequency: %{marker.color}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Word Embeddings Visualization ({method.upper()})',
        xaxis_title=f'{method.upper()} Dimension 1',
        yaxis_title=f'{method.upper()} Dimension 2',
        width=1000,
        height=800,
        showlegend=False
    )
    
    fig.write_html(f'artifacts/nlp_advanced/embeddings_{method}.html')
    print(f"Saved interactive visualization to artifacts/nlp_advanced/embeddings_{method}.html")

def explore_word_relationships(model, target_words=None):
    """Explore semantic relationships between words"""
    
    if target_words is None:
        # Default interesting words to explore
        target_words = [
            'good', 'bad', 'excellent', 'terrible',
            'fast', 'slow', 'delivery', 'quality',
            'price', 'cheap', 'expensive', 'value',
            'product', 'order', 'package', 'service'
        ]
    
    relationships = {}
    
    for word in target_words:
        if word in model.wv:
            # Find most similar words
            similar = model.wv.most_similar(word, topn=10)
            relationships[word] = similar
            
            print(f"\nWords similar to '{word}':")
            for similar_word, score in similar:
                print(f"  {similar_word}: {score:.3f}")
    
    # Word analogies
    print("\n" + "="*50)
    print("Word Analogies (A is to B as C is to ?):")
    print("="*50)
    
    analogies = [
        ('good', 'bad', 'fast'),
        ('cheap', 'expensive', 'small'),
        ('delivery', 'arrived', 'order'),
    ]
    
    for a, b, c in analogies:
        try:
            if all(word in model.wv for word in [a, b, c]):
                result = model.wv.most_similar(positive=[c, b], negative=[a], topn=3)
                print(f"\n{a} is to {b} as {c} is to:")
                for word, score in result:
                    print(f"  {word}: {score:.3f}")
        except:
            pass
    
    return relationships

def analyze_embedding_clusters(embeddings, vocab_df, n_clusters=10):
    """Cluster words based on their embeddings"""
    from sklearn.cluster import KMeans
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Add cluster labels to vocabulary
    vocab_df['cluster'] = clusters
    
    # Show sample words from each cluster
    print("\nWord Clusters based on Embeddings:")
    print("="*50)
    
    for i in range(n_clusters):
        cluster_words = vocab_df[vocab_df['cluster'] == i].nlargest(10, 'frequency')
        print(f"\nCluster {i}:")
        print(", ".join(cluster_words['word'].values))
    
    # Visualize clusters
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create scatter plot colored by cluster
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=clusters, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Word Embedding Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Annotate some high-frequency words
    top_words = vocab_df.nlargest(20, 'frequency')
    for _, word_data in top_words.iterrows():
        idx = word_data.name
        plt.annotate(word_data['word'], 
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.savefig('artifacts/nlp_advanced/embedding_clusters.png', dpi=300, bbox_inches='tight')
    print("\nSaved cluster visualization to artifacts/nlp_advanced/embedding_clusters.png")

def create_embedding_dashboard():
    """Create a comprehensive dashboard of embedding analytics"""
    
    # Load data
    vocab_df, embeddings, model = load_embeddings()
    
    if model is None:
        return
    
    print("Creating Embedding Analysis Dashboard...")
    print("="*50)
    
    # 1. Basic statistics
    print(f"\nEmbedding Statistics:")
    print(f"- Vocabulary size: {len(vocab_df)}")
    print(f"- Embedding dimensions: {embeddings.shape[1]}")
    print(f"- Total word occurrences: {vocab_df['frequency'].sum()}")
    print(f"- Most frequent words: {', '.join(vocab_df.nlargest(10, 'frequency')['word'].values)}")
    
    # 2. Visualize embeddings
    print("\nCreating 2D visualizations...")
    visualize_embeddings_2d(embeddings, vocab_df, n_words=100, method='tsne')
    visualize_embeddings_2d(embeddings, vocab_df, n_words=100, method='pca')
    
    # 3. Explore relationships
    print("\nExploring word relationships...")
    relationships = explore_word_relationships(model)
    
    # 4. Analyze clusters
    print("\nAnalyzing embedding clusters...")
    analyze_embedding_clusters(embeddings, vocab_df)
    
    # 5. Create similarity heatmap for common words
    common_words = ['good', 'bad', 'fast', 'slow', 'quality', 'price', 
                   'delivery', 'product', 'service', 'excellent']
    
    # Filter words that exist in vocabulary
    existing_words = [w for w in common_words if w in model.wv]
    
    if len(existing_words) > 2:
        similarity_matrix = np.zeros((len(existing_words), len(existing_words)))
        
        for i, word1 in enumerate(existing_words):
            for j, word2 in enumerate(existing_words):
                similarity_matrix[i, j] = model.wv.similarity(word1, word2)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=existing_words,
                   yticklabels=existing_words,
                   annot=True, fmt='.2f',
                   cmap='coolwarm', center=0,
                   square=True)
        plt.title('Word Similarity Heatmap')
        plt.tight_layout()
        plt.savefig('artifacts/nlp_advanced/similarity_heatmap.png', dpi=300)
        print("\nSaved similarity heatmap to artifacts/nlp_advanced/similarity_heatmap.png")
    
    print("\n✅ Embedding analysis complete!")
    print("Check the artifacts/nlp_advanced/ folder for visualizations.")

if __name__ == "__main__":
    create_embedding_dashboard()