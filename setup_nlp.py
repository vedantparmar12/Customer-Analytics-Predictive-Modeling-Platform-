#!/usr/bin/env python3
"""
Setup script for NLP dependencies
"""
import subprocess
import sys

def setup_nlp():
    """Download required NLP models and data"""
    
    print("Setting up NLP dependencies...")
    
    # Download NLTK data
    print("\n1. Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    # Download SpaCy model
    print("\n2. Downloading SpaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("\n✅ NLP setup complete!")
    print("You can now run the advanced NLP analysis.")

if __name__ == "__main__":
    setup_nlp()