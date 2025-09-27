from typing import List
import numpy as np
from config.settings import settings
import logging
# Use FastEmbed instead of SentenceTransformer
from fastembed import TextEmbedding

class EmbeddingManager:
    def __init__(self):
        self.model = None
        # The model name for FastEmbed. You can place this in your settings file.
        self.model_name = "BAAI/bge-small-en-v1.5" 
        self.load_model()
    
    def load_model(self):
        """Load the lightweight FastEmbed model."""
        try:
            print(f"Loading embedding model: {self.model_name}...")
            # Initialize with default parameters (quantized, runs on CPU)
            # This is significantly faster than sentence-transformers
            self.model = TextEmbedding(model_name=self.model_name)
            print(f"Successfully loaded embedding model: {self.model_name}")
            logging.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            logging.error(f"Error loading embedding model: {e}")
            self.model = None # Ensure model is None if loading fails

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            logging.error("Model not loaded, cannot generate embedding.")
            # Fallback to a zero vector of the expected dimension (384 for bge-small)
            return [0.0] * 384
            
        try:
            # FastEmbed's embed method returns a list of embeddings
            embedding = self.model.embed(text)[0]
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return [0.0] * 384

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            logging.error("Model not loaded, cannot generate embeddings.")
            return [[0.0] * 384 for _ in texts]
            
        try:
            embeddings = self.model.embed(texts)
            # Convert each numpy array in the list to a Python list
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logging.error(f"Error generating batch embeddings: {e}")
            return [[0.0] * 384 for _ in texts]

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            a = np.array(embedding1)
            b = np.array(embedding2)
            # Add a small epsilon to the denominator to avoid division by zero
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
        except Exception as e:
            logging.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def embed_business_requirements(self) -> dict:
        """Generate embeddings for business requirements."""
        requirements = settings.BUSINESS_REQUIREMENTS
        embeddings = {}
        
        for category, items in requirements.items():
            # Use the efficient batch method
            category_embeddings = self.generate_embeddings_batch(items)
            embeddings[category] = category_embeddings
        
        return embeddings

embedding_manager = EmbeddingManager()