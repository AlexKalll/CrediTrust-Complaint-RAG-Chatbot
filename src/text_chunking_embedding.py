import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import faiss

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Paths ---
BASE_DIR = Path(__file__).parent.parent
INPUT_DATA_PATH = BASE_DIR / 'data/filtered_complaints.csv'
CHUNKS_OUTPUT_PATH = BASE_DIR / 'data/complaint_chunks.parquet'
FAISS_INDEX_PATH = BASE_DIR / 'vectorstore/faiss_index.bin'
METADATA_OUTPUT_PATH = BASE_DIR / 'vectorstore/metadata.parquet'

# --- Model and Chunking Parameters ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 32
FAISS_NLIST = 100  # Number of clusters for IVF index

def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found at {file_path}")
    
    logger.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def chunk_narratives(df: pd.DataFrame, text_column: str = 'Cleaned_Narrative') -> pd.DataFrame:
    """
    Chunks text narratives into smaller segments with metadata.
    
    Args:
        df: DataFrame containing narratives
        text_column: Name of column containing text to chunk
        
    Returns:
        DataFrame with chunks and metadata
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {text_column, 'Complaint ID', 'Product'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Chunking narratives (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    all_chunks = []
    for _, row in df.iterrows():
        narrative = str(row[text_column]).strip()
        if not narrative:
            continue

        chunks = text_splitter.create_documents(
            texts=[narrative],
            metadatas=[{
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'issue': row.get('Issue', 'N/A'),
                'original_length': len(narrative)
            }]
        )
        
        for chunk in chunks:
            all_chunks.append({
                **chunk.metadata,
                'chunk_text': chunk.page_content,
                'chunk_length': len(chunk.page_content)
            })
    
    df_chunks = pd.DataFrame(all_chunks)
    logger.info(f"Created {len(df_chunks)} chunks from {len(df)} narratives")
    return df_chunks

def load_embedding_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer model with error handling.
    
    Returns:
        Initialized SentenceTransformer model
        
    Raises:
        RuntimeError: If model fails to load
    """
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load embedding model: {str(e)}")

def generate_embeddings(df_chunks: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Generates embeddings for text chunks in batches.
    
    Args:
        df_chunks: DataFrame containing chunks
        model: Initialized SentenceTransformer
        
    Returns:
        Numpy array of embeddings
    """
    if 'chunk_text' not in df_chunks.columns:
        raise ValueError("DataFrame missing 'chunk_text' column")
    
    logger.info("Generating embeddings (this may take a while)...")
    texts = df_chunks['chunk_text'].tolist()
    
    try:
        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Embedding generation failed: {str(e)}")

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Creates optimized FAISS index for embeddings.
    
    Args:
        embeddings: Array of embeddings to index
        
    Returns:
        Constructed FAISS index
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings provided for indexing")
    
    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={dimension}, nlist={FAISS_NLIST})...")
    
    # Create quantizer and IVF index for better performance
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, FAISS_NLIST)
    
    # Train on a subset if dataset is large
    if len(embeddings) > 10_000:
        train_sample = embeddings[:10_000]
    else:
        train_sample = embeddings
        
    index.train(train_sample)
    index.add(embeddings)
    
    logger.info(f"Index contains {index.ntotal} vectors")
    return index

def save_outputs(df_chunks: pd.DataFrame, index: faiss.Index) -> None:
    """
    Saves all outputs to disk.
    
    Args:
        df_chunks: Chunks DataFrame
        index: FAISS index
    """
    # Ensure directories exist
    for path in [CHUNKS_OUTPUT_PATH, FAISS_INDEX_PATH, METADATA_OUTPUT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    df_chunks.to_parquet(CHUNKS_OUTPUT_PATH)
    logger.info(f"Saved chunks to {CHUNKS_OUTPUT_PATH}")
    
    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    
    # Save metadata (without embeddings)
    metadata_cols = ['complaint_id', 'product', 'issue', 'chunk_text']
    df_chunks[metadata_cols].to_parquet(METADATA_OUTPUT_PATH)
    logger.info(f"Saved metadata to {METADATA_OUTPUT_PATH}")

def main():
    """Orchestrates the chunking, embedding and indexing pipeline."""
    try:
        # 1. Load data
        df = load_data(INPUT_DATA_PATH)
        
        # 2. Chunk narratives
        df_chunks = chunk_narratives(df)
        if df_chunks.empty:
            raise ValueError("No chunks generated - check input data")
        
        # 3. Generate embeddings
        model = load_embedding_model()
        embeddings = generate_embeddings(df_chunks, model)
        
        # 4. Create and save FAISS index
        index = create_faiss_index(embeddings)
        
        # 5. Save all outputs
        save_outputs(df_chunks, index)
        
        logger.info("Text chunking and embedding completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()