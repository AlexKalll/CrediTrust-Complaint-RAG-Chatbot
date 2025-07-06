# src/text_chunking_embedding.py

import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os

# --- Configuration Paths ---
# Path to the cleaned and filtered dataset from Task 1
INPUT_DATA_PATH = '../data/filtered_complaints.csv'
# Path to save the generated text chunks (with metadata)
CHUNKS_OUTPUT_PATH = '../data/complaint_chunks.csv'
# Path to save the FAISS index file
FAISS_INDEX_PATH = '../vectorstore/faiss_index.bin'
# Path to save the metadata associated with each chunk (for retrieval)
METADATA_OUTPUT_PATH = '../vectorstore/metadata.csv'

# --- Model and Chunking Parameters ---
# Embedding model to use for generating vector representations of text
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# Size of each text chunk (in characters or words, depending on length_function)
CHUNK_SIZE = 500
# Overlap between consecutive chunks to maintain context
CHUNK_OVERLAP = 50

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the preprocessed complaints data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame. Returns an empty DataFrame if file not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input data file not found at {file_path}. Please ensure Task 1 is complete.")
        return pd.DataFrame()
    print(f"Loading filtered complaints data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def chunk_narratives(df: pd.DataFrame, text_column: str = 'Cleaned_Narrative') -> pd.DataFrame:
    """
    Chunks the text narratives from the DataFrame into smaller, overlapping segments.
    Each chunk is associated with its original complaint ID and product.

    Args:
        df (pd.DataFrame): The DataFrame containing the cleaned narratives.
        text_column (str): The name of the column containing the text to chunk.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a chunk,
                      including 'complaint_id', 'product', and 'chunk' text.
    """
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in the DataFrame. "
              "Please ensure your data preprocessing created this column.")
        return pd.DataFrame()

    print(f"Chunking narratives from '{text_column}' column (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    
    # Initialize LangChain's RecursiveCharacterTextSplitter
    # It splits text by characters by default, which is usually robust.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, # Use character length
        add_start_index=True, # Optional: adds character offset metadata
    )

    all_chunks = []
    # Iterate through each row of the DataFrame to process narratives
    for index, row in df.iterrows():
        narrative = str(row[text_column]) # Ensure narrative is a string
        
        # Skip if the narrative is empty or just whitespace after cleaning
        if pd.isna(narrative) or not narrative.strip():
            continue

        # Split the current narrative into smaller chunks
        # create_documents returns a list of Document objects.
        # We pass metadata for the whole document, which gets copied to each chunk.
        chunks_for_narrative = text_splitter.create_documents(
            texts=[narrative],
            metadatas=[{
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'issue': row.get('Issue', 'N/A') # Include 'Issue' if available, or 'N/A'
            }]
        )
        
        # Extract the chunk text and its associated metadata
        for chunk_doc in chunks_for_narrative:
            all_chunks.append({
                'complaint_id': chunk_doc.metadata['complaint_id'],
                'product': chunk_doc.metadata['product'],
                'issue': chunk_doc.metadata['issue'], # Add issue to metadata
                'chunk': chunk_doc.page_content # The actual text content of the chunk
            })
    
    df_chunks = pd.DataFrame(all_chunks)
    print(f"Finished chunking. Total {len(df_chunks)} chunks created.")
    return df_chunks

def generate_embeddings(df_chunks: pd.DataFrame) -> np.ndarray:
    """
    Generates embeddings for each text chunk using a Sentence Transformer model.

    Args:
        df_chunks (pd.DataFrame): DataFrame containing the 'chunk' column.

    Returns:
        np.ndarray: A NumPy array of embeddings.
    """
    if 'chunk' not in df_chunks.columns:
        print("Error: 'chunk' column not found in the chunks DataFrame.")
        return np.array([])

    print(f"Initializing SentenceTransformer model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Generating embeddings for chunks (this may take a while)...")
    # Encode all chunks into embeddings. batch_size can be adjusted based on memory.
    embeddings = model.encode(df_chunks['chunk'].tolist(), batch_size=32, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings

def create_and_save_faiss_index(embeddings: np.ndarray, index_file_path: str):
    """
    Creates a FAISS index from the embeddings and saves it to a file.

    Args:
        embeddings (np.ndarray): The NumPy array of embeddings.
        index_file_path (str): The path where the FAISS index will be saved.
    """
    if embeddings.size == 0:
        print("No embeddings to index. Skipping FAISS index creation.")
        return

    dimension = embeddings.shape[1] # Dimension of the embeddings (e.g., 384 for MiniLM)
    print(f"Building FAISS index (dimension: {dimension})...")
    
    # Use IndexFlatL2 for a simple L2 distance (Euclidean distance) based index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings) # Add the embeddings to the index
    
    faiss.write_index(index, index_file_path)
    print(f"Saved FAISS index with {index.ntotal} vectors to {index_file_path}")

def main():
    """
    Main function to orchestrate data loading, chunking, embedding,
    and FAISS index creation and persistence.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(CHUNKS_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_OUTPUT_PATH), exist_ok=True)

    # 1. Load the filtered data
    df = load_data(INPUT_DATA_PATH)
    if df.empty:
        return

    # 2. Chunk the narratives
    df_chunks = chunk_narratives(df, text_column='Cleaned_Narrative')
    if df_chunks.empty:
        print("No chunks were generated. Exiting.")
        return
    
    # Save the chunks DataFrame to CSV (as requested)
    df_chunks[['complaint_id', 'product', 'issue', 'chunk']].to_csv(CHUNKS_OUTPUT_PATH, index=False)
    print(f"Saved {len(df_chunks)} chunks to {CHUNKS_OUTPUT_PATH}")

    # 3. Generate embeddings for the chunks
    embeddings = generate_embeddings(df_chunks)
    if embeddings.size == 0:
        print("No embeddings generated. Exiting.")
        return

    # 4. Create and save FAISS index
    create_and_save_faiss_index(embeddings, FAISS_INDEX_PATH)
    
    # 5. Save metadata separately (excluding the 'embedding' column if it were added to df_chunks)
    # We'll save the core metadata that links back to the original chunks
    df_chunks[['complaint_id', 'product', 'issue', 'chunk']].to_csv(METADATA_OUTPUT_PATH, index=False)
    print(f"Saved metadata to {METADATA_OUTPUT_PATH}")
    print("Task 2: Text Chunking, Embedding, and Vector Store Indexing completed successfully!")

if __name__ == "__main__":
    main()
