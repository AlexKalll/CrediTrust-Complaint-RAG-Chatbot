import os
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import faiss

from src.data_preprocessing import load_data, preprocess_data
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration Paths ---
BASE_DIR = Path(__file__).parent.parent
CHUNKS_OUTPUT_PATH = BASE_DIR / "data/complaint_chunks.parquet"
FAISS_INDEX_PATH = BASE_DIR / "vectorstore/faiss_index.bin"
METADATA_OUTPUT_PATH = BASE_DIR / "vectorstore/metadata.parquet"

# --- Model and Chunking Parameters ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 32
FAISS_NLIST = 100  # Number of clusters for IVF index


class TextChunkingEmbedding:
    """
    A class to handle text chunking, embedding generation, and FAISS indexing.
    It encapsulates the entire process from loading raw data to creating a searchable FAISS index.
    """

    def __init__(
        self,
        chunks_output_path: Path = CHUNKS_OUTPUT_PATH,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        metadata_output_path: Path = METADATA_OUTPUT_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
        faiss_nlist: int = FAISS_NLIST,
    ):
        """
        Initializes the TextChunkingEmbedding pipeline with specified paths and parameters.

        Args:
            chunks_output_path (Path): Path where processed chunks will be saved.
            faiss_index_path (Path): Path where the FAISS index will be saved.
            metadata_output_path (Path): Path where chunk metadata will be saved.
            embedding_model_name (str): Name of the SentenceTransformer model to use for embeddings.
            chunk_size (int): Maximum size of text chunks.
            chunk_overlap (int): Overlap between consecutive text chunks.
            embedding_batch_size (int): Batch size for generating embeddings.
            faiss_nlist (int): Number of clusters for the FAISS IVF index.
        """

        self.chunks_output_path = chunks_output_path
        self.faiss_index_path = faiss_index_path
        self.metadata_output_path = metadata_output_path
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size
        self.faiss_nlist = faiss_nlist
        self.model = None

    def chunk_narratives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chunks text narratives into smaller segments with metadata using RecursiveCharacterTextSplitter.

        Args:
            df (pd.DataFrame): DataFrame containing narratives, expected to have 'Complaint ID', 'Product', and 'Cleaned_Narrative'.

        Returns:
            pd.DataFrame: DataFrame with each row representing a chunk, including its text and associated metadata.

        Raises:
            ValueError: If required columns ('Complaint ID', 'Product', 'Cleaned_Narrative') are missing from the input DataFrame.
        """
        required_columns = {"Cleaned_Narrative", "Complaint ID", "Product"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(
            f"Chunking narratives (size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        for _, row in df.iterrows():
            narrative = str(row["Cleaned_Narrative"]).strip()
            if not narrative:
                continue

            chunks = text_splitter.create_documents(
                texts=[narrative],
                metadatas=[
                    {
                        "complaint_id": row["Complaint ID"],
                        "product": row["Product"],
                        "issue": row.get("Issue", "General complaint"),
                        "original_length": len(narrative),
                    }
                ],
            )

            for chunk in chunks:
                all_chunks.append(
                    {
                        **chunk.metadata,
                        "chunk_text": chunk.page_content,
                        "chunk_length": len(chunk.page_content),
                    }
                )

        df_chunks = pd.DataFrame(all_chunks)
        logger.info(f"Created {len(df_chunks)} chunks from {len(df)} narratives")
        return df_chunks

    def load_embedding_model(self) -> SentenceTransformer:
        """
        Loads the SentenceTransformer model specified by `self.embedding_model_name`.
        The model is loaded only once and reused for subsequent calls.

        Returns:
            SentenceTransformer: An initialized SentenceTransformer model instance.

        Raises:
            RuntimeError: If the model fails to load, typically due to network issues or invalid model name.
        """
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Could not load embedding model: {str(e)}")
        return self.model

    def generate_embeddings(self, df_chunks: pd.DataFrame) -> np.ndarray:
        """
        Generates embeddings for text chunks in batches using the loaded SentenceTransformer model.

        Args:
            df_chunks (pd.DataFrame): DataFrame containing the text chunks, expected to have a 'chunk_text' column.

        Returns:
            np.ndarray: A NumPy array of embeddings, where each row corresponds to a chunk's embedding.

        Raises:
            ValueError: If the input DataFrame `df_chunks` does not contain a 'chunk_text' column.
            RuntimeError: If the embedding generation process fails (e.g., model not loaded, encoding error).
        """
        if "chunk_text" not in df_chunks.columns:
            raise ValueError("DataFrame missing 'chunk_text' column")

        model = self.load_embedding_model()
        logger.info("Generating embeddings (this may take a while)...")
        texts = df_chunks["chunk_text"].tolist()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            logger.info(
                f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Could not generate embeddings: {str(e)}")

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Creates and trains a FAISS IndexIVFFlat index from the given embeddings.

        Args:
            embeddings (np.ndarray): A NumPy array of embeddings.

        Returns:
            faiss.Index: A trained FAISS index.

        Raises:
            ValueError: If no embeddings are provided or if embeddings have an invalid shape.
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to create FAISS index.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        dimension = embeddings.shape[1]
        logger.info(
            f"Creating FAISS index with dimension {dimension} and nlist {self.faiss_nlist}..."
        )

        # Ensure nlist is not greater than the number of embeddings
        nlist = min(self.faiss_nlist, embeddings.shape[0])
        if nlist == 0:
            raise ValueError("Cannot create FAISS index with zero embeddings.")

        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        try:
            index.train(embeddings)
            index.add(embeddings)
            logger.info(
                f"FAISS index created and populated with {index.ntotal} vectors."
            )
            return index
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise RuntimeError(f"Could not create FAISS index: {str(e)}")

    def save_outputs(self, df_chunks: pd.DataFrame, faiss_index: faiss.Index):
        """
        Saves the processed chunks metadata and the FAISS index to specified paths.

        Args:
            df_chunks (pd.DataFrame): DataFrame containing chunk metadata.
            faiss_index (faiss.Index): The trained FAISS index.
        """
        self.chunks_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving chunks metadata to {self.metadata_output_path}...")
        # Select only relevant columns for metadata to save space and ensure consistency
        metadata_cols = [
            "complaint_id",
            "product",
            "issue",
            "chunk_text",
            "chunk_length",
        ]
        df_chunks[metadata_cols].to_parquet(self.metadata_output_path, index=False)

        logger.info(f"Saving FAISS index to {self.faiss_index_path}...")
        faiss.write_index(faiss_index, str(self.faiss_index_path))

        logger.info("Outputs saved successfully.")

    def run_pipeline(self, input_data_path: Path) -> Tuple[pd.DataFrame, faiss.Index]:
        """
        Runs the complete text chunking and embedding pipeline.

        Args:
            input_data_path (Path): The path to the input CSV data file.

        Returns:
            Tuple[pd.DataFrame, faiss.Index]: A tuple containing the DataFrame of chunks and the FAISS index.
        """
        logger.info("Starting Text Chunking and Embedding Pipeline...")

        # 1. Load Data
        df = load_data(input_data_path)

        # 2. Preprocess Data
        df = preprocess_data(df)

        # 3. Chunk Narratives
        df_chunks = self.chunk_narratives(df)

        # 4. Generate Embeddings
        embeddings = self.generate_embeddings(df_chunks)

        # 5. Create FAISS Index
        faiss_index = self.create_faiss_index(embeddings)

        # 6. Save Outputs
        self.save_outputs(df_chunks, faiss_index)

        logger.info("Text Chunking and Embedding Pipeline Completed.")
        return df_chunks, faiss_index


if __name__ == "__main__":
    # Example Usage:
    # Ensure you have a 'filtered_complaints.csv' in the 'data' directory
    dummy_data_path = BASE_DIR / "data/filtered_complaints.csv"
    if not dummy_data_path.exists():
        print(f"Creating dummy data at {dummy_data_path} for demonstration.")
        dummy_df = pd.DataFrame(
            {
                "Complaint ID": [1, 2, 3],
                "Product": ["Credit card", "Personal loan", "Credit card"],
                "Consumer complaint narrative": [
                    "This is a test complaint about a credit card. It has numbers like 123 and symbols!@#.",
                    "Another complaint regarding bank services. Missing values here.",
                    "A third complaint, very clear and concise. No issues.",
                ],
                "Issue": ["Billing error", "Unclear terms", "No issues"],
            }
        )
        dummy_df.to_csv(dummy_data_path, index=False)

    pipeline = TextChunkingEmbedding()
    chunks_df, index = pipeline.run_pipeline(dummy_data_path)
    print(
        f"\nSuccessfully processed {len(chunks_df)} chunks and created FAISS index with {index.ntotal} vectors."
    )

    # Clean up dummy data if created by this script
    if "dummy_df" in locals() and dummy_data_path.exists():
        print(f"Cleaning up dummy data at {dummy_data_path}.")
        os.remove(dummy_data_path)
