import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.text_chunking_embedding import (
    load_data,
    chunk_narratives,
    load_embedding_model,
    generate_embeddings,
    create_faiss_index,
    save_outputs,
    sample_data,
    CHUNKS_OUTPUT_PATH,
    FAISS_INDEX_PATH,
    METADATA_OUTPUT_PATH
)

# Mock data for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'Complaint ID': [1, 2, 3, 4, 5],
        'Product': ['Credit card', 'Personal loan', 'Credit card', 'Buy Now, Pay Later (BNPL)', 'Savings account'],
        'Cleaned_Narrative': [
            'This is a test complaint about a credit card. It has some text.',
            'Another complaint regarding a personal loan. The terms were unclear.',
            'Third complaint, credit card issue. Billing error.',
            'BNPL service charged me incorrectly. Need a refund.',
            'Savings account interest rate is too low. Unhappy with the bank.'
        ],
        'Issue': ['Billing error', 'Unclear terms', 'Interest rate', 'Incorrect charge', 'Low interest']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_file(tmp_path, sample_dataframe):
    file_path = tmp_path / "test_complaints.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def temp_output_paths(tmp_path):
    # Override global paths for testing
    global CHUNKS_OUTPUT_PATH, FAISS_INDEX_PATH, METADATA_OUTPUT_PATH
    original_chunks_path = CHUNKS_OUTPUT_PATH
    original_faiss_path = FAISS_INDEX_PATH
    original_metadata_path = METADATA_OUTPUT_PATH

    CHUNKS_OUTPUT_PATH = tmp_path / 'data' / 'complaint_chunks.parquet'
    FAISS_INDEX_PATH = tmp_path / 'vectorstore' / 'faiss_index.bin'
    METADATA_OUTPUT_PATH = tmp_path / 'vectorstore' / 'metadata.parquet'

    yield

    # Restore original paths after testing
    CHUNKS_OUTPUT_PATH = original_chunks_path
    FAISS_INDEX_PATH = original_faiss_path
    METADATA_OUTPUT_PATH = original_metadata_path


def test_load_data(temp_data_file, sample_dataframe):
    df = load_data(temp_data_file)
    pd.testing.assert_frame_equal(df, sample_dataframe)
    with pytest.raises(FileNotFoundError):
        load_data(Path("non_existent_file.csv"))
    with pytest.raises(ValueError):
        empty_file = temp_data_file.parent / "empty.csv"
        pd.DataFrame().to_csv(empty_file)
        load_data(empty_file)

def test_sample_data(sample_dataframe):
    sampled_df = sample_data(sample_dataframe, frac=0.5, random_state=42)
    assert not sampled_df.empty
    # Check if sampling reduced the number of rows
    assert len(sampled_df) < len(sample_dataframe)
    # Check if all products are still represented (if possible with small sample)
    assert set(sampled_df['Product'].unique()) == set(sample_dataframe['Product'].unique())

def test_chunk_narratives(sample_dataframe):
    df_chunks = chunk_narratives(sample_dataframe)
    assert not df_chunks.empty
    assert 'chunk_text' in df_chunks.columns
    assert 'complaint_id' in df_chunks.columns
    assert 'product' in df_chunks.columns
    assert len(df_chunks) >= len(sample_dataframe) # Chunks should be equal or more than original narratives

    with pytest.raises(ValueError, match="Missing required columns"):
        chunk_narratives(sample_dataframe.drop(columns=['Product']))

def test_load_embedding_model():
    model = load_embedding_model()
    assert model is not None
    assert isinstance(model, SentenceTransformer)

def test_generate_embeddings(sample_dataframe):
    df_chunks = chunk_narratives(sample_dataframe)
    model = load_embedding_model()
    embeddings = generate_embeddings(df_chunks, model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(df_chunks)
    assert embeddings.shape[1] > 0 # Embeddings should have a dimension

    with pytest.raises(ValueError, match="'chunk_text' column"):
        generate_embeddings(sample_dataframe.drop(columns=['Cleaned_Narrative']), model)

def test_create_faiss_index():
    # Create dummy embeddings
    embeddings = np.random.rand(100, 384).astype('float32') # 100 embeddings of 384 dimensions
    index = create_faiss_index(embeddings)
    assert index is not None
    assert index.ntotal == 100

    with pytest.raises(ValueError, match="No embeddings provided"):
        create_faiss_index(np.array([]))

def test_save_outputs(sample_dataframe, temp_output_paths):
    df_chunks = chunk_narratives(sample_dataframe)
    embeddings = np.random.rand(len(df_chunks), 384).astype('float32')
    index = create_faiss_index(embeddings)

    save_outputs(df_chunks, index)

    assert CHUNKS_OUTPUT_PATH.exists()
    assert FAISS_INDEX_PATH.exists()
    assert METADATA_OUTPUT_PATH.exists()

    # Verify content
    loaded_chunks = pd.read_parquet(CHUNKS_OUTPUT_PATH)
    pd.testing.assert_frame_equal(loaded_chunks, df_chunks)

    loaded_metadata = pd.read_parquet(METADATA_OUTPUT_PATH)
    expected_metadata_cols = ['complaint_id', 'product', 'issue', 'chunk_text']
    pd.testing.assert_frame_equal(loaded_metadata, df_chunks[expected_metadata_cols])

    loaded_index = faiss.read_index(str(FAISS_INDEX_PATH))
    assert loaded_index.ntotal == index.ntotal