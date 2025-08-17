# Project Design Document: CrediTrust Complaint RAG Chatbot

## 1. Introduction
This document outlines the design and architecture of the CrediTrust Complaint RAG (Retrieval-Augmented Generation) Chatbot. The primary goal of this project is to develop an intelligent system that can answer user queries related to customer complaints by leveraging a knowledge base of past complaint narratives. The system combines information retrieval (using FAISS for vector search) with a language model to generate relevant and coherent responses.

## 2. Project Structure
The project is organized into several key directories:

- `src/`: Contains the core Python source code for data processing, embedding, and the RAG pipeline.
- `data/`: Stores raw and processed data files, including complaint datasets and generated chunks.
- `vectorstore/`: Holds the FAISS index and associated metadata for efficient similarity search.
- `dashboards/`: (Formerly `app/`) Contains Streamlit applications for interactive visualization and demonstration.
- `tests/`: Houses unit and integration tests to ensure code quality and functionality.
- `.github/workflows/`: Contains CI/CD pipeline definitions for automated testing and deployment.
- `models/`: (Optional) Could be used for storing pre-trained models or model checkpoints.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, prototyping, and evaluation.

## 3. Core Components and Functionality

### 3.1 Data Preprocessing (`src/data_preprocessing.py`)
This module is responsible for cleaning and preparing the raw complaint data. Key steps include:
- Loading raw CSV data.
- Handling missing values.
- Text cleaning (e.g., lowercasing, removing special characters, stop words, and punctuation).
- Tokenization and lemmatization.

### 3.2 Text Chunking and Embedding (`src/text_chunking_embedding.py`)
This module processes the cleaned complaint narratives to create a searchable knowledge base.
- **Chunking**: Large complaint narratives are split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter` from Langchain. This ensures that relevant information is captured in manageable segments.
- **Embedding**: Each text chunk is converted into a high-dimensional vector (embedding) using a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
- **FAISS Indexing**: The generated embeddings are indexed using FAISS (Facebook AI Similarity Search) for fast and efficient similarity search. An `IndexIVFFlat` index is used for optimized performance, especially with large datasets.
- **Metadata Storage**: Alongside embeddings, relevant metadata (e.g., `complaint_id`, `product`, `issue`) for each chunk is stored to facilitate retrieval and context provision.

### 3.3 RAG Pipeline (`src/rag_pipeline.py`)
This module orchestrates the retrieval-augmented generation process.
- **VectorRetriever**: This class loads the pre-built FAISS index and metadata. It takes a user query, converts it into an embedding, and performs a similarity search to retrieve the most relevant text chunks from the knowledge base.
- **Context Formulation**: The retrieved chunks are combined to form a coherent context for the language model.
- **Language Model Integration**: (To be implemented/integrated) A language model (e.g., from OpenAI, Hugging Face) will take the user query and the retrieved context to generate a comprehensive and relevant answer.

### 3.4 Utility Functions (`src/utils.py`)
Contains helper functions used across different modules, such as logging configuration, path management, or common data manipulation tasks.

## 4. Interactive Dashboards (`dashboards/app.py`)
This directory will contain Streamlit applications designed to provide interactive visualizations and demonstrations of the RAG chatbot's capabilities. This includes:
- A user interface for interacting with the chatbot.
- Visualizations of retrieved chunks and their relevance scores.
- (Potentially) Metrics and insights into complaint data.

## 5. Testing (`tests/`)
Unit and integration tests will be developed to ensure the correctness and robustness of the system. This includes testing:
- Data preprocessing steps.
- Text chunking logic.
- Embedding generation.
- FAISS index creation and retrieval accuracy.
- End-to-end RAG pipeline functionality.

## 6. CI/CD Pipeline (`.github/workflows/ci.yml`)
An automated CI/CD pipeline is set up using GitHub Actions to ensure code quality and facilitate continuous integration and deployment. The pipeline includes:
- **Build and Test**: Automatically runs tests on every push to `main` and pull requests to `main`.
- **Deployment**: (Placeholder) Steps for deploying the application to a production environment upon successful completion of tests on the `main` branch.
* **Purpose:** Automate code quality checks and deployment readiness.
* **Stages:** Linting (`flake8`), Unit Tests, Integration Tests.
* **Tool:** GitHub Actions.

## 7. Reproducibility
- **Relative File Paths**: All internal file paths are defined relative to the project root to ensure portability.
- **`requirements.txt`**: A `requirements.txt` file specifies all necessary Python dependencies with their versions, allowing for easy environment setup.
- **Clear Instructions**: The `README.md` and this `design-doc.md` provide comprehensive instructions for setting up, running, and understanding the project.

## 8. Future Enhancements
- Integration with a specific large language model (LLM) for response generation.
- Advanced evaluation metrics for RAG performance.
* Containerization (Docker).
* Deployment to cloud platforms.
* More advanced LLMs or fine-tuning.
* Scalable data ingestion.
- Deployment to cloud platforms (e.g., Azure, AWS, GCP).
- Incorporating user feedback for continuous improvement.