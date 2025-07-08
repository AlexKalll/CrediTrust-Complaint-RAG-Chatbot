# CrediTrust Complaint RAG Chatbot

## Project Overview

This project aims to develop an intelligent complaint analysis tool for CrediTrust Financial, a fast-growing digital finance company. The tool, a Retrieval-Augmented Generation (RAG) powered chatbot, will transform raw, unstructured customer complaint data into actionable insights, enabling internal stakeholders like Product Managers, Support, and Compliance teams to quickly understand customer pain points and emerging trends across various financial products.

The primary objective is to decrease the time it takes to identify major complaint trends from days to minutes, empower non-technical teams to get answers without needing a data analyst, and shift the company from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## Project Structure

The project is organized into the following directories:

```
CrediTrust-Complaint-RAG-Chatbot/
├── README.md                 \# Project overview and setup instructions
├── requirements.txt          \# Python dependencies
├── data/                     \# Stores raw and preprocessed datasets
│   └── filtered\_complaints.csv \# Cleaned and filtered complaint data
├── notebooks/                \# Jupyter notebooks for EDA, experimentation, etc.
│   └── EDA\_and\_Preprocessing.ipynb \# Exploratory Data Analysis and data cleaning steps
├── src/                      \# Source code for the RAG pipeline components
│   ├── **init**.py           \# Makes src a Python package
│   ├── data\_preprocessing.py \# Contains functions for data loading, cleaning, and filtering
│   ├── text\_chunking\_embedding.py \# Handles text chunking, embedding generation, and vector store management
│   ├── rag\_pipeline.py       \# Implements the core RAG logic (retriever, generator, prompt engineering)
│   └── utils.py              \# Helper functions
├── vectorstore/              \# Persisted vector store files (FAISS/ChromaDB)
├── app/                      \# Contains the interactive chat interface
│   └── app.py                \# Gradio/Streamlit application script
└── docs/                     \# Documentation, reports, and evaluation summaries
    └── report.md             \# Final project report and evaluation summary

```

## Setup Instructions

### Prerequisites

* Python 3.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AlexKalll/CrediTrust-Complaint-RAG-Chatbot.git
    cd CrediTrust-Complaint-RAG-Chatbot
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used in this project is sourced from the Consumer Financial Protection Bureau (CFPB) complaint data. It includes real customer complaints across various financial products. The core input for the RAG system is the `Consumer complaint narrative` column.

The `data/filtered_complaints.csv` file contains the preprocessed and filtered subset of this data, specifically targeting the five key product categories and removing entries without narratives.

## Current Progress

**Task 1: Exploratory Data Analysis and Data Preprocessing**
* **Status:** Completed.
* **Description:** Performed initial EDA on the CFPB complaint dataset, analyzed complaint distributions, narrative lengths, and identified records with/without narratives. The dataset has been filtered to include only records for "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", "Savings account", and "Money transfers" products, and records with empty `Consumer complaint narrative` fields have been removed. Text narratives have been cleaned (lowercasing, special character removal).
* **Output:** The cleaned and filtered dataset is saved at `data/filtered_complaints.csv`. A detailed analysis is provided in `docs/report.md` and the `notebooks/EDA_and_Preprocessing.ipynb` notebook.

**Task 2: Text Chunking, Embedding, and Vector Store Indexing**
* **Status:** Completed.
* **Description:** Implemented logic for text chunking using `LangChain's RecursiveCharacterTextSplitter`. Chosen `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings. Integrated FAISS for efficient vector storage and retrieval, ensuring metadata (Complaint ID, Product, Issue) is stored alongside embeddings for source attribution.
* **Output:** Expected outputs include `data/complaint_chunks.csv`, `vectorstore/faiss_index.bin`, and `vectorstore/metadata.csv`. Detailed findings and justifications will be added to `docs/report.md` after execution.

## How to Run

### 1. Launch the Chat Interface
```bash
python app/app.py
```
Access at: `http://localhost:7860`

### 2. Command Line Query
```bash
python src/rag_pipeline.py --query "What are common BNPL issues?"
```

### System Requirements
| Component | Minimum Specs |
|-----------|---------------|
| CPU | 4 cores (Intel i5+) |
| RAM | 8GB |
| Disk Space | 5GB (including models) |

## Key Features

✅ **Complaint Analysis**
- Semantic search across 400K+ complaints
- Product-specific filtering (5 categories)

✅ **User Experience**
- Streaming responses
- Source attribution
- Feedback collection

✅ **Deployment Ready**
- CPU-optimized
- Container support (Dfile included)
- Modular architecture

## Example Usage

1. **Ask about product issues:**
   ```
   "Show recent complaints about credit card late fees"
   ```

2. **Compare products:**
   ```
   "What issues are unique to BNPL vs personal loans?"
   ```

3. **Investigate trends:**
   ```
   "Are mobile app complaints increasing?"
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow responses | Reduce chunk size in `src/retriever.py` |
| OOM errors | Decrease batch size in `src/generator.py` |
| Model not found | Download GGUF model to `models/` |



## License
Apache 2.0 - See [LICENSE](LICENSE)
