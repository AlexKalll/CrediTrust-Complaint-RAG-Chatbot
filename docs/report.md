# CrediTrust Complaint RAG Chatbot: Project Report

## 1. Introduction

This report details the development of an intelligent complaint-answering chatbot for CrediTrust Financial. The goal is to leverage Retrieval-Augmented Generation (RAG) to transform unstructured customer complaint data into actionable insights, improving response times to customer issues and enabling proactive problem identification. This system aims to empower various internal teams, including Product Management, Customer Support, and Compliance, to quickly and efficiently understand customer pain points across CrediTrust's key financial products.

## 2. Task 1: Exploratory Data Analysis (EDA) and Data Preprocessing

### Objective
The primary objective of Task 1 was to understand the structure, content, and quality of the raw Consumer Financial Protection Bureau (CFPB) complaint data and prepare it for subsequent stages of the RAG pipeline. This involved loading the dataset, performing initial exploratory data analysis, filtering records based on project requirements, and cleaning the text narratives.

### Methodology

1.  **Data Loading:** The full CFPB complaint dataset was loaded into a Pandas DataFrame.
2.  **Initial EDA:**
    * Inspected data types, missing values, and overall dataset dimensions.
    * Analyzed the distribution of complaints across the `Product` column to understand the prevalence of different financial products.
    * Calculated and visualized the word count of `Consumer complaint narrative` to identify the range of narrative lengths.
    * Identified the number of complaints with and without available narratives in the `Consumer complaint narrative` column.
3.  **Data Filtering:**
    * Filtered the dataset to include only complaints related to the five specified product categories: 'Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)', 'Savings account', and 'Money transfers'.
    * Removed all records where the `Consumer complaint narrative` field was empty or null, as these narratives are central to the RAG system.
4.  **Text Cleaning:** The `Consumer complaint narrative` text was preprocessed using the following steps:
    * **Lowercasing:** All text was converted to lowercase to ensure consistency and reduce vocabulary size.
    * **Special Character Removal:** Punctuation, numbers, and other non-alphabetic characters were removed to focus on the semantic content of the text. (e.g., using regular expressions).
    * *Other normalization techniques are also considered or applied, e.g., removing boilerplate text like "I am writing to file a complaint...".*

### Key Findings from EDA

- The initial dataset was massive, with 9,609,797 records, posing potential memory challenges, which was effectively managed by loading from Google Drive.

- A substantial portion of the original data (6,629,041 records) lacked a consumer complaint narrative, making them unusable for this RAG project.

- The Product distribution in the raw data was heavily skewed towards credit reporting, necessitating careful filtering to focus on the project's target product categories.

- The cleaning process, including lowercasing, special character removal, and boilerplate text elimination, successfully transformed raw, noisy narratives into a cleaner, more consistent format suitable for text processing and embedding.

### Output

The result of Task 1 is `data/filtered_complaints.csv`, a cleaned and filtered dataset containing `357,284` rows and `18` columns. This dataset is now ready for text chunking and embedding. The detailed steps and visualizations are available in `notebooks/EDA_and_Preprocessing.ipynb`.

## 3. Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objective
The objective of Task 2 was to transform the cleaned text narratives from Task 1 into a format suitable for efficient semantic search. This involved implementing a text chunking strategy, selecting and applying an appropriate embedding model, and creating a persistent vector store to index these embeddings along with relevant metadata.

### Methodology

1.  **Text Chunking Strategy:**
    * The `Cleaned_Narrative` column from the `filtered_complaints.csv` dataset was used as input.
    * `LangChain's RecursiveCharacterTextSplitter` was employed for chunking.
    * Initial parameters chosen were `chunk_size=[YOUR_CHUNK_SIZE_HERE]` and `chunk_overlap=[YOUR_CHUNK_OVERLAP_HERE]`.
    * Each chunk was associated with metadata, including the `Complaint ID`, `Product`, and `Issue` from the original complaint.
    * The generated chunks were saved to `data/complaint_chunks.csv`.
2.  **Embedding Model Choice:**
    * The `sentence-transformers/all-MiniLM-L6-v2` model was selected for generating vector embeddings.
    * *(Justification Placeholder: Briefly explain why this model was chosen, e.g., its balance of performance and efficiency for semantic similarity tasks.)*
3.  **Vector Store Integration (FAISS):**
    * Embeddings were generated for all text chunks.
    * A FAISS (Facebook AI Similarity Search) index was created using these embeddings.
    * The FAISS index was persisted to `vectorstore/faiss_index.bin`.
    * The corresponding metadata for each chunk (linking the vector to its original text and complaint details) was saved separately to `vectorstore/metadata.csv`.

### Key Findings and Justification

* **Chunking Strategy:**
    * *(Placeholder: Discuss your chosen `chunk_size` and `chunk_overlap`. Justify why these values were effective for the nature of customer complaint narratives. For example, did they help preserve context while keeping chunks manageable for embedding?)*
    * *(Placeholder: Mention the total number of chunks created.)*
* **Embedding Model:**
    * *(Placeholder: Reiterate the choice of `all-MiniLM-L6-v2` and provide a brief justification for its suitability for this project, e.g., good performance on semantic similarity tasks, efficiency for deployment.)*

### Output

Upon successful execution, the following outputs were generated:
* `data/complaint_chunks.csv`: A CSV file containing all the generated text chunks along with their associated `complaint_id`, `product`, and `issue`.
* `vectorstore/faiss_index.bin`: The binary file containing the persisted FAISS index.
* `vectorstore/metadata.csv`: A CSV file containing the metadata (`complaint_id`, `product`, `issue`, `chunk` text) corresponding to the vectors in the FAISS index, enabling retrieval of original context.

## 4. Task 3: Building the RAG Core Logic and Evaluation

*(To be filled after completing Task 3)*

## 5. Task 4: Creating an Interactive Chat Interface

*(To be filled after completing Task 4)*

## 6. Conclusion and Future Improvements

*(To be filled at the very end of the project)*
