import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from transformers import pipeline, TextIteratorStreamer
from langchain_community.llms import LlamaCpp
from threading import Thread
from sentence_transformers import SentenceTransformer
import logging
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class VectorRetriever:
    """Handles vector store operations and retrieval"""
    
    def __init__(self, index_path: Path, metadata_path: Path, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            self.index = faiss.read_index(str(index_path))
            self.metadata = pd.read_parquet(metadata_path)
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error initializing VectorRetriever: {e}")
            raise

    def embed_query(self, question: str) -> np.ndarray:
        """Convert question to embedding vector"""
        try:
            return self.model.encode([question])
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def retrieve_chunks(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[Dict]]:
        """
        Retrieves the top-k most relevant chunks from the FAISS index based on the query embedding.
        It also returns the associated metadata for these chunks.

        Args:
            query_embedding (np.ndarray): The embedding of the query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            Tuple[List[str], List[Dict]]: A tuple containing:
                - A list of chunk texts.
                - A list of dictionaries, where each dictionary contains the chunk's text, product,
                  issue, and relevance score.
        """
        try:
            # Ensure k doesn't exceed the number of available vectors
            k = min(k, self.index.ntotal)
            if k == 0:
                return [], []
                
            distances, indices = self.index.search(query_embedding, k)
            chunks = []
            sources = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):  # Check bounds
                    chunk_data = {
                        'text': self.metadata.iloc[idx]['chunk_text'],
                        'product': self.metadata.iloc[idx]['product'],
                        'issue': self.metadata.iloc[idx]['issue'],
                        'score': float(distances[0][i])
                    }
                    chunks.append(chunk_data['text'])
                    sources.append(chunk_data)
                
            return chunks, sources
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

class LLMGenerator:
    """Handles LLM interactions and response generation using a pre-trained language model."""
    
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"):
        """
        Initializes the LLMGenerator with a specified model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.model_name = model_name
        self.pipe = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """
        Initializes the text generation pipeline using the Hugging Face transformers library.

        Returns:
            pipeline: A Hugging Face text generation pipeline object.

        Raises:
            Exception: If there is an error during pipeline initialization.
        """
        try:
            root_dir = Path(__file__).resolve().parent.parent
            model_path = root_dir / "models" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

            if not model_path.exists():
                logger.warning(f"Model file not found at {model_path}. Using fallback pipeline.")
                # Fallback to a simpler text generation approach
                return self._create_fallback_pipeline()

            return LlamaCpp(
                model_path=str(model_path),
                temperature=0.3,
                max_tokens=200,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error initializing LLM pipeline: {e}")
            logger.info("Using fallback pipeline due to initialization error.")
            return self._create_fallback_pipeline()
    
    def _create_fallback_pipeline(self):
        """Creates a fallback pipeline when the main model is not available"""
        try:
            # Use a smaller, more accessible model
            return pipeline(
                "text-generation",
                model="gpt2",
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
        except Exception as e:
            logger.error(f"Error creating fallback pipeline: {e}")
            # Return a dummy pipeline that provides basic responses
            return self._create_dummy_pipeline()
    
    def _create_dummy_pipeline(self):
        """Creates a dummy pipeline for when no models are available"""
        class DummyPipeline:
            def __call__(self, text, **kwargs):
                return [{"generated_text": "I'm sorry, but I'm currently unable to generate detailed responses. Please ensure you have the required model files installed."}]
        return DummyPipeline()

    def build_prompt(self, question: str, context: List[str]) -> str:
        """
        Constructs a RAG-style prompt by combining the user's question with retrieved context.

        Args:
            question (str): The user's question.
            context (List[str]): A list of relevant text chunks retrieved from the vector store.

        Returns:
            str: The formatted prompt string ready for the LLM.

        Raises:
            Exception: If there is an error during prompt construction.
        """
        try:
            context_str = "\n".join([f"• {c[:200]}..." for c in context])
            return f"""You are a CreditTrust financial analyst. Use ONLY these complaint excerpts:\n\n{context_str}\n\nQuestion: {question}\nAnswer concisely in 2-3 sentences. If unsure, say "I don't have enough information":"""
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise

class RAGEvaluator:
    """Handles pipeline evaluation and quality metrics for RAG responses."""
    
    @staticmethod
    def format_sources(sources: List[Dict]) -> str:
        """
        Formats a list of retrieved sources for display.

        Args:
            sources (List[Dict]): A list of dictionaries, where each dictionary represents a source
                                   and contains 'product', 'score', and 'text' keys.

        Returns:
            str: A formatted string of the top 2 sources.

        Raises:
            Exception: If there is an error during source formatting.
        """
        try:
            return "\n\n".join(
                f"📌 {src['product']} (relevance: {src['score']:.2f}):\n"
                f"> {src['text'][:150]}..."
                for src in sources[:2]
            )
        except Exception as e:
            logger.error(f"Error formatting sources: {e}")
            raise

    @staticmethod
    def evaluate_response(question: str, answer: str, sources: List[Dict]) -> Dict:
        """
        Generates evaluation metrics for a given RAG response.

        Args:
            question (str): The original question.
            answer (str): The answer generated by the LLM.
            sources (List[Dict]): The list of sources retrieved for the question.

        Returns:
            Dict: A dictionary containing evaluation metrics such as question, answer,
                  number of sources used, average relevance, and product coverage.

        Raises:
            Exception: If there is an error during response evaluation.
        """
        try:
            return {
                'question': question,
                'answer': answer,
                'sources_used': len(sources),
                'avg_relevance': np.mean([s['score'] for s in sources]),
                'product_coverage': list(set(s['product'] for s in sources))
            }
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            raise

class RAGPipeline:
    """Main pipeline combining VectorRetriever, LLMGenerator, and RAGEvaluator components."""
    
    def __init__(self, index_path: Path, metadata_path: Path):
        """
        Initializes the RAGPipeline with paths to the FAISS index and metadata.

        Args:
            index_path (Path): The path to the FAISS index file.
            metadata_path (Path): The path to the metadata Parquet file.

        Raises:
            Exception: If there is an error during initialization of any component.
        """
        try:
            self.retriever = VectorRetriever(index_path, metadata_path)
            self.generator = LLMGenerator()
            self.evaluator = RAGEvaluator()
            logger.info("RAG pipeline initialized")
        except Exception as e:
            logger.error(f"Error initializing RAGPipeline: {e}")
            raise

    def query(self, question: str, product_filter: str = None) -> Dict:
        """
        Handles an end-to-end query by retrieving relevant information, generating a response,
        and providing streaming capabilities.

        Args:
            question (str): The user's question.
            product_filter (str, optional): An optional product filter to narrow down the search.
                                            Defaults to None, meaning no filter.

        Returns:
            Dict: A dictionary containing the original question, a response streamer,
                  retrieved sources, the applied product filter, and the generation thread.

        Raises:
            Exception: If an error occurs during any stage of the query process.
        """
        try:
            # Apply product filter if specified
            current_metadata = self.retriever.metadata.copy()
            if product_filter and product_filter != "All Products":
                current_metadata = current_metadata[current_metadata['product'] == product_filter]
            
            # Retrieve relevant chunks using the potentially filtered metadata
            query_embed = self.retriever.embed_query(question)
            # Temporarily set metadata for retrieval, then restore
            original_metadata = self.retriever.metadata
            self.retriever.metadata = current_metadata
            chunks, sources = self.retriever.retrieve_chunks(query_embed)
            self.retriever.metadata = original_metadata # Restore original metadata
            
            # Prepare streaming
            prompt = self.generator.build_prompt(question, chunks)
            
            # Generate response based on pipeline type
            if hasattr(self.generator.pipe, 'stream'):
                # LlamaCpp pipeline
                full_response = ""
                for chunk in self.generator.pipe.stream(prompt):
                    full_response += chunk['choices'][0]['text']
            else:
                # Fallback pipeline (transformers or dummy)
                try:
                    result = self.generator.pipe(prompt, max_length=200, do_sample=True)
                    if isinstance(result, list) and len(result) > 0:
                        full_response = result[0].get('generated_text', '')
                    else:
                        full_response = str(result)
                except Exception as e:
                    logger.error(f"Error generating response with fallback pipeline: {e}")
                    full_response = "I apologize, but I encountered an error while generating a response. Please try again."
            
            return {
                'question': question,
                'response': full_response,
                'sources': sources,
                'product_filter': product_filter,
            }
        except Exception as e:
            logger.error(f"Error during RAG pipeline query: {e}")
            raise

    def evaluate(self, questions: List[str]) -> pd.DataFrame:
        """
        Performs a batch evaluation of the RAG pipeline for a list of questions.

        Args:
            questions (List[str]): A list of questions to evaluate.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation results for each question.

        Raises:
            Exception: If an error occurs during the evaluation of any question.
        """
        results = []
        for question in questions:
            result = self.query(question)
            response = result['response']
            eval_result = self.evaluator.evaluate_response(
                question=question,
                answer=response,
                sources=result['sources']
            )
            results.append(eval_result)
        return pd.DataFrame(results)
