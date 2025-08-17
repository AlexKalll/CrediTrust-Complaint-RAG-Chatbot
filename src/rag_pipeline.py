import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        """Retrieve top-k relevant chunks with metadata"""
        try:
            distances, indices = self.index.search(query_embedding, k)
            chunks = []
            sources = []
            
            for idx in indices[0]:
                chunk_data = {
                    'text': self.metadata.iloc[idx]['chunk_text'],
                    'product': self.metadata.iloc[idx]['product'],
                    'issue': self.metadata.iloc[idx]['issue'],
                    'score': float(distances[0][idx])
                }
                chunks.append(chunk_data['text'])
                sources.append(chunk_data)
                
            return chunks, sources
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

class LLMGenerator:
    """Handles LLM interactions and response generation"""
    
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"):
        self.model_name = model_name
        self.pipe = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the text generation pipeline"""
        try:
            return pipeline(
                "text-generation",
                model=self.model_name,
                model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                device_map="auto",
                max_new_tokens=200,
                temperature=0.3
            )
        except Exception as e:
            logger.error(f"Error initializing LLM pipeline: {e}")
            raise

    def build_prompt(self, question: str, context: List[str]) -> str:
        """Construct RAG prompt with context"""
        try:
            context_str = "\n".join([f"• {c[:200]}..." for c in context])
            return f"""You are a CreditTrust financial analyst. Use ONLY these complaint excerpts:\n\n{context_str}\n\nQuestion: {question}\nAnswer concisely in 2-3 sentences. If unsure, say "I don't have enough information":"""
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise

class RAGEvaluator:
    """Handles pipeline evaluation and quality metrics"""
    
    @staticmethod
    def format_sources(sources: List[Dict]) -> str:
        """Format retrieved sources for display"""
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
        """Generate evaluation metrics for a response"""
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
    """Main pipeline combining all components"""
    
    def __init__(self, index_path: Path, metadata_path: Path):
        try:
            self.retriever = VectorRetriever(index_path, metadata_path)
            self.generator = LLMGenerator()
            self.evaluator = RAGEvaluator()
            logger.info("RAG pipeline initialized")
        except Exception as e:
            logger.error(f"Error initializing RAGPipeline: {e}")
            raise

    def query(self, question: str, product_filter: str = None) -> Dict:
        """End-to-end query handling"""
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
            streamer = TextIteratorStreamer(self.generator.pipe.tokenizer)
            
            # Start generation in a separate thread
            generation_kwargs = dict(
                prompt,
                streamer=streamer,
                max_new_tokens=200,
                temperature=0.3
            )
            thread = Thread(target=self.generator.pipe, kwargs=generation_kwargs)
            thread.start()
            
            return {
                'question': question,
                'response_streamer': streamer,
                'sources': sources,
                'product_filter': product_filter,
                'generation_thread': thread
            }
        except Exception as e:
            logger.error(f"Error during RAG pipeline query: {e}")
            raise
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def embed_query(self, question: str) -> np.ndarray:
        """Convert question to embedding vector"""
        return self.model.encode([question])

    def retrieve_chunks(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve top-k relevant chunks with metadata"""
        distances, indices = self.index.search(query_embedding, k)
        chunks = []
        sources = []
        
        for idx in indices[0]:
            chunk_data = {
                'text': self.metadata.iloc[idx]['chunk_text'],
                'product': self.metadata.iloc[idx]['product'],
                'issue': self.metadata.iloc[idx]['issue'],
                'score': float(distances[0][idx])
            }
            chunks.append(chunk_data['text'])
            sources.append(chunk_data)
            
        return chunks, sources

class LLMGenerator:
    """Handles LLM interactions and response generation"""
    
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"):
        self.model_name = model_name
        self.pipe = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the text generation pipeline"""
        return pipeline(
            "text-generation",
            model=self.model_name,
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            device_map="auto",
            max_new_tokens=200,
            temperature=0.3
        )

    def build_prompt(self, question: str, context: List[str]) -> str:
        """Construct RAG prompt with context"""
        context_str = "\n".join([f"• {c[:200]}..." for c in context])
        return f"""You are a CreditTrust financial analyst. Use ONLY these complaint excerpts:

{context_str}

Question: {question}
Answer concisely in 2-3 sentences. If unsure, say "I don't have enough information":"""

class RAGEvaluator:
    """Handles pipeline evaluation and quality metrics"""
    
    @staticmethod
    def format_sources(sources: List[Dict]) -> str:
        """Format retrieved sources for display"""
        return "\n\n".join(
            f"📌 {src['product']} (relevance: {src['score']:.2f}):\n"
            f"> {src['text'][:150]}..."
            for src in sources[:2]
        )

    @staticmethod
    def evaluate_response(question: str, answer: str, sources: List[Dict]) -> Dict:
        """Generate evaluation metrics for a response"""
        return {
            'question': question,
            'answer': answer,
            'sources_used': len(sources),
            'avg_relevance': np.mean([s['score'] for s in sources]),
            'product_coverage': list(set(s['product'] for s in sources))
        }

class RAGPipeline:
    """Main pipeline combining all components"""
    
    def __init__(self, index_path: Path, metadata_path: Path):
        self.retriever = VectorRetriever(index_path, metadata_path)
        self.generator = LLMGenerator()
        self.evaluator = RAGEvaluator()
        logger.info("RAG pipeline initialized")

    def query(self, question: str, product_filter: str = None) -> Dict:
        """End-to-end query handling"""
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
        streamer = TextIteratorStreamer(self.generator.pipe.tokenizer)
        
        # Start generation in a separate thread
        generation_kwargs = dict(
            prompt,
            streamer=streamer,
            max_new_tokens=200,
            temperature=0.3
        )
        thread = Thread(target=self.generator.pipe, kwargs=generation_kwargs)
        thread.start()
        
        return {
            'question': question,
            'response_streamer': streamer,
            'sources': sources,
            'product_filter': product_filter,
            'generation_thread': thread
        }

    def evaluate(self, questions: List[str]) -> pd.DataFrame:
        """Batch evaluate the pipeline"""
        results = []
        for question in questions:
            result = self.query(question)
            response = "".join([chunk for chunk in result['response_stream']])
            eval_result = self.evaluator.evaluate_response(
                question=question,
                answer=response,
                sources=result['sources']
            )
            results.append(eval_result)
        return pd.DataFrame(results)