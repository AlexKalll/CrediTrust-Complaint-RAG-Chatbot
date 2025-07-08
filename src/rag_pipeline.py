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
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)
        self.model = SentenceTransformer(model_name)
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

    def generate_response(self, question: str, context: List[str]) -> Generator[str, None, None]:
        """Stream generated response token-by-token"""
        prompt = self.build_prompt(question, context)
        streamer = TextIteratorStreamer(self.pipe.tokenizer)
        
        generation_kwargs = dict(
            prompt,
            streamer=streamer,
            max_new_tokens=200,
            temperature=0.3
        )
        
        thread = Thread(target=self.pipe, kwargs=generation_kwargs)
        thread.start()
        
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            if "Answer:" in full_response:
                yield full_response.split("Answer:")[1].strip()
        
        thread.join()

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
        metadata = self.retriever.metadata
        if product_filter and product_filter != "All Products":
            metadata = metadata[metadata['product'] == product_filter]
            self.retriever.metadata = metadata
        
        # Retrieve relevant chunks
        query_embed = self.retriever.embed_query(question)
        chunks, sources = self.retriever.retrieve_chunks(query_embed)
        
        # Generate and stream response
        response_stream = self.generator.generate_response(question, chunks)
        
        return {
            'question': question,
            'response_stream': response_stream,
            'sources': sources,
            'product_filter': product_filter
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