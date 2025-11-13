import numpy as np
import faiss
import pickle
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

load_dotenv()

class JobRAGSystem:
    def __init__(self, embeddings_dir: Optional[str] = None):
        """
        Initialize the RAG system
        
        Args:
            embeddings_dir: Directory containing the embeddings
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings_dir = embeddings_dir or os.getenv("EMBEDDINGS_DIR", "embeddings")
        self.index = None
        self.df = None
        self.embeddings = None
        self.load_embeddings()
        
    def load_embeddings(self):
        """Load embeddings, index, and metadata from disk"""
        print("Loading embeddings...")
        
        # Load embeddings
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found at '{embeddings_path}'. "
                "Ensure the embeddings directory is available or set the EMBEDDINGS_DIR environment variable."
            )
        self.embeddings = np.load(embeddings_path)
        
        # Load FAISS index
        index_path = os.path.join(self.embeddings_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Ensure the embeddings directory is available or set the EMBEDDINGS_DIR environment variable."
            )
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(self.embeddings_dir, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at '{metadata_path}'. "
                "Ensure the embeddings directory is available or set the EMBEDDINGS_DIR environment variable."
            )
        with open(metadata_path, 'rb') as f:
            self.df = pickle.load(f)
        
        print(f"Loaded {self.index.ntotal} embeddings and {len(self.df)} job records")
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query
        
        Args:
            query: User query string
            
        Returns:
            Query embedding as numpy array
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def search_similar_jobs(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar jobs using semantic search
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of matching job dictionaries
        """
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve matching jobs
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            job = self.df.iloc[idx].to_dict()
            job['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
            results.append(job)
        
        return results
    
    def generate_rag_response(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Generate a RAG response using retrieved jobs
        
        Args:
            query: User query
            top_k: Number of jobs to retrieve
            
        Returns:
            Tuple of (AI response, retrieved jobs)
        """
        # Retrieve relevant jobs
        relevant_jobs = self.search_similar_jobs(query, top_k)
        
        # Prepare context for the LLM
        context = "Here are relevant job listings:\n\n"
        for i, job in enumerate(relevant_jobs, 1):
            context += f"""
Job {i}:
- Title: {job['job_title']}
- Company: {job['company_name']}
- Location: {job['location']}
- Salary: {job['salary']}
- Industry: {job['industry_type']}
- Type: {job['job_type']}
- Description: {job['description'][:300]}...
- URL: {job['job_url']}

"""
        
        # Generate response using GPT
        messages = [
            {
                "role": "system",
                "content": """You are a helpful job search assistant. Based on the job listings provided, 
                give personalized recommendations and insights to help the user find the best job match. 
                Be conversational and helpful."""
            },
            {
                "role": "user",
                "content": f"User Query: {query}\n\n{context}\n\nBased on these job listings, please provide helpful insights and recommendations."
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        return ai_response, relevant_jobs
    
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """
        Main query interface
        
        Args:
            user_query: User's question or search query
            top_k: Number of jobs to retrieve
            
        Returns:
            Dictionary with AI response and job results
        """
        print(f"Processing query: {user_query}")
        ai_response, jobs = self.generate_rag_response(user_query, top_k)
        
        return {
            "query": user_query,
            "ai_response": ai_response,
            "jobs": jobs,
            "total_jobs_found": len(jobs)
        }


if __name__ == "__main__":
    # Example usage
    rag_system = JobRAGSystem()
    
    # Test queries
    test_queries = [
        "I'm looking for data analyst positions in Brisbane",
        "Show me senior developer roles with good salary",
        "What business analyst jobs are available?"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        result = rag_system.query(query, top_k=3)
        print(f"\nQuery: {result['query']}")
        print(f"\nAI Response:\n{result['ai_response']}")
        print(f"\nFound {result['total_jobs_found']} relevant jobs")
