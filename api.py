from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_system import JobRAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize RAG system
rag_system = None

@app.before_request
def initialize_rag():
    """Initialize RAG system on first request"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = JobRAGSystem()
            print("RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Job RAG API is running"
    }), 200

@app.route('/api/search', methods=['POST'])
def search_jobs():
    """
    Search for jobs using RAG
    
    Request body:
    {
        "query": "user query string",
        "top_k": 5  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            top_k = 5
        
        # Get RAG response
        result = rag_system.query(query, top_k=top_k)
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        print(f"Error processing search: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/similar-jobs', methods=['POST'])
def get_similar_jobs():
    """
    Get similar jobs without AI response
    
    Request body:
    {
        "query": "user query string",
        "top_k": 5  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        
        # Get similar jobs
        jobs = rag_system.search_similar_jobs(query, top_k=top_k)
        
        return jsonify({
            "success": True,
            "data": {
                "query": query,
                "jobs": jobs,
                "total": len(jobs)
            }
        }), 200
        
    except Exception as e:
        print(f"Error getting similar jobs: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        return jsonify({
            "success": True,
            "data": {
                "total_jobs": len(rag_system.df),
                "total_embeddings": rag_system.index.ntotal,
                "embedding_dimension": rag_system.embeddings.shape[1]
            }
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
