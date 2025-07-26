# Complete RAG System Deployment Guide

## 🚀 Installation & Setup

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# rag_env\Scripts\activate  # Windows

# Install core dependencies
pip install flask flask-cors
pip install sentence-transformers faiss-cpu
pip install PyPDF2 PyMuPDF nltk langdetect
pip install psycopg2-binary pymongo pinecone-client
pip install langchain langchain-openai langchain-google-genai langchain-community
pip install matplotlib seaborn pandas numpy scikit-learn
pip install requests python-dotenv
```

### 2. Configuration Setup
```bash
# Generate configuration file
python rag_api_server.py --generate-config

# Edit rag_config.json with your API keys
```

### 3. Build Knowledge Base
```python
from multilingual_rag import MultilingualRAG

# Initialize and build knowledge base
rag = MultilingualRAG()
rag.build_knowledge_base(["path/to/HSC26_Bangla_1st_paper.pdf"])
```

## 🌐 API Server Deployment

### Local Development
```bash
# Start the API server
python rag_api_server.py --host 0.0.0.0 --port 5000 --debug

# Server will be available at http://localhost:5000
```

### Production Deployment with Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 rag_api_server:api.app
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "rag_api_server:api.app"]
```

```bash
# Build and run Docker container
docker build -t multilingual-rag .
docker run -p 5000:5000 -v $(pwd)/data:/app/data multilingual-rag
```

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Chat Endpoint
```
POST /chat
Content-Type: application/json

{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "session_id": "optional_session_id",
  "llm_provider": "openai",
  "include_sources": true
}
```

### 3. Upload Document
```
POST /upload
Content-Type: multipart/form-data

file: [PDF file]
```

### 4. System Statistics
```
GET /stats
```

### 5. Session Management
```
GET /session/{session_id}     # Get session history
DELETE /session/{session_id}  # Clear session
```

## 💻 API Client Examples

### Python Client
```python
import requests
import json

class RAGClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_id = None
    
    def chat(self, query, llm_provider="openai", include_sources=True):
        """Send a chat query to the RAG system"""
        url = f"{self.base_url}/chat"
        
        data = {
            "query": query,
            "llm_provider": llm_provider,
            "include_sources": include_sources
        }
        
        if self.session_id:
            data["session_id"] = self.session_id
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            self.session_id = result.get("session_id")
            return result
        else:
            return {"error": f"Request failed: {response.status_code}"}
    
    def upload_document(self, file_path):
        """Upload a document to the RAG system"""
        url = f"{self.base_url}/upload"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        return response.json()
    
    def get_stats(self):
        """Get system statistics"""
        url = f"{self.base_url}/stats"
        response = requests.get(url)
        return response.json()
    
    def clear_session(self):
        """Clear current session"""
        if self.session_id:
            url = f"{self.base_url}/session/{self.session_id}"
            requests.delete(url)
            self.session_id = None

# Example usage
client = RAGClient()

# Test Bengali query
result = client.chat("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
print(f"Answer: {result['response']}")

# Test English query
result = client.chat("What was Kalyani's actual age at the time of marriage?")
print(f"Answer: {result['response']}")

# Get system stats
stats = client.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### JavaScript Client
```javascript
class RAGClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async chat(query, llmProvider = 'openai', includeSources = true) {
        const url = `${this.baseUrl}/chat`;
        
        const data = {
            query: query,
            llm_provider: llmProvider,
            include_sources: includeSources
        };
        
        if (this.sessionId) {
            data.session_id = this.sessionId;
        }
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.sessionId = result.session_id;
                return result;
            } else {
                return { error: `Request failed: ${response.status}` };
            }
        } catch (error) {
            return { error: error.message };
        }
    }
    
    async uploadDocument(file) {
        const url = `${this.baseUrl}/upload`;
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }
    
    async getStats() {
        const url = `${this.baseUrl}/stats`;
        
        try {
            const response = await fetch(url);
            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }
}

// Example usage
const client = new RAGClient();

// Test query
client.chat("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
    .then(result => console.log("Answer:", result.response));
```

### cURL Examples
```bash
# Health check
curl -X GET http://localhost:5000/health

# Chat query
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "llm_provider": "openai",
    "include_sources": true
  }'

# Upload document
curl -X POST http://localhost:5000/upload \
  -F "file=@path/to/document.pdf"

# Get statistics
curl -X GET http://localhost:5000/stats
```

## 🧪 Evaluation System Usage

### Running Basic Evaluation
```python
from multilingual_rag import MultilingualRAG
from rag_evaluation_system import RAGEvaluator

# Initialize systems
rag = MultilingualRAG()
evaluator = RAGEvaluator(rag)

# Run evaluation
report = evaluator.run_evaluation()

# Save results
evaluator.save_evaluation_report(report)

# Generate visualizations
evaluator.visualize_results()
```

### Custom Test Dataset
```json
{
  "test_cases": [
    {
      "query": "Your question here",
      "expected_answer": "Expected answer",
      "category": "question_type",
      "language": "bn"
    }
  ]
}
```

### Benchmark Evaluation
```python
from rag_evaluation_system import BenchmarkRunner

benchmark = BenchmarkRunner(rag)
results = benchmark.run_multilingual_benchmark()
```

## 🔧 Advanced Configuration

### Vector Database Options

#### PostgreSQL with pgvector
```json
{
  "vector_database": {
    "type": "postgres",
    "connection_params": {
      "host": "localhost",
      "database": "rag_db",
      "user": "postgres",
      "password": "your_password"
    }
  }
}
```

#### Pinecone
```json
{
  "vector_database": {
    "type": "pinecone",
    "connection_params": {
      "api_key": "your-pinecone-api-key",
      "index_name": "multilingual-rag"
    }
  }
}
```

#### MongoDB with Vector Search
```json
{
  "vector_database": {
    "type": "mongodb",
    "connection_params": {
      "connection_string": "mongodb://localhost:27017/",
      "database": "rag_db"
    }
  }
}
```

### LLM Provider Configuration

#### OpenAI
```json
{
  "llm_providers": {
    "openai": {
      "api_key": "sk-your-api-key",
      "model": "gpt-4",
      "temperature": 0.1
    }
  }
}
```

#### Google Gemini
```json
{
  "llm_providers": {
    "google": {
      "api_key": "your-google-api-key",
      "model": "gemini-pro"
    }
  }
}
```

#### Local Ollama
```json
{
  "llm_providers": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "model": "llama2"
    }
  }
}
```

## 📊 Performance Monitoring

### Custom Metrics Dashboard
```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_performance_dashboard(evaluation_results):
    """Create a performance monitoring dashboard"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Response time over time
    times = [r.response_time for r in evaluation_results]
    axes[0, 0].plot(times)
    axes[0, 0].set_title('Response Time Trend')
    axes[0, 0].set_ylabel('Seconds')
    
    # Accuracy distribution
    accuracies = [r.answer_similarity for r in evaluation_results]
    axes[0, 1].hist(accuracies, bins=20, alpha=0.7)
    axes[0, 1].set_title('Answer Accuracy Distribution')
    axes[0, 1].set_xlabel('Similarity Score')
    
    # Language performance comparison
    lang_performance = {}
    for result in evaluation_results:
        lang = result.language
        if lang not in lang_performance:
            lang_performance[lang] = []
        lang_performance[lang].append(result.answer_similarity)
    
    lang_means = {lang: np.mean(scores) for lang, scores in lang_performance.items()}
    axes[1, 0].bar(lang_means.keys(), lang_means.values())
    axes[1, 0].set_title('Performance by Language')
    axes[1, 0].set_ylabel('Average Similarity')
    
    # Groundedness vs Relevance
    groundedness = [r.groundedness_score for r in evaluation_results]
    relevance = [r.relevance_score for r in evaluation_results]
    axes[1, 1].scatter(groundedness, relevance, alpha=0.6)
    axes[1, 1].set_xlabel('Groundedness')
    axes[1, 1].set_ylabel('Relevance')
    axes[1, 1].set_title('Groundedness vs Relevance')
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 🐛 Troubleshooting

### Common Issues

1. **PDF Text Extraction Issues**
```python
# Try different extraction methods
def robust_pdf_extraction(pdf_path):
    try:
        # Try PyMuPDF first
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        # Fallback to PyPDF2
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
```

2. **Bengali Text Encoding Issues**
```python
import unicodedata

def fix_bengali_encoding(text):
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    # Remove problematic characters
    text = re.sub(r'[\u200c\u200d]', '', text)  # Remove zero-width chars
    return text
```

3. **Low Retrieval Accuracy**
```python
# Adjust retrieval parameters
retrieval_config = {
    "top_k": 10,  # Increase number of retrieved chunks
    "similarity_threshold": 0.2,  # Lower threshold
    "chunk_size": 300,  # Smaller chunks for better precision
    "overlap": 75  # More overlap between chunks
}
```

4. **Memory Issues with Large Documents**
```python
# Process in batches
def process_large_document_in_batches(pdf_path, batch_size=50):
    pages = extract_pages_from_pdf(pdf_path)
    
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i+batch_size]
        process_batch(batch)
        # Optional: save intermediate results
```

## 🔐 Security Considerations

### API Security
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    # Your chat endpoint
    pass
```

### Input Validation
```python
from flask import request
import re

def validate_query(query):
    """Validate user query"""
    if not query or len(query.strip()) == 0:
        raise ValueError("Query cannot be empty")
    
    if len(query) > 1000:
        raise ValueError("Query too long")
    
    # Check for potential injection attempts
    dangerous_patterns = [r'<script', r'javascript:', r'eval\(']
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Invalid query content")
    
    return query.strip()
```

## 📈 Scaling Considerations

### Horizontal Scaling
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: .
    ports:
      - "5000-5002:5000"
    deploy:
      replicas: 3
    environment:
      - REDIS_URL=redis://redis:6379
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Caching Strategy
```python
import redis
import json
import hashlib

class ResponseCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour
    
    def get_cache_key(self, query, params):
        """Generate cache key for query and parameters"""
        cache_data = {
            "query": query.lower().strip(),
            "params": params
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query, params):
        """Get cached response"""
        key = self.get_cache_key(query, params)
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query, params, response):
        """Cache response"""
        key = self.get_cache_key(query, params)
        self.redis_client.setex(
            key, 
            self.cache_ttl,
            json.dumps(response, ensure_ascii=False)
        )
```

## 🎯 Testing Strategy

### Unit Tests
```python
import unittest
from multilingual_rag import MultilingualRAG

class TestRAGSystem(unittest.TestCase):
    
    def setUp(self):
        self.rag = MultilingualRAG()
    
    def test_bengali_text_processing(self):
        """Test Bengali text preprocessing"""
        text = "এটি একটি পরীক্ষা।"
        processed = self.rag.document_processor.clean_text(text, 'bn')
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)
    
    def test_chunking(self):
        """Test document chunking"""
        text = "This is a test document. " * 100
        chunks = self.rag.document_processor.chunk_text(text, 'en')
        self.assertGreater(len(chunks), 1)
    
    def test_retrieval(self):
        """Test document retrieval"""
        # This requires a built knowledge base
        query = "test query"
        results = self.rag.retrieve_relevant_chunks(query, k=3)
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests
```python
import requests
import time

class TestAPIIntegration(unittest.TestCase):
    
    def setUp(self):
        self.base_url = "http://localhost:5000"
        # Wait for server to be ready
        time.sleep(2)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_chat_endpoint(self):
        """Test chat functionality"""
        payload = {
            "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "llm_provider": "openai"
        }
        response = requests.post(f"{self.base_url}/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("session_id", data)
```

This comprehensive deployment guide provides everything needed to deploy and scale your multilingual RAG system effectively. The system includes REST API endpoints, multiple LLM provider support, comprehensive evaluation metrics, and production-ready deployment configurations.
