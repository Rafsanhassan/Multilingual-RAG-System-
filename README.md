## 📚 Multilingual RAG System for Bengali Documents

This repository contains a multilingual Retrieval-Augmented Generation (RAG) system built to handle Bangla and English queries over academic PDF documents. It supports local and cloud vector databases, multiple LLMs (OpenAI, Google Gemini, Ollama), and includes evaluation tools and a REST API server.

---

## ⚙️ Setup Guide

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/bangla-rag-system.git
   cd bangla-rag-system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add API keys in `rag_config.json`**
   Fill in the keys for OpenAI, Google Gemini, or Ollama in the config.

4. **Run the API server**

   ```bash
   python rest_api_server.ipynb
   ```

5. **Evaluate system**

   ```bash
   python Evaluation_system_ipnyb.ipynb
   ```

---

## 🧰 Tools, Libraries, and Packages

* **LangChain**, **OpenAI**, **Gemini API**, **Ollama**
* **PyMuPDF** for PDF parsing
* **FAISS**, **Pinecone**, **PostgreSQL**, **MongoDB** for vector storage
* **Flask** for API server
* **Matplotlib**, **Seaborn**, **Pandas** for evaluation visualization

---

## 💬 Sample Queries and Outputs

**Bangla:**

```
প্রশ্ন: 'বাংলা সাহিত্য কীভাবে রেনেসাঁস প্রভাবিত হয়েছে?'

উত্তর: বাংলা সাহিত্যে রেনেসাঁস-এর প্রভাব বাংলা উপন্যাস ও প্রবন্ধে নতুন চিন্তাধারার উন্মেষ ঘটিয়েছিল...
```

**English:**

```
Q: How did the Renaissance influence Bengali literature?

A: The Renaissance period introduced new philosophical thoughts that shaped modern Bengali essays and prose writing...
```

---

## 🔌 API Documentation

### `/chat` (POST)

```json
{
  "query": "Your question here",
  "llm_provider": "openai | google | ollama",
  "include_sources": true
}
```

**Returns:** JSON with answer, metadata, and sources.

---

### `/upload` (POST)

Upload and process a new PDF document.

---

### `/stats` (GET)

Returns system stats: # of chunks, vector DB info, LLMs used, etc.

---

### `/session/<session_id>` (GET/DELETE)

View or clear a session history.

---

## 📈 Evaluation Matrix

From `Evaluation_system_ipnyb.ipynb`:

* **Groundedness**: 0.86
* **Relevance**: 0.89
* **Answer Similarity**: 0.91
* **Response Time**: \~1.3s
* **Exact Match Rate**: 0.78

---

## ❓ Reflective Questions Answered

### ✅ What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

We used **PyMuPDF** due to its speed and decent layout preservation. Yes, some PDFs had formatting issues (misaligned headers, footnotes merged into body text), which affected chunk boundaries.

---

### ✅ What chunking strategy did you choose?

**Paragraph-based chunking** was used with optional sentence segmentation. This ensures coherent meaning is preserved for semantic retrieval, as random character/windowing approaches often split ideas mid-sentence.

---

### ✅ What embedding model did you use?

We used **`all-MiniLM-L6-v2`** from SentenceTransformers. It's fast, multilingual, and performs well in semantic similarity tasks. It captures meaning by projecting sentences into a dense vector space where similar ideas lie closer.

---

### ✅ How are you comparing the query with stored chunks?

We use **cosine similarity** between the query embedding and chunk embeddings. The vector store (FAISS or Pinecone) indexes these for fast retrieval. This method is efficient and effective for dense vector search.

---

### ✅ How do you ensure meaningful comparisons?

We:

* Normalize the text (lowercasing, punctuation cleaning)
* Ensure both query and chunks are embedded using the same model
* Set a **similarity threshold (e.g., 0.3)** to filter noise

If the query is vague, the system may return loosely related chunks. In such cases, improving chunking, adjusting thresholds, or rephrasing the query helps.

---

### ✅ Do the results seem relevant?

Yes, in most tests relevance was above 85%. To improve:

* Better PDF parsing (remove headers/footnotes)
* Use larger transformer models
* Improve human annotation and feedback loops

---

