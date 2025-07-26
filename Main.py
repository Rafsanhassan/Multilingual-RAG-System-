import os
import re
import json
import pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Core libraries for document processing
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# For text preprocessing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# For Bengali text processing
import re
import unicodedata

# Language detection
from langdetect import detect

# For generating responses (you can replace with your preferred LLM API)
import openai  # Optional: replace with your preferred LLM

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    text: str
    source: str
    page_number: int
    chunk_id: str
    embedding: np.ndarray = None
    language: str = None

class BengaliTextProcessor:
    """Handles Bengali text preprocessing and normalization"""

    def __init__(self):
        # Bengali stopwords (you can expand this list)
        self.bengali_stopwords = {
            'এবং', 'বা', 'কিন্তু', 'তবে', 'যদি', 'তাহলে', 'কারণ', 'যেহেতু',
            'সেহেতু', 'অথচ', 'তথাপি', 'বরং', 'অতএব', 'সুতরাং', 'ফলে',
            'এর', 'তার', 'আমার', 'তোমার', 'আপনার', 'আমাদের', 'তাদের',
            'এই', 'ঐ', 'সেই', 'ওই', 'কোন', 'কোনো', 'যে', 'যা', 'যার',
            'আছে', 'নেই', 'হয়', 'হবে', 'ছিল', 'থাকে', 'যায়', 'আসে',
            'দেয়', 'নেয়', 'করে', 'হয়ে', 'থেকে', 'পর্যন্ত', 'দিয়ে'
        }

    def normalize_bengali_text(self, text: str) -> str:
        """Normalize Bengali text by handling Unicode variations"""
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep Bengali characters
        text = re.sub(r'[^\u0980-\u09FF\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)

        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """Remove Bengali stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.bengali_stopwords]
        return ' '.join(filtered_words)

    def preprocess_bengali(self, text: str) -> str:
        """Complete Bengali text preprocessing pipeline"""
        text = self.normalize_bengali_text(text)
        # You can add more preprocessing steps here
        return text

class MultilingualDocumentProcessor:
    """Handles document processing for multiple languages"""

    def __init__(self):
        self.bengali_processor = BengaliTextProcessor()

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        pages_text = []

        try:
            # Try with PyMuPDF first (better for complex layouts)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    pages_text.append((text, page_num + 1))
            doc.close()
        except Exception as e:
            print(f"PyMuPDF failed: {e}, trying PyPDF2...")
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append((text, page_num + 1))

        return pages_text

    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            # Check if text contains Bengali characters
            if re.search(r'[\u0980-\u09FF]', text):
                return 'bn'
            else:
                lang = detect(text)
                return lang if lang in ['en', 'bn'] else 'en'
        except:
            return 'en'

    def clean_text(self, text: str, language: str) -> str:
        """Clean and preprocess text based on language"""
        if language == 'bn':
            return self.bengali_processor.preprocess_bengali(text)
        else:
            # English preprocessing
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)
            return text.strip()

    def chunk_text(self, text: str, language: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        # Use sentence tokenization for better chunks
        if language == 'bn':
            # Simple sentence splitting for Bengali (can be improved)
            sentences = re.split(r'[।!?]', text)
        else:
            sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

class VectorStore:
    """Handles vector storage and retrieval using FAISS"""

    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = None

    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        if not chunks:
            return

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Initialize FAISS index if not exists
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self.chunks.append(chunk)

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        """Save vector store to disk"""
        data = {
            'chunks': self.chunks,
            'dimension': self.dimension
        }

        with open(f"{path}_chunks.pkl", 'wb') as f:
            pickle.dump(data, f)

        if self.index is not None:
            faiss.write_index(self.index, f"{path}_index.faiss")

    def load(self, path: str):
        """Load vector store from disk"""
        try:
            with open(f"{path}_chunks.pkl", 'rb') as f:
                data = pickle.load(f)

            self.chunks = data['chunks']
            self.dimension = data['dimension']

            if os.path.exists(f"{path}_index.faiss"):
                self.index = faiss.read_index(f"{path}_index.faiss")

            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

class ConversationMemory:
    """Handles short-term conversation memory"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []

    def add_interaction(self, query: str, response: str):
        """Add a query-response pair to memory"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        })

        # Keep only recent interactions
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_context(self, include_last_n: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""

        recent_history = self.conversation_history[-include_last_n:]
        context_parts = []

        for interaction in recent_history:
            context_parts.append(f"Q: {interaction['query']}")
            context_parts.append(f"A: {interaction['response']}")

        return "\n".join(context_parts)

class MultilingualRAG:
    """Main RAG system class"""

    def __init__(self, vector_store_path: str = "rag_vectorstore"):
        self.document_processor = MultilingualDocumentProcessor()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
        self.vector_store_path = vector_store_path

        # Try to load existing vector store
        if not self.vector_store.load(vector_store_path):
            print("No existing vector store found. You'll need to build the knowledge base first.")

    def build_knowledge_base(self, pdf_paths: List[str], save_path: str = None):
        """Build knowledge base from PDF documents"""
        all_chunks = []

        for pdf_path in pdf_paths:
            print(f"Processing {pdf_path}...")

            # Extract text from PDF
            pages_text = self.document_processor.extract_text_from_pdf(pdf_path)

            for page_text, page_num in pages_text:
                # Detect language
                language = self.document_processor.detect_language(page_text)

                # Clean text
                cleaned_text = self.document_processor.clean_text(page_text, language)

                if not cleaned_text.strip():
                    continue

                # Chunk text
                chunks = self.document_processor.chunk_text(cleaned_text, language)

                # Create DocumentChunk objects
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        chunk = DocumentChunk(
                            text=chunk_text,
                            source=pdf_path,
                            page_number=page_num,
                            chunk_id=f"{pdf_path}_page{page_num}_chunk{i}",
                            language=language
                        )
                        all_chunks.append(chunk)

        print(f"Created {len(all_chunks)} chunks from {len(pdf_paths)} documents")

        # Add to vector store
        self.vector_store.add_documents(all_chunks)

        # Save vector store
        save_path = save_path or self.vector_store_path
        self.vector_store.save(save_path)
        print(f"Knowledge base saved to {save_path}")

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant document chunks for a query"""
        return self.vector_store.search(query, k)

    def generate_response(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]],
                         use_openai: bool = False, api_key: str = None) -> str:
        """Generate response based on retrieved chunks"""

        # Prepare context from retrieved chunks
        context_parts = []
        for chunk, score in retrieved_chunks:
            context_parts.append(f"[Score: {score:.3f}] {chunk.text}")

        context = "\n\n".join(context_parts)

        # Get conversation history
        conversation_context = self.memory.get_context()

        # Detect query language
        query_language = self.document_processor.detect_language(query)

        if use_openai and api_key:
            # Use OpenAI API
            openai.api_key = api_key

            system_prompt = """You are a helpful multilingual assistant that answers questions based on provided context.
            Answer in the same language as the question. If the question is in Bengali, answer in Bengali.
            If the question is in English, answer in English.
            Base your answer strictly on the provided context. If the answer is not in the context, say so politely."""

            user_prompt = f"""
            Previous conversation:
            {conversation_context}

            Context from documents:
            {context}

            Question: {query}

            Please provide a direct and accurate answer based on the context provided.
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")
                # Fallback to simple extraction

        # Simple rule-based response generation (fallback)
        return self._simple_response_generation(query, retrieved_chunks, query_language)

    def _simple_response_generation(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]],
                                  query_language: str) -> str:
        """Simple response generation without external LLM"""
        if not retrieved_chunks:
            if query_language == 'bn':
                return "দুঃখিত, আপনার প্রশ্নের উত্তর আমার জ্ঞানভাণ্ডারে খুঁজে পাইনি।"
            else:
                return "I couldn't find relevant information in the knowledge base to answer your question."

        # For now, return the most relevant chunk
        best_chunk, score = retrieved_chunks[0]

        if score < 0.3:  # Low similarity threshold
            if query_language == 'bn':
                return "দুঃখিত, আপনার প্রশ্নের সাথে সম্পর্কিত কোনো তথ্য খুঁজে পাইনি।"
            else:
                return "I couldn't find sufficiently relevant information to answer your question."

        # Extract potential answer from the best chunk
        chunk_text = best_chunk.text

        # Simple keyword matching for Bengali questions
        if query_language == 'bn':
            # Look for potential answers in the chunk
            sentences = re.split(r'[।!?]', chunk_text)
            for sentence in sentences:
                if any(word in sentence for word in query.split()):
                    return sentence.strip()

        # Return the most relevant chunk with some context
        return f"Based on the available information: {chunk_text[:300]}..."

    def chat(self, query: str, use_openai: bool = False, api_key: str = None) -> str:
        """Main chat interface"""
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)

        # Generate response
        response = self.generate_response(query, relevant_chunks, use_openai, api_key)

        # Add to conversation memory
        self.memory.add_interaction(query, response)

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_chunks': len(self.vector_store.chunks),
            'conversation_history_length': len(self.memory.conversation_history),
            'languages_in_knowledge_base': list(set(chunk.language for chunk in self.vector_store.chunks if chunk.language))
        }

def main():
    """Colab-compatible usage of the RAG system with Google Drive dataset"""

    from google.colab import drive
    import glob
    import os

    # Step 1: Mount Google Drive
    drive.mount('/content/drive')

    # Step 2: Set path to your PDF folder in Drive
    # 👉 CHANGE this to match the folder in your Google Drive where PDFs are stored
    pdf_folder = '/content/drive/MyDrive/thedatasetpdf'  # <-- change this as needed

    # Step 3: Find all PDF files in that folder
    pdf_paths = glob.glob(os.path.join(pdf_folder, '*.pdf'))

    if not pdf_paths:
        print("❗ No PDF files found in the folder. Please upload PDFs to your Google Drive folder first.")
        return

    # Step 4: Set path to store FAISS index + chunks (saved in Drive)
    vector_store_path = os.path.join(pdf_folder, 'vector_store')

    # Step 5: Initialize RAG system
    rag = MultilingualRAG(vector_store_path=vector_store_path)

    # Step 6: Build the knowledge base (if not already loaded)
    if not rag.vector_store.chunks:
        rag.build_knowledge_base(pdf_paths, save_path=vector_store_path)

    # Step 7: Test queries (can be replaced with user input)
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "What is the main theme of the text?",
        "Who are the main characters?"
    ]

    print("✅ Multilingual RAG System is ready!")
    print("=" * 50)

    # Step 8: Interactive mode
    while True:
        user_input = input("\nEnter your question (or 'quit' to exit): ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.strip():
            response = rag.chat(user_input)
            print(f"\nAnswer: {response}")

            # Show retrieved chunks for debugging
            chunks = rag.retrieve_relevant_chunks(user_input, k=3)
            print(f"\nRetrieved {len(chunks)} relevant chunks:")
            for i, (chunk, score) in enumerate(chunks):
                print(f"{i+1}. Score: {score:.3f} | {chunk.text[:100]}...")

    # Step 9: Show system stats
    stats = rag.get_stats()
    print(f"\n📊 System Statistics: {stats}")

if __name__ == "__main__":
    main()
