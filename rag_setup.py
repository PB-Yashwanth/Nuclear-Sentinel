import os
import sys
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

print("System Check: Validating Google API Key...")
google_api_key = os.environ.get("GOOGLE_API_KEY") 
if not google_api_key:
    print("❌ FATAL ERROR: GOOGLE_API_KEY is missing.")
    sys.exit(1)

print("Connecting to Google Gemini GenAI API...")
# Using Google's new unified SDK and active model
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2-preview", 
    api_key=google_api_key
)
Settings.llm = None 

print("Initializing ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("maintenance_manuals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Reading manuals from ./data/manuals/ ...")
documents = SimpleDirectoryReader("./data/manuals").load_data()

print("Indexing documents into Vector Database...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

print("✅ RAG Database Built Successfully! Vectors saved to ./chroma_db/")