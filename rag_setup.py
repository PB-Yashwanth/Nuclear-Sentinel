# rag_setup.py
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

print("Loading local HuggingFace embedding model...")
# Using a fast, free local embedding model that runs perfectly on laptop CPUs
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None # We only need retrieval, not generation

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