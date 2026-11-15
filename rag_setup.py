import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding

print("Connecting to HuggingFace Inference API...")
# Safely pull the token from Render's environment variables
hf_token = os.environ.get("HF_TOKEN") 

Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    token=hf_token
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