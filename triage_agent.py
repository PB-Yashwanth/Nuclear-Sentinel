import os
import sys
import sqlite3
import chromadb
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ==========================================
# 1. ENVIRONMENT & EMBEDDING CONFIGURATION
# ==========================================
print("System Check: Initializing Triage Agent Environment...")
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    print("❌ FATAL ERROR: GOOGLE_API_KEY environment variable is missing.")
    sys.exit(1)

# Route all vector embeddings through the free Gemini Developer API
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2-preview", 
    api_key=google_api_key
)
Settings.llm = None  # Using RAG purely for retrieval, LLM disabled

# ==========================================
# 2. CHROMADB VECTOR DATABASE CONNECTION
# ==========================================
try:
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("maintenance_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from the pre-built local database
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=3)
    print("✅ RAG Database Connected Successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Failed to connect to ChromaDB. Ensure rag_setup.py was run first. Error: {e}")
    sys.exit(1)


# ==========================================
# 3. SQLITE DATABASE INITIALIZATION (FIXED)
# ==========================================
def init_db():
    """
    Initializes the SQLite database schema for tracking sensor alerts 
    and automated maintenance logs.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("nuclearsentinel.db")
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        engine_id INTEGER,
        cycle INTEGER,
        anomaly_score REAL,
        severity TEXT,
        sensor_readings TEXT,
        image_path TEXT
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id INTEGER,
        recommendation TEXT,
        classification TEXT,
        status TEXT DEFAULT 'open',
        created_at TEXT,
        FOREIGN KEY (alert_id) REFERENCES sensor_alerts(id)
    );
    """)
    
    conn.commit()
    conn.close()
    print("✅ SQLite Alert Database Initialized Successfully.")


# ==========================================
# 4. LANGGRAPH STATE & WORKFLOW NODES
# ==========================================
class AgentState(TypedDict):
    sensor_data: Dict[str, Any]
    anomaly_probability: float
    retrieved_manuals: List[str]
    work_order_report: str

def logger_node(state: AgentState):
    """Logs the anomaly event and incoming telemetry."""
    return state

def retrieval_node(state: AgentState):
    """Queries ChromaDB for maintenance manuals based on sensor alerts."""
    search_query = "Critical anomaly detected in machinery. Retrieve maintenance procedures and troubleshooting steps."
    
    # Fetch relevant chunks from the database
    docs = retriever.retrieve(search_query)
    retrieved_texts = [doc.node.text for doc in docs]
    
    return {"retrieved_manuals": retrieved_texts}

def drafter_node(state: AgentState):
    """Compiles the sensor data and retrieved manuals into a work order."""
    manuals = "\n\n".join(state.get("retrieved_manuals", []))
    prob = state.get("anomaly_probability", 0.0)
    
    report = f"URGENT: Anomaly detected with {prob*100:.2f}% probability.\n\n"
    report += f"Recommended Standard Operating Procedures:\n{manuals}\n"
    
    return {"work_order_report": report}

# ==========================================
# 5. LANGGRAPH ORCHESTRATION BUILDER
# ==========================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("logger", logger_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("drafter", drafter_node)

# Define execution flow
workflow.set_entry_point("logger")
workflow.add_edge("logger", "retrieval")
workflow.add_edge("retrieval", "drafter")
workflow.add_edge("drafter", END)

# Compile the final agent to be called by your FastAPI backend
triage_app = workflow.compile()