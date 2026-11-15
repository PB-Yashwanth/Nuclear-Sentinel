# src/triage_agent.py
import sqlite3
import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- NEW RAG IMPORTS ---
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# --- INITIALIZE EMBEDDING MODEL ---
# This runs once when the API starts so it doesn't slow down the agent during an alert
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# --- 1. SQLITE DATABASE SETUP ---
def init_db():
    """Initializes the production alert database."""
    conn = sqlite3.connect('nuclearsentinel.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_tier TEXT,
            probability REAL
        )
    ''')
    conn.commit()
    return conn

# --- 2. LANGGRAPH STATE ---
class AgentState(TypedDict):
    alert_tier: str
    probability: float
    retrieved_manual: str
    final_report: str

# --- 3. AGENT NODES (Functions) ---
def log_to_database(state: AgentState):
    """Node 1: Logs the anomaly to SQLite for auditability."""
    conn = sqlite3.connect('nuclearsentinel.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alert_history (timestamp, alert_tier, probability) VALUES (?, ?, ?)",
        (datetime.datetime.now().isoformat(), state["alert_tier"], state["probability"])
    )
    conn.commit()
    return state

def retrieve_maintenance_docs(state: AgentState):
    """Node 2: True RAG retrieval from ChromaDB."""
    try:
        # Connect to the local vector database
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("maintenance_manuals")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Search the database for the specific anomaly
        retriever = index.as_retriever(similarity_top_k=1)
        query = f"What is the maintenance procedure for a {state['alert_tier']} alert in the PHT sensor?"
        nodes = retriever.retrieve(query)
        
        if nodes:
            doc = nodes[0].text
        else:
            doc = "Manual intervention required. No specific document found in database."
            
    except Exception as e:
        doc = f"RAG Retrieval Offline: {str(e)}"
        
    return {"retrieved_manual": doc}

def draft_report(state: AgentState):
    """Node 3: Compiles the final diagnostic report."""
    report = f"""
    🛑 AUTOMATED MAINTENANCE WORK ORDER
    -----------------------------------
    Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Status: {state['alert_tier']} (Confidence: {state['probability']:.2%})
    
    Recommended Action (via RAG Knowledge Base):
    {state['retrieved_manual']}
    
    Routing To: Shift Charge Engineer
    """
    return {"final_report": report}

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("logger", log_to_database)
workflow.add_node("rag_retrieval", retrieve_maintenance_docs)
workflow.add_node("drafter", draft_report)

# Define edges
workflow.set_entry_point("logger")
workflow.add_edge("logger", "rag_retrieval")
workflow.add_edge("rag_retrieval", "drafter")
workflow.add_edge("drafter", END)

# Compile the agent
triage_app = workflow.compile()

# --- 5. TEST THE AGENT ---
if __name__ == "__main__":
    init_db()
    print("Database initialized.")
    test_state = {"alert_tier": "CRITICAL (High-High) - Trip Imminent", "probability": 0.92}
    print("\nExecuting True RAG LangGraph Agent Workflow...")
    result = triage_app.invoke(test_state)
    print("\n" + result["final_report"])