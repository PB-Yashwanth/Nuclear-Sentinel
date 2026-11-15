from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import uvicorn

# IMPORT YOUR NEW LANGGRAPH AGENT & DB
from triage_agent import triage_app, init_db

# --- 1. DEFINE PYTORCH ARCHITECTURE ---
class AnomalyLSTM(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2):
        super(AnomalyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :] 
        return self.sigmoid(self.fc(final_state))

# --- 2. INITIALIZE APP, DB & LOAD MODEL ---
app = FastAPI(title="NuclearSentinel API", version="1.0")

print("Initializing SQLite Alert Database...")
init_db()

print("Loading LSTM model weights...")
DEVICE = torch.device('cpu')
model = AnomalyLSTM().to(DEVICE)
try:
    model.load_state_dict(torch.load("lstm_anomaly_model.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 3. DATA CONTRACT ---
class SensorPayload(BaseModel):
    sequence: list[list[float]]

# --- 4. PREDICTION & TRIAGE ENDPOINT ---
@app.post("/api/v1/predict")
def predict_anomaly(payload: SensorPayload):
    try:
        input_tensor = torch.FloatTensor([payload.sequence]).to(DEVICE)
        
        if input_tensor.shape != (1, 15, 21):
            raise ValueError(f"Invalid shape. Expected (1, 15, 21), got {input_tensor.shape}")

        with torch.no_grad():
            prediction = model(input_tensor).item()

        alert_status = "NORMAL"
        if prediction > 0.85:
            alert_status = "CRITICAL (High-High) - Trip Imminent"
        elif prediction > 0.60:
            alert_status = "WARNING (High) - Degradation Detected"

        # --- LANGGRAPH TRIGGER ---
        report = None
        if alert_status != "NORMAL":
            # If an anomaly is detected, run the LangGraph workflow
            state_input = {"alert_tier": alert_status, "probability": prediction}
            agent_result = triage_app.invoke(state_input)
            report = agent_result["final_report"]

        return {
            "status": "success",
            "anomaly_probability": round(prediction, 4),
            "alert_tier": alert_status,
            "maintenance_report": report
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)