import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def generate_alert_image(sensor_df: pd.DataFrame, engine_id: int, cycle: int, severity: str) -> str:
    """
    Generates a high-contrast sensor telemetry plot, uses OpenCV to overlay 
    an engineering alert bounding box and warning text, and saves it to disk.
    """
    os.makedirs("output/alerts", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/alerts/engine_{engine_id}_cycle_{cycle}_{timestamp_str}.png"

    # 1. Plot Telemetry Trends using Matplotlib
    plt.figure(figsize=(8, 4))
    if 'sensor_2' in sensor_df.columns and 'sensor_3' in sensor_df.columns:
        plt.plot(sensor_df['cycle'], sensor_df['sensor_2'], label='Sensor 2 (HPC Temp)', color='blue', linewidth=1.5)
        plt.plot(sensor_df['cycle'], sensor_df['sensor_3'], label='Sensor 3 (HPC Press)', color='orange', linewidth=1.5)
    else:
        plt.plot(sensor_df.index, sensor_df.iloc[:, 0], label='Telemetry Trend A', color='blue')
        
    plt.title(f"Telemetry Degradation Alert - Engine #{engine_id} (Cycle {cycle})", fontsize=10, fontweight='bold')
    plt.xlabel("Operating Cycles", fontsize=9)
    plt.ylabel("Normalized Sensor Values", fontsize=9)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    temp_fig_path = "output/temp_plot.png"
    plt.savefig(temp_fig_path, dpi=150)
    plt.close()

    # 2. Load into OpenCV for Computer Vision Post-Processing / Annotation
    img = cv2.imread(temp_fig_path)
    
    h, w, _ = img.shape
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)

    cv2.putText(img, f"STATUS: {severity.upper()}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    
    if os.path.exists(temp_fig_path):
        os.remove(temp_fig_path)

    print(f"🖼️ OpenCV Anomaly Chart Generated: {output_path}")
    return output_path