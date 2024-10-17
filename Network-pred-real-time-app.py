import tkinter as tk
from tkinter import filedialog
from scapy.all import sniff
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque
import threading

# Global variables for storing real-time data
packet_data = deque(maxlen=100)  # Stores last 100 packets for real-time monitoring
timestamps = deque(maxlen=100)
packet_lengths = deque(maxlen=100)
monitoring = False  # Flag to control packet monitoring

# Function to handle real-time packet capture
def start_real_time_monitoring():
    global monitoring
    monitoring = True  # Set the flag to True
    try:
        threading.Thread(target=sniff_packets).start()
    except Exception as e:
        print(f"Error starting real-time monitoring: {e}")

# Function to stop real-time packet capture
def stop_real_time_monitoring():
    global monitoring
    monitoring = False  # Set the flag to False
    print("STOP")

# Function to capture live network traffic in real-time
def sniff_packets():
    # Sniff packets while monitoring is True
    global monitoring
    try:
        # Sniff packets while monitoring is True
        while monitoring:
            sniff(prn=process_real_time_packet, store=False, timeout=1)  # Set a small timeout for responsiveness
    except Exception as e:
        print(f"Error in packet sniffing: {e}")

# Callback function to process each packet in real-time
def process_real_time_packet(pkt):
    if not monitoring:  # Check if monitoring is stopped
        return
    
    try:
        if hasattr(pkt, 'time'):
            pkt_time = pkt.time
            pkt_length = len(pkt)
            
            # Append to real-time data queues
            packet_data.append([pkt_time, pkt_length])
            timestamps.append(pkt_time)
            packet_lengths.append(pkt_length)
            
            # Process packets for real-time traffic prediction every 10 packets
            if len(packet_data) >= 10:
                process_real_time_data()
    except Exception as e:
        print(f"Error processing packet: {e}")


# Function to process real-time packet data and update the model
def process_real_time_data():
    df = pd.DataFrame(list(packet_data), columns=['timestamp', 'length'])
    
    # Convert timestamps
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    # Resample for 1-second intervals
    df.set_index('timestamp', inplace=True)
    df_resampled = df['length'].resample('1s').sum().fillna(0)
    df_resampled = df_resampled.reset_index()
    df_resampled['previous_traffic'] = df_resampled['length'].shift(1).fillna(0)
    
    # Prepare data for model
    X = df_resampled[['previous_traffic']]
    y = df_resampled['length']
    
    # Check if there is enough data to train
    if len(X) < 2:
        return
    
    # Train the model in real-time
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future traffic
    y_pred = model.predict(X)

    # Update the GUI safely
    window.after(0, update_gui, y, y_pred)

# Function to update the GUI
def update_gui(y_test, y_pred):
    try:
        plot_results(y_test, y_pred)
        perform_anomaly_detection(y_test, y_pred)
        suggest_capacity_planning(y_pred)
        suggest_network_allocation(y_pred)
    except Exception as e:
        print(f"Error updating GUI: {e}")

# Function to plot the results in real-time
def plot_results(y_test, y_pred):
    # Clear the current figures and create new subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Plot Actual Traffic
    ax1.clear()
    ax1.plot(y_test.values, label='Actual Traffic', color='blue')
    ax1.set_title('Actual Network Traffic')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Traffic (units)')
    ax1.legend()

    # Plot Predicted Traffic
    ax2.clear()
    ax2.plot(y_pred, label='Predicted Traffic', linestyle='--', color='orange')
    ax2.set_title('Predicted Network Traffic')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Traffic (units)')
    ax2.legend()

    # Embed the plot in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function for Anomaly Detection
def perform_anomaly_detection(y_test, y_pred, threshold=0.3):
    anomalies = []
    for actual, predicted in zip(y_test, y_pred):
        if actual != 0 and np.abs((actual - predicted) / actual) > threshold:
            anomalies.append(True)
        else:
            anomalies.append(False)
    
    if any(anomalies):
        label_anomaly.config(text="Anomaly Detected in Traffic!")
    else:
        label_anomaly.config(text="No Anomaly Detected.")

# Function for Capacity Planning
def suggest_capacity_planning(y_pred):
    max_traffic = np.max(y_pred)
    suggested_capacity = max_traffic * 1.2  # 20% buffer for capacity planning
    label_capacity.config(text=f"Suggested Capacity: {suggested_capacity:.2f} units")

# Function for Network Resource Allocation
def suggest_network_allocation(y_pred):
    avg_traffic = np.mean(y_pred)
    if avg_traffic < 100:
        allocation = "Low Allocation"
    elif avg_traffic < 500:
        allocation = "Medium Allocation"
    else:
        allocation = "High Allocation"
    
    label_allocation.config(text=f"Network Resource Allocation: {allocation}")

# Create main application window
window = tk.Tk()
window.title("Real-time Network Traffic Prediction App")
window.geometry("800x600")

# Create UI elements
label = tk.Label(window, text="Real-time Network Traffic Prediction", font=('Arial', 16))
label.pack(pady=10)

label_file = tk.Label(window, text="Real-time Monitoring Enabled", font=('Arial', 12))
label_file.pack(pady=5)

button_start = tk.Button(window, text="Start Real-time Monitoring", command=start_real_time_monitoring, font=('Arial', 12))
button_start.pack(pady=10)

# Stop button to halt monitoring
button_stop = tk.Button(window, text="Stop Real-time Monitoring", command=stop_real_time_monitoring, font=('Arial', 12))
button_stop.pack(pady=10)

label_mse = tk.Label(window, text="Mean Squared Error: N/A", font=('Arial', 12))
label_mse.pack(pady=10)

# Anomaly detection label
label_anomaly = tk.Label(window, text="Anomaly Detection: N/A", font=('Arial', 12))
label_anomaly.pack(pady=10)

# Capacity planning label
label_capacity = tk.Label(window, text="Suggested Capacity: N/A", font=('Arial', 12))
label_capacity.pack(pady=10)

# Network resource allocation label
label_allocation = tk.Label(window, text="Network Resource Allocation: N/A", font=('Arial', 12))
label_allocation.pack(pady=10)

# Run the Tkinter event loop
try:
    window.mainloop()
except Exception as e:
    print(f"Error running the application: {e}")
