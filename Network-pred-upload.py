import tkinter as tk
from tkinter import filedialog, ttk
from scapy.all import rdpcap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Function to handle file selection and feature extraction
def select_pcap_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        label_file.config(text=f"Selected File: {filepath}")
        process_pcap(filepath)

# Function to process PCAP file and train the model
def process_pcap(filepath):
    packets = rdpcap(filepath)
    
    # Extract timestamp and packet length from the PCAP file
    data = []
    for pkt in packets:
        if hasattr(pkt, 'time'):
            pkt_time = pkt.time
        else:
            pkt_time = None
        pkt_length = len(pkt)
        data.append([pkt_time, pkt_length])
    
    df = pd.DataFrame(data, columns=['timestamp', 'length'])
    print("Raw timestamps from PCAP file:")
    print(df['timestamp'].head(10))

    # Convert the timestamp to numeric first
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # Convert numeric timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Check if the DataFrame is empty before proceeding
    if df.empty:
        print("Timestamp conversion resulted in an empty DataFrame.")
        return
    print("Timestamp conversion succeeded.")

    # Resample the data for traffic prediction using lowercase 's'
    df.set_index('timestamp', inplace=True)
    df_resampled = df['length'].resample('1s').sum().fillna(0)
    df_resampled = df_resampled.reset_index()
    df_resampled['previous_traffic'] = df_resampled['length'].shift(1).fillna(0)

    # Check if resampling worked
    if df_resampled.empty:
        print("Resampling resulted in an empty DataFrame.")
        return

    print(f"Number of samples: {len(df_resampled)}")  # Check the number of rows

    # Prepare data for model
    X = df_resampled[['previous_traffic']]
    y = df_resampled['length']
    
    # Ensure there are enough samples for splitting
    if len(X) < 2:
        print("Not enough data for train_test_split.")
        return
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if training set has samples
    if len(X_train) == 0 or len(y_train) == 0:
        print("Training set is empty.")
        return

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    label_mse.config(text=f"Mean Squared Error: {mse:.4f}")
    
    # Plot results
    plot_traffic_movement(df_resampled, y_test, y_pred)


# Function to plot the traffic movement
def plot_traffic_movement(df_resampled, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot actual traffic
    ax1.plot(df_resampled['timestamp'], df_resampled['length'], label='Actual Traffic', color='blue')
    ax1.set_title('Actual Network Traffic')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Traffic Length')
    ax1.legend()
    ax1.grid()

    # Plot predicted traffic
    ax2.plot(df_resampled['timestamp'].iloc[-len(y_test):], y_test, label='Actual Traffic', color='blue')
    ax2.plot(df_resampled['timestamp'].iloc[-len(y_test):], y_pred, label='Predicted Traffic', linestyle='--', color='orange')
    ax2.set_title('Predicted Network Traffic')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Traffic Length')
    ax2.legend()
    ax2.grid()

    # Embed the plot in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def perform_anomaly_detection(y_test, y_pred, threshold=0.3):
    anomalies = []
    for actual, predicted in zip(y_test, y_pred):
        if actual == 0:
            # Avoid division by zero, consider it an anomaly if actual traffic is zero
            if predicted != 0:
                anomalies.append(True)
            else:
                anomalies.append(False)
        else:
            # Normal case for non-zero actual values
            if np.abs((actual - predicted) / actual) > threshold:
                anomalies.append(True)
            else:
                anomalies.append(False)
    
    if any(anomalies):
        label_anomaly.config(text="Anomaly Detected in Traffic!", foreground="red")
    else:
        label_anomaly.config(text="No Anomaly Detected.", foreground="green")


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
window.title("Network Traffic Prediction App")
window.geometry("800x600")

# Create UI elements
label = tk.Label(window, text="Network Traffic Prediction from PCAP", font=('Arial', 16))
label.pack(pady=10)

label_file = tk.Label(window, text="No File Selected", font=('Arial', 12))
label_file.pack(pady=5)

button_select = tk.Button(window, text="Select PCAP File", command=select_pcap_file, font=('Arial', 12))
button_select.pack(pady=10)

label_mse = tk.Label(window, text="Mean Squared Error: N/A", font=('Arial', 12))
label_mse.pack(pady=10)

label_anomaly = ttk.Label(window, text="Anomaly Detection: N/A", font=('Arial', 12))
label_anomaly.pack(pady=5)

# Capacity planning label
label_capacity = ttk.Label(window, text="Suggested Capacity: N/A", font=('Arial', 12))
label_capacity.pack(pady=5)

# Network resource allocation label
label_allocation = ttk.Label(window, text="Network Resource Allocation: N/A", font=('Arial', 12))
label_allocation.pack(pady=5)

# Run the Tkinter event loop
window.mainloop()
