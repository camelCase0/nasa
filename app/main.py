import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import butter, filtfilt
import uuid
from datetime import datetime, timedelta
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Дозволяємо конкретні джерела
    allow_credentials=True,
    allow_methods=["*"],  # Дозволяємо всі методи (GET, POST, тощо)
    allow_headers=["*"],  # Дозволяємо всі заголовки
)

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Utility functions for seismic processing (from your original code)
def load_data(filename):
    st = read(filename)
    return st[0]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_probability(amplitude, threshold):
    if amplitude >= threshold:
        return 100
    elif amplitude >= 0.85 * threshold:
        return 90
    elif amplitude >= 0.7 * threshold:
        return 70
    elif amplitude >= 0.5 * threshold:
        return 50
    else:
        return 0

def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def merge_activities(probabilities, sampling_rate, min_gap_seconds=1):
    min_gap_samples = min_gap_seconds * sampling_rate
    activity_indices = np.where(probabilities > 0)[0]
    if len(activity_indices) == 0:
        return []
    segments = []
    current_segment = [activity_indices[0]]
    for i in range(1, len(activity_indices)):
        if activity_indices[i] - activity_indices[i - 1] <= min_gap_samples and probabilities[activity_indices[i]] > 50:
            current_segment.append(activity_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [activity_indices[i]]
    if current_segment:
        segments.append(current_segment)
    return segments

def random_date(start_date= datetime(2022, 1, 1), end_date= datetime(2024, 9, 30) ):
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randint(0, days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

    # Define the start and end dates

def save_activity(signal, probabilities, start_idx, end_idx, sampling_rate, now):
    activity_time = np.linspace(start_idx / sampling_rate, end_idx / sampling_rate, end_idx - start_idx + 1)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(activity_time, signal[start_idx:end_idx + 1], color='red')
    plt.title(f'Seismic Activity from {start_idx / sampling_rate:.2f}s to {end_idx / sampling_rate:.2f}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (cm/s)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(activity_time, probabilities[start_idx:end_idx + 1], color='blue')
    plt.title(f'Probability of Seismic Activity')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability (%)')
    plt.grid()

    plt.tight_layout()
    file_name = f"static/images/activity_{now}.png"
    plt.savefig(file_name)
    plt.close()
    return file_name

def save_full_signal_and_probabilities(t, signal, probabilities,now):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, color='blue')
    plt.title('Full Seismic Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (cm/s)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, probabilities, color='red')
    plt.title('Probability of Seismic Activity')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability (%)')
    plt.grid()
    
    plt.tight_layout()
    file_name = f"static/images/full_signal_{now}.png"
    plt.savefig(file_name)
    plt.close()
    return file_name

# FastAPI endpoints
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload an .mseed file and process it."""
    if not file.filename.endswith(".mseed"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .mseed is accepted.")
    
    file_location = f"uploads/{uuid.uuid4()}.mseed"
    
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())
    
    # Process the seismic data
    tr = load_data(file_location)
    sampling_rate = tr.stats.sampling_rate
    n = len(tr.data)
    t = np.linspace(0, n / sampling_rate, n, endpoint=False)

    constant_amplitude_threshold = 0.07 + 0.1 * 0.3
    filtered_signal = lowpass_filter(tr.data, cutoff=1.0, fs=sampling_rate)
    probabilities = np.array([calculate_probability(np.abs(amp), constant_amplitude_threshold) for amp in filtered_signal])
    smoothed_probabilities = smooth_data(probabilities, window_size=500)
    segments = merge_activities(smoothed_probabilities, sampling_rate)
    now = random_date().strftime("%Y-%m-%d")
    # Save full signal and probabilities
    full_signal_file = save_full_signal_and_probabilities(t, filtered_signal, smoothed_probabilities,now)

    # Save individual activities
    activities = []
    if segments:
        for activity_number, segment in enumerate(segments, start=1):
            start_idx = segment[0]
            end_idx = segment[-1]
            duration = (end_idx - start_idx + 1) / sampling_rate
            if duration >= 0.5:
                activity_file = save_activity(filtered_signal, smoothed_probabilities, start_idx, end_idx, sampling_rate, now)
                activities.append(activity_file)
    
    return {"message": "File processed successfully", "activities": activities, "full_signal": full_signal_file}

@app.get("/images/")
async def get_images():
    """Get a list of all generated images."""
    images = os.listdir("static/images")
    return {"images": images}

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve an image by its name."""
    image_path = f"static/images/{image_name}"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.delete("/images/")
async def delete_all_images():
    """Delete all images in the directory."""
    image_dir = "static/images"
    try:
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return {"message": "All images have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))