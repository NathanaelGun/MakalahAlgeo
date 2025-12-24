import requests
import time
import pandas as pd
from datetime import datetime

# Konfigurasi
PARK_ID = 64
URL = f"https://queue-times.com/parks/{PARK_ID}/queue_times.json"
FILE_NAME = "data_antrean_5_wahana.csv"
INTERVAL_SECONDS = 600 # 10 menit

# List wahana target (Sesuaikan dengan nama persis di API)
TARGET_RIDES = [
    "Hagrid's Magical Creatures Motorbike Adventure™",
    "Jurassic World VelociCoaster",
    "The Incredible Hulk Coaster®",
    "The Amazing Adventures of Spider-Man®",
    "Jurassic Park River Adventure™"
]

def fetch_5_rides():
    try:
        response = requests.get(URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        extracted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []
        
        for land in data['lands']:
            for ride in land['rides']:
                if ride['name'] in TARGET_RIDES:
                    records.append({
                        "timestamp": extracted_at,
                        "ride_name": ride['name'],
                        "wait_time": ride['wait_time'],
                        "status": "Open" if ride['is_open'] else "Closed"
                    })
        return records
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gangguan: {e}")
        return None

print(f"Memulai pengambilan data untuk 5 wahana utama...")

while True:
    data_points = fetch_5_rides()
    if data_points:
        df_new = pd.DataFrame(data_points)
        header_needed = not pd.io.common.file_exists(FILE_NAME)
        df_new.to_csv(FILE_NAME, mode='a', index=False, header=header_needed)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data 5 wahana tersimpan.")
    
    time.sleep(INTERVAL_SECONDS)