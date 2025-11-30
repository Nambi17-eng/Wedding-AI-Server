import os
import time
import threading
import pickle
import socket
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
INPUT_FOLDER = 'incoming_photos'
DB_FILE = 'database.pkl'

# VGG-Face is the "AK-47" of Face Recognition. 
# It works in bad lighting, side angles, and groups.
MODEL_NAME = "VGG-Face" 

app = Flask(__name__)

# --- DATABASE SYSTEM ---
face_database = []

def load_database():
    global face_database
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                face_database = pickle.load(f)
            print(f"[SYSTEM] Loaded {len(face_database)} photos from memory.")
        except:
            face_database = []

def save_database():
    with open(DB_FILE, 'wb') as f:
        pickle.dump(face_database, f)

# --- AI ENGINE ---
def generate_embedding(img_path):
    try:
        # Generate Vector using VGG-Face
        results = DeepFace.represent(
            img_path=img_path, 
            model_name=MODEL_NAME, 
            enforce_detection=True, 
            detector_backend="opencv"
        )
        return results
    except ValueError:
        return None
    except Exception as e:
        print(f"[AI WARNING] Could not process {img_path}: {e}")
        return None

def process_file(filename):
    if filename.startswith('.') or filename.endswith('.tmp'): return
    file_path = os.path.join(INPUT_FOLDER, filename)
    time.sleep(1)

    print(f"[PROCESSING] {filename}...")
    results = generate_embedding(file_path)
    
    if results:
        count = 0
        for face in results:
            face_database.append({
                'path': filename,
                'embedding': face['embedding']
            })
            count += 1
        save_database()
        print(f"[SUCCESS] Added {filename} ({count} faces found)")
    else:
        print(f"[SKIPPED] No face found: {filename}")

# --- FOLDER MONITORING ---
class NewPhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            threading.Thread(target=process_file, args=(filename,)).start()

# --- WEB SERVER ---
@app.route('/', methods=['GET', 'POST'])
def home():
    matches = []
    is_search = False

    if request.method == 'POST':
        is_search = True
        file = request.files.get('file')
        
        if file and file.filename != '':
            print("\n" + "="*30)
            print("📸 NEW SELFIE UPLOADED")
            print("="*30)
            
            temp_path = "temp_selfie.jpg"
            file.save(temp_path)
            
            try:
                # 1. Get Selfie Vector (Allow loose detection for selfies)
                selfie_results = DeepFace.represent(
                    temp_path, 
                    model_name=MODEL_NAME, 
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                
                if selfie_results:
                    # Check every face found in the selfie (usually just 1)
                    for selfie_face in selfie_results:
                        target_vector = selfie_face['embedding']
                        
                        # Compare against EVERY photo in DB
                        for entry in face_database:
                            db_vector = entry['embedding']
                            
                            # SCIPY CALCULATION (Accurate Math)
                            # Lower score = Better match (0.0 is identical, 1.0 is opposite)
                            score = cosine(target_vector, db_vector)
                            
                            # LOGGING: See exactly what's happening
                            # Only print if it's somewhat close to reduce spam
                            if score < 0.6: 
                                print(f"Checking {entry['path']}... Score: {round(score, 3)}")
                            
                            # THRESHOLD FOR VGG-Face
                            # 0.40 is the industry standard for VGG-Face
                            if score < 0.40:
                                print(f"✅ MATCH FOUND! ({entry['path']})")
                                matches.append(entry['path'])
                    
                    matches = list(set(matches))
                    print(f"[RESULT] Returning {len(matches)} photos.")
                else:
                    print("[RESULT] No faces detected in selfie.")

            except Exception as e:
                print(f"Selfie Error: {e}")
            
            if os.path.exists(temp_path): os.remove(temp_path)

    return render_template('index.html', photos=matches, searched=is_search)

@app.route('/photos/<path:filename>')
def serve_hd(filename):
    return send_from_directory(INPUT_FOLDER, filename)

# --- STARTUP ---
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except: return "127.0.0.1"

if __name__ == '__main__':
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    # Pre-load model (First run will download VGG-Face)
    print("--------------------------------------------------")
    print("🚀 DOWNLOADING VGG-FACE MODEL (First Run Only)...")
    print("--------------------------------------------------")
    try: DeepFace.build_model(MODEL_NAME)
    except: pass

    load_database()
    
    # Re-scan clean
    print("[SYSTEM] Scanning existing files...")
    # NOTE: We clear the RAM database on startup to ensure we don't have old ghosts
    face_database = [] 
    for f in os.listdir(INPUT_FOLDER):
        process_file(f)

    observer = Observer()
    observer.schedule(NewPhotoHandler(), path=INPUT_FOLDER, recursive=False)
    observer.start()

    ip = get_ip()
    print("\n" + "="*40)
    print(f"🚀 VGG-FACE SERVER STARTED")
    print(f"🔗 URL: http://{ip}:5000")
    print("="*40 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)