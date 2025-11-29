import os
import time
import shutil
import threading
import pickle
import socket
import face_recognition
import numpy as np
from flask import Flask, request, render_template
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC support (iPhone Photos)
register_heif_opener()

# --- CONFIGURATION ---
RAW_FOLDER = 'raw_photos'
PROCESSED_FOLDER = 'static'
DB_FILE = 'face_db.pkl'
# We allow these, but we ignore DNG/RAW in logic to prevent lag
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'}

app = Flask(__name__)

# --- DATABASE MANAGEMENT ---
def load_database():
    if os.path.exists(DB_FILE):
        print("[SYSTEM] Loading existing face database...")
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def save_database():
    with open(DB_FILE, 'wb') as f:
        pickle.dump(face_database, f)
    # print("[SYSTEM] Database saved.") # Commented out to reduce noise

face_database = load_database()

# --- STEP 1: THE AI PROCESSOR ---
def process_new_photo(file_path):
    global face_database
    filename = os.path.basename(file_path)
    
    # 1. CRITICAL: Ignore Temp files and RAW files (Speed Protection)
    if filename.endswith('.crdownload') or filename.endswith('.tmp') or filename.startswith('.'):
        return
    
    # 2. RAW FILTER: If photographer dumps DNG/ARW, ignore them. Use JPG only.
    if filename.lower().endswith(('.dng', '.arw', '.cr2', '.nef')):
        print(f"[SKIPPED] RAW file ignored (Use JPG for speed): {filename}")
        return

    target_path = os.path.join(PROCESSED_FOLDER, filename)
    
    # Wait for file lock to release (Photographer transfer delay)
    time.sleep(1)
    
    try:
        # Move file to static folder
        shutil.move(file_path, target_path)
        print(f"[PROCESSING] New photo detected: {filename}")

        # 3. FORCE CONVERT TO RGB (Fixes WebP/PNG/HEIC issues)
        pil_image = Image.open(target_path)
        pil_image = pil_image.convert("RGB") 
        image_array = np.array(pil_image)
        
        # Detect Faces
        encodings = face_recognition.face_encodings(image_array)

        if len(encodings) > 0:
            for encoding in encodings:
                face_database.append({'encoding': encoding, 'path': f"static/{filename}"})
            save_database()
            print(f"[SUCCESS] Found {len(encodings)} faces in {filename}")
        else:
            print(f"[SKIPPED] No faces found in {filename}")

    except Exception as e:
        print(f"[ERROR] Could not process {filename}: {e}")

# --- STEP 2: FOLDER MONITORING ---
class NewPhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            threading.Thread(target=process_new_photo, args=(event.src_path,)).start()

def start_monitoring():
    observer = Observer()
    observer.schedule(NewPhotoHandler(), path=RAW_FOLDER, recursive=False)
    observer.start()

# --- STEP 3: THE WEB SERVER ---
@app.route('/', methods=['GET', 'POST'])
def home():
    matches = []
    is_search = False  # Track if a search actually happened

    if request.method == 'POST':
        is_search = True
        
        if 'file' not in request.files: return "No file uploaded"
        file = request.files['file']
        if file.filename == '': return "No file selected"

        try:
            # Load User Selfie & Force RGB
            pil_image = Image.open(file)
            pil_image = pil_image.convert("RGB")
            selfie_image = np.array(pil_image)
            
            selfie_encodings = face_recognition.face_encodings(selfie_image)
            
            if len(selfie_encodings) > 0:
                user_face = selfie_encodings[0]
                for entry in face_database:
                    # Tolerance 0.5 = High Accuracy (Few false positives)
                    # Tolerance 0.6 = Loose (More matches, potential wrong ones)
                    match = face_recognition.compare_faces([entry['encoding']], user_face, tolerance=0.5)
                    if match[0]: matches.append(entry['path'])
                matches = list(set(matches))
        except Exception as e:
            print(f"Error processing selfie: {e}")
            
    return render_template('index.html', matched_photos=matches, searched=is_search)

# --- STEP 4: AUTO-DETECT IP ADDRESS ---
def get_ip_address():
    try:
        # Connect to a public DNS to find our own IP on the network
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Start Monitoring
    start_monitoring()
    
    # Get the Local Hotspot IP
    my_ip = get_ip_address()
    
    print("\n" + "="*50)
    print(f"🚀 WEDDING AI SERVER STARTED!")
    print(f"📡 1. Turn on Mobile Hotspot on this laptop.")
    print(f"🔗 2. YOUR URL FOR QR CODE: http://{my_ip}:5000")
    print("="*50 + "\n")
    
    # Run Server
    app.run(host='0.0.0.0', port=5000, debug=False)