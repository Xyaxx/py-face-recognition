import cv2
import numpy as np
import sqlite3
import threading
import ctypes
import tkinter as tk
from PIL import Image, ImageTk

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global variables
running = False
cap = None

# Database setup
conn = sqlite3.connect("faces.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, name TEXT, face BLOB)")
conn.commit()

# GUI setup
root = tk.Tk()
root.title("Face Recognition with OpenCV & SQLite © 2025")
root.geometry("1000x700")


# Load the icon image using PIL
icon = Image.open("app.ico")
icon = ImageTk.PhotoImage(icon)
 
# Set the taskbar icon
root.iconphoto(True, icon)
	
myappid = u'py-face-detection.v1.0.5'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# Canvas to display video feed
canvas = tk.Canvas(root, width=640, height=400)
canvas.pack()

# Listbox to display saved faces
listbox = tk.Listbox(root, height=5)
listbox.pack(padx=5, pady=5)

# Entry for user name input
name_entry = tk.Entry(root)
name_entry.pack(padx=5, pady=5)

def save_face():
    """Save the detected face in the database."""
    global cap
    if not cap:
        return
    
    ret, frame = cap.read()
    if not ret:
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return
    
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_img, (100, 100))
    
    name = name_entry.get()
    if not name:
        return
    
    # Convert face image to bytes
    _, buffer = cv2.imencode(".jpg", face_resized)
    face_bytes = buffer.tobytes()
    
    cursor.execute("INSERT INTO faces (name, face) VALUES (?, ?)", (name, face_bytes))
    conn.commit()
    listbox.insert(tk.END, name)

def load_faces():
    """Load saved faces from the database."""
    cursor.execute("SELECT name FROM faces")
    rows = cursor.fetchall()
    listbox.delete(0, tk.END)
    for row in rows:
        listbox.insert(tk.END, row[0])

def recognize_face():
    """Recognize the current face from the database."""
    global cap
    if not cap:
        return
    
    ret, frame = cap.read()
    if not ret:
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return
    
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_img, (100, 100))

    # Load all stored faces
    cursor.execute("SELECT id, name, face FROM faces")
    rows = cursor.fetchall()
    
    for row in rows:
        face_id, name, face_data = row
        np_arr = np.frombuffer(face_data, dtype=np.uint8)
        stored_face = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        # Compute similarity
        diff = np.abs(face_resized.astype("float") - stored_face.astype("float")).sum()
        print(diff)
        if diff < 150000:  # Adjust threshold if needed
            print(f"Recognized: {name}")
            return

    print("Face not recognized")

def start_detection():
    """Start webcam feed and detect faces."""
    global running, cap
    running = True
    cap = cv2.VideoCapture(0)  # Open webcam
    update_frame()

def stop_detection():
    """Stop the webcam feed."""
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    canvas.delete("all")

def update_frame():
    """Capture frames and detect faces in real-time."""
    if not running:
        return
    
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert frame to display in Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img_tk = img_tk

    # Repeat update every 30ms
    root.after(30, update_frame)

# Buttons
btn_start = tk.Button(root, text="Start Detection", command=lambda: threading.Thread(target=start_detection).start())
btn_start.pack(padx=5, pady=5)

btn_stop = tk.Button(root, text="Stop Detection", command=stop_detection)
btn_stop.pack(padx=5, pady=5)

btn_save = tk.Button(root, text="Save Face", command=save_face)
btn_save.pack(padx=5, pady=5)

btn_recognize = tk.Button(root, text="Recognize Face", command=recognize_face)
btn_recognize.pack(padx=5, pady=5)

# Load existing faces from DB
load_faces()

# Run the GUI
root.mainloop()
 