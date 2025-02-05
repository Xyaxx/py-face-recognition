import cv2
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
running = False
cap = None

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

# Create GUI window
root = tk.Tk()
root.title("Face Detection with OpenCV")
root.geometry("640x480")

# Create start & stop buttons
btn_start = tk.Button(root, text="Start Detection", command=lambda: threading.Thread(target=start_detection).start())
btn_start.pack()

btn_stop = tk.Button(root, text="Stop Detection", command=stop_detection)
btn_stop.pack()

# Create canvas to display video feed
canvas = tk.Canvas(root, width=640, height=400)
canvas.pack()

# Run the GUI
root.mainloop()
