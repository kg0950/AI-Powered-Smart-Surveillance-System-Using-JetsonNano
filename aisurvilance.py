import cv2
import torch
import threading
import time
from flask import Flask, Response

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize Flask app
app = Flask(__name__)

# Global variable for video frame
global_frame = None

def detect_objects():
    """Capture video, apply YOLOv5 object detection, and update the global frame."""
    global global_frame
    cap = cv2.VideoCapture(0)  # Use CSI or USB Camera

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Object Detection
        results = model(frame)
        for det in results.xyxy[0]:  # Iterate over detected objects
            x1, y1, x2, y2, conf, cls = det
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        global_frame = frame

    cap.release()

def generate_frames():
    """Flask video streaming function."""
    global global_frame
    while True:
        if global_frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', global_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page."""
    return "<h1>Jetson Nano AI Surveillance</h1><img src='/video_feed' width='50%'>"

if __name__ == '__main__':
    # Start object detection in a separate thread
    threading.Thread(target=detect_objects, daemon=True).start()
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
