import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from collections import defaultdict
import time

app = Flask(__name__)

# --- CONFIGURATION ---
# Use 0 for Webcam, or replace with 'video.mp4' for a file
VIDEO_SOURCE = 0 
model = YOLO('yolov8n.pt')  # Loads the lightweight YOLOv8 model

# --- STATE VARIABLES ---
track_history = defaultdict(lambda: [])
dwell_time = {} # Store start time for each ID
completed_dwell_times = [] # Store duration of people who left
counts = {"up": 0, "down": 0}
heatmap_accumulator = None # Will be initialized based on frame size

# Line position for counting (0-1 scale relative to height)
LINE_POSITION = 0.6 

def process_video():
    global heatmap_accumulator
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize heatmap (black image)
    heatmap_accumulator = np.zeros((h, w), dtype=np.float32)
    
    # Line coordinates
    line_y = int(h * LINE_POSITION)
    
    # IDs that have crossed the line
    crossed_ids = set()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 tracking
        # persist=True keeps track of IDs between frames
        results = model.track(frame, persist=True, classes=[0], verbose=False) # class 0 is Person

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w_box, h_box = box
                center_x, center_y = int(x), int(y)
                
                # --- 1. DWELL TIME LOGIC ---
                if track_id not in dwell_time:
                    dwell_time[track_id] = time.time()
                
                # --- 2. HEATMAP LOGIC ---
                # Add intensity to the accumulator at the center point
                # We add a Gaussian blob or simple circle
                try:
                    # Simple efficient way: add value to a slice
                    radius = 15
                    y_slice = slice(max(0, center_y-radius), min(h, center_y+radius))
                    x_slice = slice(max(0, center_x-radius), min(w, center_x+radius))
                    heatmap_accumulator[y_slice, x_slice] += 2.0 # Increment heat
                except Exception:
                    pass

                # --- 3. COUNTING LOGIC (Line Crossing) ---
                # Logic: Check previous position vs current position relative to line
                # Simplified for this demo: Check if close to line and direction
                if track_id not in crossed_ids:
                    if (line_y - 10) < center_y < (line_y + 10):
                        counts['down'] += 1 # Assuming moving down
                        crossed_ids.add(track_id)
                        
                        # Record dwell time when they "leave" (cross line)
                        duration = time.time() - dwell_time[track_id]
                        completed_dwell_times.append(duration)
                        # Keep list short
                        if len(completed_dwell_times) > 50: 
                            completed_dwell_times.pop(0)

                # Visualization: Draw Box & Center
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        # --- VISUALIZATION LAYERS ---
        
        # 1. Draw the Counting Line
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Line", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 2. Apply Heatmap Overlay
        # Normalize accumulator to 0-255
        heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend original frame with heatmap (Weight: 0.6 Frame, 0.4 Heatmap)
        frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

        # Encode frame to JPG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame for web streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    # Calculate average dwell time
    avg_dwell = 0
    if completed_dwell_times:
        avg_dwell = sum(completed_dwell_times) / len(completed_dwell_times)
    
    # Current active people
    current_people = len(dwell_time) # Approximation for demo
    
    return jsonify({
        'counts': counts,
        'avg_dwell_time': round(avg_dwell, 1),
        'current_people': current_people,
        'recent_dwells': completed_dwell_times[-10:] # Last 10 records for charts
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)