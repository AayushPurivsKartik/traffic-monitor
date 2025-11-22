End-to-End Traffic Analytics Dashboard 🚦📊

A real-time Computer Vision application that tracks people, counts entries/exits, generates movement heatmaps, and calculates dwell time, all visualized on a modern web dashboard.

🌟 Features

Real-time People Tracking: Uses YOLOv8 and ByteTrack algorithm to assign unique IDs to individuals across frames.

Line Crossing Counter: Automatically counts how many people cross a specific virtual line in the video feed.

Dynamic Heatmaps: Visualizes high-traffic areas by overlaying a heatmap on the video feed in real-time.

Dwell Time Analytics: Calculates how long individuals stay in the frame before leaving.

Web Dashboard: A responsive Flask-based interface using TailwindCSS and Chart.js to display live stats and graphs.

📂 Project Structure

traffic-monitor/
├── app.py                # Main application logic (Flask + OpenCV + YOLO)
├── requirements.txt      # List of python dependencies
└── templates/
    └── index.html        # Frontend dashboard (HTML/JS/Tailwind)


🚀 Installation & Setup

Follow these steps to run the project locally.

1. Clone the Repository

git clone https://github.com/AayushPurivsKartik/traffic-monitor.git
cd traffic-monitor


2. Set up a Virtual Environment (Recommended)

It's best practice to use a virtual environment to manage dependencies.

Windows:

python -m venv venv
venv\Scripts\activate


Mac/Linux:

python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


Note: On the first run, the application will automatically download the YOLOv8 weights file (yolov8n.pt), which is about 6MB.

🏃‍♂️ Usage

Run the application:

python app.py


Open your web browser and navigate to:

[http://127.0.0.1:5000](http://127.0.0.1:5000)


Allow camera access if prompted.

⚙️ Configuration

To change the input source (e.g., from Webcam to a Video File), open app.py and modify line 12:

# For Webcam
VIDEO_SOURCE = 0

# For Video File
VIDEO_SOURCE = "path/to/your_video.mp4"


You can also adjust the counting line position by changing LINE_POSITION (0.0 to 1.0).

🧠 How It Works

Detection: The YOLOv8 model detects objects classified as "Person" in every frame.

Tracking: The logic persists IDs across frames so the system knows "Person 1" is the same entity in Frame 10 and Frame 50.

Heatmap Logic: A NumPy array accumulates intensity values at the center coordinates of every detected person. This array is normalized and color-mapped to create the visual overlay.

Counting: The system monitors the centroid of tracked persons. If they cross the defined horizontal threshold, the counter increments.

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request