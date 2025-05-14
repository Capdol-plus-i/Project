#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import logging
import numpy as np
import time
import subprocess
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

app = Flask(__name__)

HTML = """
<!DOCTYPE html><html><head><title>Hand Tracking Stream</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{text-align:center;margin:20px;font-family:sans-serif}
img{max-width:640px;width:100%;border:1px solid #ccc}
.status-bar{display:flex;justify-content:center;gap:20px;margin:10px 0;flex-wrap:wrap}
.status-item{padding:5px 10px;border-radius:5px;background:#f0f0f0}
#error{display:none;color:red;padding:20px;border:1px solid #ccc;margin:20px auto;max-width:640px}
</style>
</head><body>
<h1>Hand Tracking Stream</h1>
<div><img src="/video_feed" onerror="this.style.display='none';document.getElementById('error').style.display='block';">
<div id="error">Camera stream unavailable. Check console for details.</div>
</div>
<div class="status-bar">
    <div class="status-item">Status: <span id="status">Connecting...</span></div>
    <div class="status-item">FPS: <span id="fps">0</span></div>
</div>
<div id="camera-info"></div>

<script>
let connectedStatus = false;
let lastFrameTime = Date.now();
let frameCount = 0;
let fps = 0;

// Update FPS calculation
function updateFps() {
    const now = Date.now();
    const elapsed = now - lastFrameTime;
    if (elapsed > 1000) {
        fps = Math.round((frameCount / elapsed) * 1000);
        document.getElementById('fps').textContent = fps;
        frameCount = 0;
        lastFrameTime = now;
    }
}

// Update status every second
setInterval(function() {
    document.getElementById('status').textContent = connectedStatus ? 'Connected' : 'Reconnecting...';
    updateFps();
}, 1000);

// Handle image load events (each new frame)
document.querySelector('img').onload = function() {
    connectedStatus = true;
    frameCount++;
};

// Handle image errors (stream interruptions)
document.querySelector('img').onerror = function() {
    connectedStatus = false;
};

// Get camera info on page load
fetch('/camera_info')
    .then(response => response.json())
    .then(data => {
        const infoDiv = document.getElementById('camera-info');
        infoDiv.innerHTML = '<h3>Camera Information</h3>';
        if (data.cameras.length === 0) {
            infoDiv.innerHTML += '<p>No cameras detected</p>';
        } else {
            const list = document.createElement('ul');
            list.style.listStyle = 'none';
            list.style.padding = '0';
            list.style.maxWidth = '640px';
            list.style.margin = '0 auto';
            data.cameras.forEach(camera => {
                const item = document.createElement('li');
                item.style.padding = '10px';
                item.style.margin = '5px 0';
                item.style.backgroundColor = '#f9f9f9';
                item.style.borderRadius = '5px';
                item.style.textAlign = 'left';
                item.innerHTML = `<strong>${camera.device}</strong>: ${camera.name}`;
                list.appendChild(item);
            });
            infoDiv.appendChild(list);
        }
    })
    .catch(error => {
        console.error('Error fetching camera info:', error);
    });
</script>
</body></html>
"""

def get_camera_info():
    """Get information about available cameras using v4l2-ctl"""
    cameras = []
    try:
        # List video devices
        if os.path.exists('/dev/video0'):  # Check if we're on a Linux system with video devices
            for i in range(10):  # Check first 10 potential cameras
                device_path = f'/dev/video{i}'
                if os.path.exists(device_path):
                    try:
                        # Try to get camera name using v4l2-ctl
                        cmd = ['v4l2-ctl', '--device', device_path, '--info']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                        name = "Unknown camera"
                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                name = line.split(':')[1].strip()
                                break
                        cameras.append({
                            'device': device_path,
                            'id': i,
                            'name': name
                        })
                    except (subprocess.SubprocessError, FileNotFoundError):
                        # Fallback if v4l2-ctl fails or isn't installed
                        cameras.append({
                            'device': device_path,
                            'id': i,
                            'name': "Camera (details unavailable)"
                        })
        else:
            # Non-Linux system or no video devices
            logger.info("No Linux video devices found or not a Linux system")
    except Exception as e:
        logger.error(f"Error getting camera info: {e}")
    
    # If no cameras found through v4l2, try using OpenCV
    if not cameras:
        logger.info("Trying to detect cameras using OpenCV")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    'device': f'Camera {i}',
                    'id': i,
                    'name': "Camera detected by OpenCV"
                })
                cap.release()
    
    return cameras

def create_error_frame(text):
    """Create a simple error frame with text"""
    img = np.zeros((480, 640, 3), np.uint8)
    # Add a more visible background for text
    cv2.rectangle(img, (20, 210), (620, 270), (50, 50, 50), -1)
    # Add text
    cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Add instructions
    cv2.putText(img, "Please check camera connections and permissions", 
                (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img

def create_test_pattern():
    """Create a test pattern image when no camera is available"""
    img = np.zeros((480, 640, 3), np.uint8)
    
    # Add color bars
    colors = [
        (255, 0, 0),    # Blue (BGR)
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 255) # White
    ]
    
    bar_width = img.shape[1] // len(colors)
    for i, color in enumerate(colors):
        x1 = i * bar_width
        x2 = (i + 1) * bar_width
        img[:240, x1:x2] = color
    
    # Add text
    cv2.putText(img, "Camera Not Available", (160, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Using Test Pattern", (180, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add frame counter box
    cv2.rectangle(img, (10, 430), (150, 470), (50, 50, 50), -1)
    
    return img

def gen_frames():
    width, height = 640, 480
    camera_id = 0  # Start with camera 0
    
    # Get list of available cameras
    cameras = get_camera_info()
    if cameras:
        logger.info(f"Found {len(cameras)} camera(s):")
        for camera in cameras:
            logger.info(f"  {camera['device']}: {camera['name']} (ID: {camera['id']})")
    else:
        logger.warning("No cameras detected!")
    
    # Initialize MediaPipe Hands outside the camera loop
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        frame_count = 0
        use_test_pattern = False
        cap = None
        
        while True:  # Endless loop for continuous reconnection attempts
            try:
                # Try to connect to a real camera if we're not using the test pattern
                if not use_test_pattern:
                    # Close previous capture if exists
                    if cap is not None and hasattr(cap, 'release'):
                        cap.release()
                        time.sleep(0.5)  # Wait before reconnecting
                    
                    logger.info(f"Connecting to camera ID: {camera_id}")
                    
                    # Try a different capture API
                    if os.name == 'posix':  # Linux/Mac
                        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                    else:  # Windows or other
                        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                    
                    if not cap.isOpened():
                        # Try without specifying backend
                        cap.release()
                        cap = cv2.VideoCapture(camera_id)
                    
                    if not cap.isOpened():
                        logger.error(f"Failed to open camera ID: {camera_id}")
                        # Try next camera ID or fall back to test pattern
                        camera_id = (camera_id + 1) % max(5, len(cameras) + 1)
                        
                        # If we've tried all cameras, use test pattern
                        if camera_id == 0:
                            logger.warning("Switching to test pattern mode")
                            use_test_pattern = True
                        
                        continue
                    
                    logger.info(f"Connected to camera ID: {camera_id}")
                    
                    # Try setting camera parameters
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Try to configure the camera format (MJPG might be faster)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Reduce buffer size to minimize latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Read a test frame to verify camera is working
                    for _ in range(5):  # Try multiple times
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            break
                        time.sleep(0.1)
                    
                    if not ret or test_frame is None or test_frame.size == 0:
                        logger.error(f"Camera {camera_id} opened but failed initial frame read")
                        # Try next camera
                        camera_id = (camera_id + 1) % max(5, len(cameras) + 1)
                        if camera_id == 0:
                            logger.warning("Switching to test pattern mode")
                            use_test_pattern = True
                        continue
                
                # Main frame processing loop
                while True:
                    frame_count += 1
                    
                    if use_test_pattern:
                        # Create a test pattern image
                        image = create_test_pattern()
                        # Add frame counter
                        cv2.putText(image, f"Frame: {frame_count}", (20, 455), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        time.sleep(0.033)  # ~30 FPS
                    else:
                        # Read from camera
                        success, image = cap.read()
                        
                        if not success or image is None or image.size == 0:
                            logger.warning(f"Failed to read frame from camera {camera_id}")
                            break  # Exit inner loop to try reconnecting
                        
                        # Process image with MediaPipe
                        try:
                            # Convert image to RGB for MediaPipe
                            image.flags.writeable = False
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = hands.process(image_rgb)
                            
                            # Convert back to BGR for OpenCV
                            image.flags.writeable = True
                            
                            # Process hand landmarks if detected
                            if results.multi_hand_landmarks and results.multi_handedness:
                                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                    # Get hand type
                                    hand_type = handedness.classification[0].label
                                    
                                    # Get index finger tip position
                                    index_finger_tip = hand_landmarks.landmark[8]
                                    tip_x = int(index_finger_tip.x * width)
                                    tip_y = int(index_finger_tip.y * height)
                                    
                                    # Set hand color based on type
                                    color = (0, 255, 0) if hand_type == "Left" else (0, 0, 255)
                                    
                                    # Display hand type text
                                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                    wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
                                    
                                    cv2.putText(
                                        image, 
                                        hand_type, 
                                        (wrist_x - 30, wrist_y - 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, 
                                        (255, 255, 255), 
                                        2
                                    )
                                    
                                    # Draw hand landmarks
                                    mp_drawing.draw_landmarks(
                                        image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style()
                                    )
                                    
                                    # Add coordinates text on image
                                    cv2.putText(
                                        image,
                                        f"Finger: x={tip_x}, y={tip_y}", 
                                        (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, 
                                        (255, 255, 255), 
                                        2
                                    )
                            
                            # Add frame counter
                            cv2.putText(
                                image,
                                f"Frame: {frame_count}", 
                                (10, height - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (255, 255, 255), 
                                2
                            )
                            
                            # Flip image for natural view
                            image = cv2.flip(image, 1)
                            
                        except Exception as e:
                            logger.error(f"Error processing frame: {e}")
                            # Just continue without processing if there's an error
                    
                    # Convert frame to JPEG for streaming
                    try:
                        ret, buffer = cv2.imencode('.jpg', image)
                        if not ret:
                            logger.error("Failed to encode frame")
                            continue
                        
                        # Yield the frame
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                               
                    except Exception as e:
                        logger.error(f"Error encoding frame: {e}")
                
                # If we exit the inner loop, camera connection is lost
                if not use_test_pattern:
                    logger.warning(f"Lost connection to camera {camera_id}, reconnecting...")
                    if cap is not None:
                        cap.release()
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Camera error: {e}")
                # Create an error frame
                error_img = create_error_frame(f"Error: {str(e)[:30]}")
                ret, buffer = cv2.imencode('.jpg', error_img)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(2)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_info')
def camera_info():
    from flask import jsonify
    return jsonify({'cameras': get_camera_info()})

if __name__ == '__main__':
    logger.info("Starting Flask server with hand tracking")
    # Use threaded=True for better performance
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)