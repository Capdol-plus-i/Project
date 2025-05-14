#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

HTML = """
<!DOCTYPE html><html><head><title>USB Camera Test</title>
<style>body{text-align:center;margin:20px;font-family:sans-serif}
img{max-width:640px;width:100%;border:1px solid #ccc}</style>
</head><body>
<h1>USB Camera Test</h1>
<div><img src="/video_feed" onerror="this.style.display='none';document.getElementById('error').style.display='block';">
<div id="error" style="display:none;color:red;padding:20px;border:1px solid #ccc;margin-top:20px;">
    Camera stream unavailable. Check console for details.</div>
</div>
<p>Status: <span id="status">Connecting...</span></p>
<script>
setTimeout(function() {
    var img = document.querySelector('img');
    if (img.naturalWidth == 0) {
        document.getElementById('status').textContent = 'Failed to connect';
        document.getElementById('error').style.display = 'block';
        img.style.display = 'none';
    } else {
        document.getElementById('status').textContent = 'Connected';
    }
}, 5000);
</script>
</body></html>
"""

def gen_frames():
    # Try camera IDs 0-9
    for camera_id in range(10):
        logger.info(f"Trying camera ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.info(f"Camera ID {camera_id} not available")
            continue
            
        logger.info(f"Successfully opened camera ID: {camera_id}")
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame")
                    break
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("Failed to encode frame")
                    break
                    
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Error streaming from camera {camera_id}: {e}")
        finally:
            cap.release()
            
    # If we get here, all cameras failed
    logger.error("No working cameras found")

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=8000, debug=True)