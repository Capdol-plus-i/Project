#!/usr/bin/env python3
import sys, termios, tty, select, logging, threading, time, os, subprocess
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Robot Control Classes
class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            return
            
        # Connect arms
        for name in self.follower_arms:
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            self.leader_arms[name].connect()

        # Import dynamically to avoid dependency issues
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        
        # First disable torque for configuration
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        
        # Configure operating mode properly for each motor type
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                # Use integer 4 (Extended Position Control Mode)
                arms[name].write("Operating_Mode", 4)
                # Re-enable torque
                arms[name].write("Torque_Enable", 1)

        self.is_connected = True
        logger.info("Robot connected successfully")

    def send_action(self, action, arm_type='follower'):
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        
        from_idx, to_idx = 0, 0
        for name in arms:
            motor_count = len(arms[name].motor_names)
            to_idx += motor_count
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            
            # Read current position and ensure numpy array
            present_pos = np.array(arms[name].read("Present_Position"), dtype=np.float32)
            
            # Convert goal position to numpy array
            if hasattr(goal_pos, 'numpy'):  # For torch tensors
                goal_pos = goal_pos.numpy()
            goal_pos = np.array(goal_pos, dtype=np.float32)
            
            # Apply safety limits
            max_delta = 150.0
            diff = goal_pos - present_pos
            diff = np.clip(diff, -max_delta, max_delta)
            safe_goal_pos = present_pos + diff
            
            # IMPORTANT: Convert to integers before sending to dynamixel
            safe_goal_pos = np.round(safe_goal_pos).astype(np.int32)
            
            # Send command to motors
            arms[name].write("Goal_Position", safe_goal_pos)

    def disconnect(self):
        if not self.is_connected:
            return
            
        logger.info("Disconnecting robot")
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class JointController:
    def __init__(self, robot, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.arms = robot.follower_arms if arm_type == 'follower' else robot.leader_arms

    def move_joint(self, joint_index, delta_value):
        try:
            # Get current position as numpy array
            current_pos = np.array(self.arms['main'].read('Present_Position'), dtype=np.float32)
            
            # Create a new array for the goal position
            goal_pos = current_pos.copy()
            goal_pos[joint_index] += delta_value
            
            # Send the goal position
            self.robot.send_action(goal_pos, arm_type=self.arm_type)
            logger.debug(f"Moved {self.arm_type} joint {joint_index} by {delta_value}")
        except Exception as e:
            logger.error(f"Error moving joint: {e}")


class KeyboardController:
    def __init__(self, robot):
        self.robot = robot
        self.old_settings = None
        self.follower_ctrl = JointController(robot, 'follower')
        self.leader_ctrl = JointController(robot, 'leader')
        self.running = False
        
        # Key mapping: key -> (controller, joint_index, delta)
        self.key_mapping = {
            # Follower arm
            'w': (self.follower_ctrl, 0, 100.0),   # Shoulder pan up
            's': (self.follower_ctrl, 0, -100.0),  # Shoulder pan down
            'e': (self.follower_ctrl, 1, 100.0),   # Shoulder tilt up
            'd': (self.follower_ctrl, 1, -100.0),  # Shoulder tilt down
            'r': (self.follower_ctrl, 2, 100.0),   # Elbow up
            'f': (self.follower_ctrl, 2, -100.0),  # Elbow down
            't': (self.follower_ctrl, 3, 100.0),   # Wrist up
            'g': (self.follower_ctrl, 3, -100.0),  # Wrist down
            
            # Leader arm
            'y': (self.leader_ctrl, 0, 100.0),     # Shoulder pan up
            'h': (self.leader_ctrl, 0, -100.0),    # Shoulder pan down
            'u': (self.leader_ctrl, 1, 100.0),     # Shoulder tilt up
            'j': (self.leader_ctrl, 1, -100.0),    # Shoulder tilt down
            'i': (self.leader_ctrl, 2, 100.0),     # Elbow up
            'k': (self.leader_ctrl, 2, -100.0),    # Elbow down
            'o': (self.leader_ctrl, 3, 100.0),     # Wrist up
            'l': (self.leader_ctrl, 3, -100.0),    # Wrist down
        }

    def start(self):
        """Start keyboard control in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._keyboard_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Keyboard control started")

    def stop(self):
        """Stop keyboard control thread"""
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        logger.info("Keyboard control stopped")

    def _keyboard_loop(self):
        """Main keyboard control loop running in a thread"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    c = sys.stdin.read(1)
                    if c == 'q':
                        logger.info("Quit command received")
                        self.running = False
                        break
                    if c in self.key_mapping:
                        controller, joint_index, delta = self.key_mapping[c]
                        controller.move_joint(joint_index, delta)
                time.sleep(0.05)  # Prevent CPU overuse
        except Exception as e:
            logger.error(f"Keyboard control error: {e}")
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


# Hand gesture control
class HandController:
    def __init__(self, robot):
        self.robot = robot
        self.follower_ctrl = JointController(robot, 'follower')
        self.leader_ctrl = JointController(robot, 'leader')
        self.is_active = False
        self.prev_landmarks = None
        self.smoothing_factor = 0.5  # For smoothing movements
        self.wrist_z_history = []    # 손목 z 위치 히스토리 저장 (관절 3 제어용)
        self.wrist_z_threshold = 0.1 # 관절 3 제어를 위한 임계값
        
    def set_active(self, active):
        """Enable or disable hand control"""
        self.is_active = active
        logger.info(f"Hand control {'activated' if active else 'deactivated'}")
        
    def process_hand_landmarks(self, hand_landmarks, width, height, handedness):
        """Process hand landmarks and control robot if active"""
        if not self.is_active:
            return
            
        try:
            # Determine which hand and which controller to use
            hand_type = handedness.classification[0].label
            controller = self.follower_ctrl if hand_type == "Left" else self.leader_ctrl
            
            # 손목 위치만 사용하여 로봇 제어
            wrist = hand_landmarks.landmark[0]  # 손목 = landmark 0
            
            # 정규화된 위치 계산 (0-1 범위)
            x_norm = wrist.x
            y_norm = wrist.y
            z_norm = wrist.z*10000000000  # z 위치를 mm로 변환
            
            # 관절 움직임으로 변환
            # x 위치(좌우) → 관절 0 (어깨 패닝)
            joint0_delta = (x_norm - 0.5) * 200
            
            # y 위치(상하) → 관절 1 (어깨 틸트)
            joint1_delta = (y_norm - 0.5) * 400
            
            # z 위치(깊이) → 관절 2 (팔꿈치)
            joint2_delta = (z_norm - 0.5) * 100
            
            # 손목의 z 위치 변화에 기반한 관절 3 (손목/그리퍼) 제어
            self.wrist_z_history.append(z_norm)
            if len(self.wrist_z_history) > 10:
                self.wrist_z_history.pop(0)
            
            # z 변화량이 임계값을 초과하면 관절 3 움직임
            if len(self.wrist_z_history) >= 2:
                z_change = self.wrist_z_history[-1] - self.wrist_z_history[0]
                joint3_delta = z_change * 500 if abs(z_change) > self.wrist_z_threshold else 0
            else:
                joint3_delta = 0
            
            # 이전 값이 있으면 움직임 스무딩 적용
            if self.prev_landmarks is not None:
                joint0_delta = self.smoothing_factor * joint0_delta + (1 - self.smoothing_factor) * self.prev_landmarks[0]
                joint1_delta = self.smoothing_factor * joint1_delta + (1 - self.smoothing_factor) * self.prev_landmarks[1]
                joint2_delta = self.smoothing_factor * joint2_delta + (1 - self.smoothing_factor) * self.prev_landmarks[2]
                joint3_delta = self.smoothing_factor * joint3_delta + (1 - self.smoothing_factor) * self.prev_landmarks[3]
            
            # 다음 스무딩을 위해 현재 값 저장
            self.prev_landmarks = [joint0_delta, joint1_delta, joint2_delta, joint3_delta]
            
            # 각 관절 움직이기
            controller.move_joint(0, joint0_delta)
            controller.move_joint(2, joint1_delta)
            controller.move_joint(1, joint2_delta)
            #controller.move_joint(3, joint3_delta)
            
            # 화면에 손목 위치 표시
            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)
            wrist_z = int(wrist.z * 10000000000)  # z 위치를 mm로 변환
            
            # 디버그 로깅
            logger.debug(f"Wrist control: {hand_type}, pos: ({wrist_x}, {wrist_y}), deltas: {joint0_delta:.2f}, {joint1_delta:.2f}, {joint2_delta:.2f}, {joint3_delta:.2f}")
            
        except Exception as e:
            logger.error(f"Hand control error: {e}")


# Flask web application
app = Flask(__name__)

# Global variables
robot = None
keyboard_controller = None
hand_controller = None
control_mode = "none"  # "keyboard", "hand", or "none"

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
    """Generate video frames with hand tracking and robot control"""
    global hand_controller
    
    width, height = 640, 480
    camera_id = 8  # Start with camera 0
    
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

                            # # Flip image for natural view 수정해봄
                            image = cv2.flip(image, 0)
                            
                            # Process hand landmarks if detected
                            if results.multi_hand_landmarks and results.multi_handedness:
                                image = cv2.flip(image, 0)
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
                                    wrist_z = int(wrist.z * 10000000000)  # Convert to mm

                                    # Draw hand landmarks
                                    mp_drawing.draw_landmarks(
                                        image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style()
                                    )

                                    # 111
                                    image = cv2.flip(image, 0)
                                    
                                    cv2.putText(
                                        image, 
                                        hand_type, 
                                        (wrist_x - 30, wrist_y - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, 
                                        (255, 255, 255), 
                                        2
                                    )

                                    # 222
                                    #image = cv2.flip(image, 0)
                                    
                                    #Add coordinates text on image
                                    cv2.putText(
                                        image,
                                        f"Finger: x={wrist_x}, y={wrist_y}, z={wrist_z}", 
                                        (100, 300), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, 
                                        (255, 255, 255), 
                                        2
                                    )

                                    # Process hand landmarks for robot control
                                    if hand_controller and control_mode == "hand":
                                        hand_controller.process_hand_landmarks(
                                            hand_landmarks, width, height, handedness)
                                    
                                    #image = cv2.flip(image, 0)
                                            
                            # Flip image for natural view 수정해봄
                            #image = cv2.flip(image, 0)

                            # Show current control mode
                            cv2.putText(
                                image,
                                f"Control: {control_mode}", 
                                (width - 200, height - 50), 
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
                            
                            # # Flip image for natural view
                            #image = cv2.flip(image, 0)
                            
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

# API Endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_info')
def camera_info():
    return jsonify({'cameras': get_camera_info()})

@app.route('/robot_status')
def robot_status():
    global robot, control_mode
    return jsonify({
        'connected': robot.is_connected if robot else False,
        'control_mode': control_mode
    })

@app.route('/robot_connect', methods=['POST'])
def robot_connect():
    global robot, keyboard_controller, hand_controller
    
    if not robot:
        try:
            # Initialize robot
            robot = ManipulatorRobot(KochRobotConfig())
            robot.connect()
            
            # Create controllers
            keyboard_controller = KeyboardController(robot)
            hand_controller = HandController(robot)
            
            return jsonify({'status': 'success', 'message': 'Robot connected successfully'})
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to connect robot: {str(e)}'})
    else:
        if not robot.is_connected:
            try:
                robot.connect()
                return jsonify({'status': 'success', 'message': 'Robot reconnected successfully'})
            except Exception as e:
                logger.error(f"Failed to reconnect robot: {e}")
                return jsonify({'status': 'error', 'message': f'Failed to reconnect robot: {str(e)}'})
        else:
            return jsonify({'status': 'info', 'message': 'Robot already connected'})

@app.route('/robot_disconnect', methods=['POST'])
def robot_disconnect():
    global robot, control_mode
    
    # First stop any control
    if control_mode != "none":
        set_control_mode_internal("none")
    
    if robot and robot.is_connected:
        try:
            robot.disconnect()
            return jsonify({'status': 'success', 'message': 'Robot disconnected successfully'})
        except Exception as e:
            logger.error(f"Failed to disconnect robot: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to disconnect robot: {str(e)}'})
    else:
        return jsonify({'status': 'info', 'message': 'Robot already disconnected'})

def set_control_mode_internal(mode):
    """Internal function to set control mode"""
    global control_mode, keyboard_controller, hand_controller
    
    # First stop current control
    if control_mode == "keyboard" and keyboard_controller:
        keyboard_controller.stop()
    elif control_mode == "hand" and hand_controller:
        hand_controller.set_active(False)
    
    # Set new control mode
    control_mode = mode
    
    # Start new control
    if mode == "keyboard" and keyboard_controller:
        keyboard_controller.start()
    elif mode == "hand" and hand_controller:
        hand_controller.set_active(True)
    
    logger.info(f"Control mode changed to: {mode}")

@app.route('/set_control_mode', methods=['POST'])
def set_control_mode():
    from flask import request
    
    global robot
    
    if not robot or not robot.is_connected:
        return jsonify({
            'status': 'error', 
            'message': 'Robot is not connected. Please connect the robot first.'
        })
    
    try:
        data = request.json
        mode = data.get('mode', 'none')
        
        if mode not in ["keyboard", "hand", "none"]:
            return jsonify({
                'status': 'error', 
                'message': f'Invalid control mode: {mode}'
            })
        
        set_control_mode_internal(mode)
        
        return jsonify({
            'status': 'success', 
            'message': f'Control mode set to {mode}'
        })
    except Exception as e:
        logger.error(f"Failed to set control mode: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Failed to set control mode: {str(e)}'
        })

# Main function
def main():
    try:
        logger.info("Starting Robot Hand Control server")
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        global robot, keyboard_controller, control_mode
        
        if control_mode == "keyboard" and keyboard_controller:
            keyboard_controller.stop()
        
        if robot and robot.is_connected:
            robot.disconnect()
        
        logger.info("Server shutdown complete")

if __name__ == '__main__':
    main()