#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial, signal, sys, csv
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import mediapipe as mp

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = f"lerobot/capdol/models/model_parameters_resnet.npz"
BUTTON_POSITIONS = {0: 110, 1: 1110}
PORT_ACM0, PORT_ACM1 = "/dev/ttyACM0", "/dev/ttyACM1"
BAUDRATE = 1000000
DYNAMIXEL_IDS_ACM0, DYNAMIXEL_IDS_ACM1 = [1, 2, 3, 4], [1]

# CSV Configuration
CSV_FILENAME = "lerobot/capdol/robot_snapshots.csv"
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y", 
               "joint_1", "joint_2", "joint_3", "joint_4"]
CSV_LOCK = threading.Lock()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands
robot, controller = None, None

# Helper functions for Dynamixel SDK
def DXL_LOBYTE(value): return value & 0xFF
def DXL_HIBYTE(value): return (value >> 8) & 0xFF
def DXL_LOWORD(value): return value & 0xFFFF
def DXL_HIWORD(value): return (value >> 16) & 0xFFFF

# CSV Management Functions
def init_csv_file():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        logger.info(f"Created new CSV file: {CSV_FILENAME}")
    else:
        logger.info(f"Using existing CSV file: {CSV_FILENAME}")

def save_snapshot_to_csv(data):
    """Save snapshot data to CSV file"""
    try:
        with CSV_LOCK:
            
            # Prepare row data
            row_data = data
            
            # Write to CSV
            with open(CSV_FILENAME, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            # Count total snapshots
            with open(CSV_FILENAME, 'r') as csvfile:
                total_snapshots = sum(1 for line in csvfile) - 1  # Subtract header
            
            logger.info(f"Snapshot saved to CSV. Total snapshots: {total_snapshots}")
            return True, total_snapshots
            
    except Exception as e:
        logger.error(f"Error saving snapshot to CSV: {e}")
        return False, 0

def get_csv_stats():
    """Get CSV file statistics"""
    try:
        if not os.path.exists(CSV_FILENAME):
            return {"exists": False, "total_snapshots": 0, "file_size": 0}
            
        with open(CSV_FILENAME, 'r') as csvfile:
            total_lines = sum(1 for line in csvfile)
            total_snapshots = max(0, total_lines - 1)  # Subtract header
            
        file_size = os.path.getsize(CSV_FILENAME)
        
        return {
            "exists": True, 
            "total_snapshots": total_snapshots, 
            "file_size": file_size,
            "filename": CSV_FILENAME
        }
    except Exception as e:
        logger.error(f"Error getting CSV stats: {e}")
        return {"exists": False, "total_snapshots": 0, "file_size": 0}

class ManipulatorRobot:
    def __init__(self):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
            
            self.PortHandler, self.PacketHandler = PortHandler, PacketHandler
            self.GroupSyncWrite, self.GroupSyncRead = GroupSyncWrite, GroupSyncRead
            
            # Protocol and address settings
            self.PROTOCOL_VERSION = 2.0
            self.ADDR_TORQUE_ENABLE = 64
            self.ADDR_OPERATING_MODE = 11
            self.ADDR_GOAL_POSITION = 116
            self.ADDR_PRESENT_POSITION = 132
            self.ADDR_POSITION_P_GAIN = 84
            self.TORQUE_ENABLE, self.TORQUE_DISABLE = 1, 0
            
            self.port_handlers = {
                'follower': self.PortHandler(PORT_ACM0),
                'leader': self.PortHandler(PORT_ACM1)
            }
            self.packet_handlers = {
                'follower': self.PacketHandler(self.PROTOCOL_VERSION),
                'leader': self.PacketHandler(self.PROTOCOL_VERSION)
            }
            self.motor_ids = {
                'follower': DYNAMIXEL_IDS_ACM0,
                'leader': DYNAMIXEL_IDS_ACM1
            }
            self.sync_writers, self.sync_readers = {}, {}
            # Track connection status for each arm
            self.connected_arms = set()
            self.is_connected = False
        except ImportError as e:
            logger.error(f"Dynamixel SDK import error: {e}")
            self.connected_arms = set()
            self.is_connected = False

    def connect(self):
        """Modified to allow partial connections - at least one port must connect"""
        if self.is_connected: return True
        
        connected_count = 0
        self.connected_arms.clear()
        
        try:
            for arm_type, port_handler in self.port_handlers.items():
                try:
                    port_name = port_handler.getPortName()
                    
                    # Check if port exists before trying to connect
                    if not os.path.exists(port_name):
                        logger.warning(f"Port {port_name} for {arm_type} does not exist, skipping...")
                        continue
                    
                    if not port_handler.openPort():
                        logger.warning(f"Failed to open port {port_name} for {arm_type}")
                        continue
                        
                    if not port_handler.setBaudRate(BAUDRATE):
                        logger.warning(f"Failed to set baudrate for {arm_type}")
                        port_handler.closePort()
                        continue
                    
                    logger.info(f"Connected '{arm_type}' on {port_name}")
                    self.connected_arms.add(arm_type)
                    connected_count += 1
                    
                    # Initialize sync instances only for connected arms
                    self.sync_writers[arm_type] = self.GroupSyncWrite(
                        port_handler, self.packet_handlers[arm_type], 
                        self.ADDR_GOAL_POSITION, 4)
                    self.sync_readers[arm_type] = self.GroupSyncRead(
                        port_handler, self.packet_handlers[arm_type],
                        self.ADDR_PRESENT_POSITION, 4)
                    
                    # Add parameters for sync read
                    for dxl_id in self.motor_ids[arm_type]:
                        self.sync_readers[arm_type].addParam(dxl_id)
                        
                except Exception as e:
                    logger.warning(f"Error connecting {arm_type}: {e}")
                    continue
            
            # Consider connection successful if at least one arm connects
            if connected_count > 0:
                self.is_connected = True
                logger.info(f"Robot partially connected: {connected_count} arm(s) connected - {list(self.connected_arms)}")
                if 'follower' not in self.connected_arms:
                    logger.warning("Follower arm not connected - gesture control will be limited")
                if 'leader' not in self.connected_arms:
                    logger.warning("Leader arm not connected - some manual controls may not work")
                return True
            else:
                logger.error("No robot arms could be connected")
                self._cleanup_connections()
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._cleanup_connections()
            return False

    def _cleanup_connections(self):
        for arm_type, port_handler in self.port_handlers.items():
            try: 
                port_handler.closePort()
            except: 
                pass
        self.connected_arms.clear()
        self.is_connected = False

    def is_arm_connected(self, arm_type):
        """Check if specific arm is connected"""
        return arm_type in self.connected_arms

    def disable_torque(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot disable torque for {arm_type} - not connected")
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            for dxl_id in self.motor_ids[arm_type]:
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            logger.info(f"Torque disabled for {arm_type}")
            return True
        except Exception as e:
            logger.error(f"Torque disable error: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot setup control for {arm_type} - not connected")
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            
            for dxl_id in self.motor_ids[arm_type]:
                # Disable torque to change operating mode
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
                # Set operating mode (Position Control Mode)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_OPERATING_MODE, 3)
                # Enable torque
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
                # Set Position P Gain
                packet_handler.write2ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_POSITION_P_GAIN, 200)
            logger.info(f"Control setup complete for {arm_type}")
            return True
        except Exception as e:
            logger.error(f"Control setup error: {e}")
            return False

    def move(self, positions, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            return False
        try:
            sync_writer = self.sync_writers[arm_type]
            sync_writer.clearParam()
            
            for i, dxl_id in enumerate(self.motor_ids[arm_type]):
                if i < len(positions):
                    position = int(positions[i])
                    # Clamp position to safe range
                    position = max(0, min(4095, position))
                    param_goal_position = [
                        DXL_LOBYTE(DXL_LOWORD(position)),
                        DXL_HIBYTE(DXL_LOWORD(position)),
                        DXL_LOBYTE(DXL_HIWORD(position)),
                        DXL_HIBYTE(DXL_HIWORD(position))
                    ]
                    sync_writer.addParam(dxl_id, param_goal_position)
            
            result = sync_writer.txPacket()
            return result == 0  # COMM_SUCCESS
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def set_joint_position(self, joint_index, position, arm_type='leader'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot set joint position for {arm_type} - not connected")
            return False
        if joint_index >= len(self.motor_ids[arm_type]): 
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            dxl_id = self.motor_ids[arm_type][joint_index]
            
            # Clamp position to safe range
            position = max(0, min(4095, int(position)))
            
            dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
                port_handler, dxl_id, self.ADDR_GOAL_POSITION, position)
            
            if dxl_comm_result != 0 or dxl_error != 0:
                logger.error(f"Joint position error: comm={dxl_comm_result}, error={dxl_error}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Joint position error: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            return [None] * len(self.motor_ids[arm_type])
        try:
            sync_reader = self.sync_readers[arm_type]
            
            if sync_reader.txRxPacket() != 0:
                return [None] * len(self.motor_ids[arm_type])
            
            positions = []
            for dxl_id in self.motor_ids[arm_type]:
                if sync_reader.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, 4):
                    positions.append(sync_reader.getData(dxl_id, self.ADDR_PRESENT_POSITION, 4))
                else:
                    positions.append(None)
            
            return positions
        except Exception as e:
            logger.error(f"Position read error: {e}")
            return [None] * len(self.motor_ids[arm_type])

    def disconnect(self):
        if not self.is_connected: return
        for arm_type in self.connected_arms.copy():
            try:
                self.disable_torque(arm_type)
                self.port_handlers[arm_type].closePort()
            except: 
                pass
        self.connected_arms.clear()
        self.is_connected = False
        logger.info("Robot disconnected")

class RobotController:
    def __init__(self, robot, model_path=None, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.cams = [None, None]
        self.width, self.height = 640, 480
        self.running = False
        self.control_active = False
        self.data_lock = threading.Lock()
        self.last_frames = [None, None]
        self.last_data = [None] * 8
        self.tip = [(0,0), (0,0)]
        self.hand_detected = [False, False]
        self.z = 10
        self.serial_port = None
        self.last_status_update = 0
        self.status_update_interval = 0.1
        
        # Load model
        self.params = None
        if model_path and os.path.exists(model_path):
            try:
                params = np.load(model_path)
                self.params = {k: params[k] for k in params.files}
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Model load error: {e}")
        else:
            logger.warning(f"Model not found at {model_path}, using dummy predictions")
        
        # Initialize MediaPipe
        self.hands = [mp_hands.Hands(
            model_complexity=1, 
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1, 
            max_num_hands=1,  # Changed from 2 to 1 for better performance
            static_image_mode=False
        ) for _ in range(2)]
        
        # Warm up models
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            try:
                hand.process(dummy)
            except Exception as e:
                logger.warning(f"MediaPipe warmup warning: {e}")

    def start(self, cam_ids=(2, 0), serial_port='/dev/ttyACM1'):
        if self.running: return True
        
        logger.info(f"Starting controller with cameras {cam_ids} and serial {serial_port}")
        
        if not self._open_cams(cam_ids): 
            logger.error("Failed to open cameras")
            return False
        
        # Try to open serial port (non-blocking)
        try:
            if os.path.exists(serial_port):
                self.serial_port = serial.Serial(serial_port, 9600, timeout=1)
                time.sleep(2)
                logger.info(f"Serial port {serial_port} connected")
            else:
                logger.warning(f"Serial port {serial_port} not found, continuing without it")
        except Exception as e:
            logger.warning(f"Serial port error: {e}, continuing without it")
        
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        if self.serial_port:
            threading.Thread(target=self._serial_listener, daemon=True).start()
        
        logger.info("Controller started successfully")
        return True

    def start_control(self):
        if not self.running or self.control_active: return False
        
        # Check if the required arm is connected
        if not self.robot.is_arm_connected(self.arm_type):
            logger.warning(f"Cannot start control - {self.arm_type} arm not connected")
            return False
            
        logger.info("Starting gesture control...")
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            logger.info("Gesture control activated")
        return success

    def stop_control(self):
        if self.control_active:
            logger.info("Stopping gesture control...")
            self.control_active = False
            if self.robot.is_arm_connected(self.arm_type):
                self.robot.disable_torque(self.arm_type)
            logger.info("Gesture control deactivated")
        return True

    def _serial_listener(self):
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    try:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        if line.startswith("CMD:ROBOT:"):
                            mode = int(line.split(":")[-1])
                            position = BUTTON_POSITIONS.get(mode)
                            if position is not None and self.robot.is_arm_connected('leader'):
                                self.robot.set_joint_position(0, position, 'leader')
                                socketio.emit('robot_mode', {'mode': mode, 'position': position})
                                logger.info(f"Serial command: Mode {mode}, Position {position}")
                            elif not self.robot.is_arm_connected('leader'):
                                logger.warning("Serial command received but leader arm not connected")
                    except UnicodeDecodeError as e:
                        logger.warning(f"Serial decode error (ignoring): {e}")
                        # Clear the buffer to avoid repeated errors
                        self.serial_port.reset_input_buffer()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Serial error: {e}")
                time.sleep(1)

    def _open_cams(self, cam_ids):
        """Open cameras with better error handling"""
        for i, cid in enumerate(cam_ids):
            try:
                cap = cv2.VideoCapture(cid)
                if not cap.isOpened():
                    logger.error(f"Camera {i+1} (ID: {cid}) failed to open")
                    # Try different backends
                    for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]:
                        try:
                            cap = cv2.VideoCapture(cid, backend)
                            if cap.isOpened():
                                logger.info(f"Camera {i+1} opened with backend {backend}")
                                break
                        except:
                            continue
                    else:
                        return False
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                
                # Verify camera settings
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"Camera {i+1}: {actual_width}x{actual_height}")
                
                self.cams[i] = cap
            except Exception as e:
                logger.error(f"Camera {i+1} error: {e}")
                return False
        return True

    def _predict(self, x, y, z):
        """Neural network prediction with safety checks"""
        if not self.params:
            # Return dummy positions that gradually move joints
            t = time.time()
            return np.array([
                2048 + int(500 * np.sin(t * 0.5)),
                2048 + int(300 * np.cos(t * 0.3)),
                2048 + int(200 * np.sin(t * 0.7)),
                2048 + int(100 * np.cos(t * 0.9))
            ])
        
        try:
            # Normalize and clamp inputs
            x_norm = max(-1, min(1, x / 650.0))
            y_norm = max(-1, min(1, y / 650.0)) 
            z_norm = max(-1, min(1, z / 650.0))
            
            A = np.array([[x_norm], [y_norm], [z_norm]])
            L = len(self.params) // 2
            
            for l in range(1, L):
                A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
            
            output = ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()
            
            # Safety clamp outputs to reasonable joint limits
            output = np.clip(output, 0, 4095)
            
            return output.astype(int)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(4, dtype=int)

    def _process_frame(self, idx):
        """Process frame with enhanced error handling"""
        if not self.cams[idx]:
            return self._create_dummy_frame(f"Camera {idx+1} not ready")
        
        # Try to read frame with retries
        frame = None
        for attempt in range(3):
            try:
                ret, f = self.cams[idx].read()
                if ret and f is not None:
                    frame = f
                    break
            except Exception as e:
                logger.warning(f"Camera {idx+1} read attempt {attempt+1} failed: {e}")
                time.sleep(0.01)
        
        if frame is None:
            return self._create_dummy_frame(f"Camera {idx+1}: No frame")

        try:
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return self._create_dummy_frame(f"Camera {idx+1}: Invalid frame")
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands[idx].process(rgb)
            
            self.hand_detected[idx] = False
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark = hand_landmarks.landmark[8]  # Index finger tip
                
                # Validate landmark coordinates
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    self.tip[idx] = (x, y)
                    self.hand_detected[idx] = True
                    
                    if idx == 1:  # Z coordinate from second camera
                        self.z = max(10, min(y, self.height - 10))  # Clamp Z value
                    
                    # Draw detection
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"Hand {idx+1}", (x-30, y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                               
            # Add status overlay
            status_color = (0, 255, 0) if self.hand_detected[idx] else (0, 0, 255)
            status_text = f"Cam{idx+1}: {'HAND' if self.hand_detected[idx] else 'NO HAND'}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                       
        except Exception as e:
            logger.error(f"Frame processing error camera {idx+1}: {e}")
            if frame is not None:
                cv2.putText(frame, f"Processing Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def _create_dummy_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, message, (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def _process_loop(self):
        while self.running:
            try:
                frames = [self._process_frame(i) for i in range(2)]
                positions = self.robot.get_positions(self.arm_type)
                
                if self.control_active and self.hand_detected[0] and self.robot.is_arm_connected(self.arm_type):
                    joints = self._predict(*self.tip[0], self.z)
                    if np.sum(np.abs(joints)) > 0:
                        self.robot.move(joints, self.arm_type)
                
                with self.data_lock:
                    self.last_frames = frames
                    self.last_data = [
                        self.tip[0][0] if self.hand_detected[0] else None, 
                        self.tip[0][1] if self.hand_detected[0] else None,
                        self.tip[1][0] if self.hand_detected[1] else None, 
                        self.tip[1][1] if self.hand_detected[1] else None
                    ] + positions
                
                # Emit status updates at controlled intervals
                current_time = time.time()
                if current_time - self.last_status_update >= self.status_update_interval:
                    self._emit_status_update()
                    self.last_status_update = current_time
                    
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        self._cleanup()

    def _emit_status_update(self):
        data = self.get_last_data()
        safe_data = []
        for v in data:
            if v is None:
                safe_data.append(None)
            elif isinstance(v, np.generic):
                safe_data.append(v.item())
            else:
                safe_data.append(int(v) if isinstance(v, (int, float)) else v)
                
        headers = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
                 "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]
                 
        socketio.emit('status_update', dict(zip(headers, safe_data)))

    def _cleanup(self):
        self._cleanup_cameras()
        if self.serial_port and self.serial_port.is_open:
            try: 
                self.serial_port.close()
            except: 
                pass
            self.serial_port = None

    def _cleanup_cameras(self):
        for hand in self.hands:
            try: 
                hand.close()
            except: 
                pass
        for i, cam in enumerate(self.cams):
            if cam:
                try: 
                    cam.release()
                except: 
                    pass
        self.cams = [None, None]

    def get_last_frame(self, idx):
        with self.data_lock:
            frame = self.last_frames[idx]
            return frame.copy() if frame is not None else self._create_dummy_frame("No frame")

    def get_last_data(self):
        with self.data_lock:
            return self.last_data.copy()

    def stop(self):
        if not self.running: return
        logger.info("Stopping controller...")
        self.running = False
        self.control_active = False

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """REST API endpoint for system status"""
    if system_initialized and controller and controller.running:
        return jsonify({
            'system_ready': True,
            'control_active': controller.control_active,
            'robot_connected': robot.is_connected,
            'connected_arms': list(robot.connected_arms),
            'hand_detected': controller.hand_detected,
            'data': controller.get_last_data()
        })
    else:
        return jsonify({
            'system_ready': False,
            'control_active': False,
            'robot_connected': robot.is_connected if robot else False,
            'connected_arms': list(robot.connected_arms) if robot else [],
            'hand_detected': [False, False],
            'data': [None] * 8
        })

@app.route('/api/csv/download')
def api_download_csv():
    """REST API endpoint to download CSV file"""
    try:
        if os.path.exists(CSV_FILENAME):
            from flask import send_file
            return send_file(CSV_FILENAME, 
                           as_attachment=True, 
                           download_name=f"robot_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mimetype='text/csv')
        else:
            return jsonify({'error': 'CSV file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv/stats')
def api_csv_stats():
    """REST API endpoint for CSV statistics"""
    try:
        stats = get_csv_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshot', methods=['POST'])
def api_take_snapshot():
    """REST API endpoint to take snapshot"""
    try:
        if not controller or not controller.running:
            return jsonify({'success': False, 'error': 'System not ready'}), 400
        
        current_data = controller.get_last_data()
        if len(current_data) != 8:
            return jsonify({'success': False, 'error': 'Invalid data length'}), 400
        
        success, total_snapshots = save_snapshot_to_csv(current_data)
        
        if success:
            return jsonify({
                'success': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data': current_data,
                'total_snapshots': total_snapshots,
                'connected_arms': list(robot.connected_arms) if robot else []
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save to CSV'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_frames(cam_idx):
    while True:
        try:
            if controller and controller.running:
                frame = controller.get_last_frame(cam_idx)
            else:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "System starting...", (70, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            if not controller or not controller.running:
                time.sleep(0.5)
            else:
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            time.sleep(0.5)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    logger.info(f"Current system state - initialized: {system_initialized}, status: {system_status_data}")
    
    # Send current system status
    socketio.emit('system_status', system_status_data, to=request.sid)
    
    # Send robot connection status
    if robot and robot.is_connected:
        socketio.emit('robot_status', robot_status_data, to=request.sid)
    else:
        socketio.emit('robot_status', {'connected_arms': []}, to=request.sid)
    
    # Send control status
    if controller and controller.running:
        socketio.emit('control_status', {'active': controller.control_active}, to=request.sid)
    else:
        socketio.emit('control_status', {'active': False}, to=request.sid)
    
    logger.info(f"Sent initial status to client {request.sid}: {system_status_data}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_control')
def handle_start_control(data=None):
    logger.info("Received start_control command")
    if not system_initialized or not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    if not robot.is_arm_connected('follower'):
        return {'success': False, 'error': 'Follower arm not connected'}
    success = controller.start_control()
    if success:
        socketio.emit('control_status', {'active': True})
    return {'success': success}

@socketio.on('stop_control')
def handle_stop_control(data=None):
    logger.info("Received stop_control command")
    if not system_initialized or not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    success = controller.stop_control()
    if success:
        socketio.emit('control_status', {'active': False})
    return {'success': success}

@socketio.on('set_robot_mode')
def handle_set_robot_mode(data):
    logger.info(f"Received set_robot_mode command: {data}")
    if not system_initialized or not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    if not robot.is_arm_connected('leader'):
        return {'success': False, 'error': 'Leader arm not connected'}
    
    mode = data.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        return {'success': False, 'error': f'Invalid mode: {mode}'}
    
    success = robot.set_joint_position(0, position, 'leader')
    if success:
        socketio.emit('robot_mode', {'mode': mode, 'position': position})
    return {'success': success, 'mode': mode, 'position': position}

@socketio.on('get_status')
def handle_get_status(data=None):
    """Get current system status via WebSocket"""
    if system_initialized and controller and controller.running:
        return {
            'system_ready': True,
            'control_active': controller.control_active,
            'robot_connected': robot.is_connected,
            'connected_arms': list(robot.connected_arms),
            'hand_detected': controller.hand_detected,
            'last_data': controller.get_last_data()
        }
    else:
        return {
            'system_ready': False,
            'control_active': False,
            'robot_connected': robot.is_connected if robot else False,
            'connected_arms': list(robot.connected_arms) if robot else [],
            'hand_detected': [False, False],
            'last_data': [None] * 8
        }

@socketio.on('take_snapshot')
def handle_take_snapshot(data=None):
    """Take snapshot of current system state and save to CSV"""
    logger.info("Received take_snapshot command")
    
    if not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    
    try:
        # Get current data
        current_data = controller.get_last_data()
        
        # Validate data (ensure we have all 8 values)
        if len(current_data) != 8:
            return {'success': False, 'error': 'Invalid data length'}
        
        # Save to CSV
        success, total_snapshots = save_snapshot_to_csv(current_data)
        
        if success:
            # Emit success event to all clients
            socketio.emit('snapshot_saved', {
                'success': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data': current_data,
                'filename': CSV_FILENAME,
                'total_snapshots': total_snapshots,
                'connected_arms': list(robot.connected_arms) if robot else []
            })
            return {'success': True, 'total_snapshots': total_snapshots}
        else:
            return {'success': False, 'error': 'Failed to save to CSV'}
            
    except Exception as e:
        error_msg = f"Snapshot error: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}

@socketio.on('get_csv_stats')
def handle_get_csv_stats(data=None):
    """Get CSV file statistics"""
    try:
        stats = get_csv_stats()
        return {'success': True, 'stats': stats}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Global system state tracking
system_initialized = False
system_status_data = {'status': 'initializing', 'message': 'System starting...'}
robot_status_data = {'connected_arms': []}

# System initialization and cleanup
def init_system(model_path=MODEL_PATH, cam_ids=(0, 2), serial_port='/dev/ttyACM3'):
    """Enhanced system initialization with better error reporting"""
    global robot, controller, system_initialized, system_status_data, robot_status_data
    try:
        logger.info("Initializing robot system...")
        system_status_data = {'status': 'initializing', 'message': 'Connecting to robot...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize robot
        robot = ManipulatorRobot()
        if not robot.connect():
            error_msg = 'Robot connection failed - no robot arms could be connected'
            logger.error(error_msg)
            system_status_data = {'status': 'error', 'message': error_msg}
            socketio.emit('system_status', system_status_data)
            return False
        
        # Emit which arms are connected
        connected_arms = list(robot.connected_arms)
        robot_status_data = {'connected_arms': connected_arms}
        socketio.emit('robot_status', robot_status_data)
        logger.info(f"Connected robot arms: {connected_arms}")
        
        system_status_data = {'status': 'initializing', 'message': 'Setting up robot arms...'}
        socketio.emit('system_status', system_status_data)
        
        # Setup robot arms (only for connected arms)
        if robot.is_arm_connected('follower'):
            robot.disable_torque('follower')
        
        if robot.is_arm_connected('leader'):
            if not robot.setup_control('leader'):
                logger.warning('Leader arm setup failed, but continuing...')
        
        system_status_data = {'status': 'initializing', 'message': 'Starting cameras and controller...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize controller
        controller = RobotController(robot, model_path, 'follower')
        if not controller.start(cam_ids, serial_port):
            robot.disconnect()
            error_msg = 'Controller start failed - check cameras and serial port'
            logger.error(error_msg)
            system_status_data = {'status': 'error', 'message': error_msg}
            socketio.emit('system_status', system_status_data)
            return False
        
        logger.info("System initialization complete!")
        status_message = f'System ready - Connected arms: {connected_arms}'
        system_status_data = {'status': 'ready', 'message': status_message}
        logger.info(f"Setting system_initialized = True, status: {system_status_data}")
        system_initialized = True
        
        # Broadcast updated status to ALL connected clients
        socketio.emit('system_status', system_status_data)
        socketio.emit('robot_status', robot_status_data)
        logger.info("Broadcasted system ready status to all connected clients")
        return True
        
    except Exception as e:
        error_msg = f'System initialization error: {str(e)}'
        logger.error(error_msg)
        system_status_data = {'status': 'error', 'message': error_msg}
        socketio.emit('system_status', system_status_data)
        return False

def cleanup_system():
    """Graceful system cleanup"""
    global system_initialized, system_status_data, robot_status_data
    logger.info("Cleaning up system...")
    system_initialized = False
    system_status_data = {'status': 'error', 'message': 'System shutting down'}
    robot_status_data = {'connected_arms': []}
    
    if controller:
        try: 
            controller.stop()
        except Exception as e:
            logger.error(f"Controller cleanup error: {e}")
    if robot:
        try: 
            robot.disconnect()
        except Exception as e:
            logger.error(f"Robot cleanup error: {e}")
    logger.info("System cleanup complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info('Received shutdown signal, cleaning up...')
    cleanup_system()
    sys.exit(0)

def main():
    """Main application entry point"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize CSV file
    init_csv_file()
    
    # Start system initialization in background
    threading.Thread(target=init_system, daemon=True).start()
    
    try:
        logger.info("Starting Flask-SocketIO server on http://0.0.0.0:5000")
        logger.info("Press Ctrl+C to shutdown")
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()