#!/usr/bin/env python3
import os
import json
import time
import threading
import logging
import numpy as np
import cv2
import csv
import serial
from contextlib import contextmanager
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, 
    GroupSyncRead, COMM_SUCCESS
)

# Suppress MediaPipe warnings and setup logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths and constants
BASE_DIR = "lerobot/capdol"
CALIBRATION_DIR = f"{BASE_DIR}/calibration_data"
DATA_DIR = f"{BASE_DIR}/collected_data"
MODEL_PATH = f"{BASE_DIR}/models/model_parameters_resnet.npz"
CSV_PATH = os.path.join(DATA_DIR, "robot_hand_data.csv")
CSV_HEADERS = [
    "camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
    "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"
]
BUTTON_POSITIONS = {0: 110, 1: 1110}  # Mode positions
os.makedirs(DATA_DIR, exist_ok=True)

# Device configurations
PORT_ACM0 = "/dev/ttyACM0"
PORT_ACM1 = "/dev/ttyACM1"
BAUDRATE = 1000000
DYNAMIXEL_IDS_ACM0 = [1, 2, 3, 4]  # IDs on first port
DYNAMIXEL_IDS_ACM1 = [1]           # ID on second port

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands

# Helper functions for byte manipulation
def dxl_lobyte(value):
    return value & 0xFF

def dxl_hibyte(value):
    return (value >> 8) & 0xFF

def dxl_loword(value):
    return value & 0xFFFF

def dxl_hiword(value):
    return (value >> 16) & 0xFFFF

class DynamixelRobot:
    """Simplified Dynamixel robot control class"""
    
    # Control table addresses (for Protocol 2.0)
    ADDR_TORQUE_ENABLE = 64
    ADDR_OPERATING_MODE = 11
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_POSITION_P_GAIN = 84
    
    # Values
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    PROTOCOL_VERSION = 2.0
    
    def __init__(self):
        self.arms = {
            'follower': {'port': PORT_ACM0, 'ids': DYNAMIXEL_IDS_ACM0},
            'leader': {'port': PORT_ACM1, 'ids': DYNAMIXEL_IDS_ACM1}
        }
        self.port_handlers = {}
        self.packet_handlers = {}
        self.sync_writers = {}
        self.sync_readers = {}
        self.is_connected = False
        self.calibration = {}
    
    def connect(self):
        """Connect to all robot arms and setup communication"""
        if self.is_connected:
            return True
            
        try:
            for arm_type, config in self.arms.items():
                # Initialize port handler
                port_handler = PortHandler(config['port'])
                if not port_handler.openPort():
                    logger.error(f"Failed to open port for {arm_type}")
                    self._cleanup_connections()
                    return False
                
                if not port_handler.setBaudRate(BAUDRATE):
                    logger.error(f"Failed to set baudrate for {arm_type}")
                    port_handler.closePort()
                    self._cleanup_connections()
                    return False
                
                # Initialize packet handler
                packet_handler = PacketHandler(self.PROTOCOL_VERSION)
                
                # Store handlers
                self.port_handlers[arm_type] = port_handler
                self.packet_handlers[arm_type] = packet_handler
                
                # Initialize sync writers and readers
                self.sync_writers[arm_type] = GroupSyncWrite(
                    port_handler, packet_handler, self.ADDR_GOAL_POSITION, 4
                )
                
                self.sync_readers[arm_type] = GroupSyncRead(
                    port_handler, packet_handler, self.ADDR_PRESENT_POSITION, 4
                )
                
                # Add parameters for sync read
                for dxl_id in config['ids']:
                    if not self.sync_readers[arm_type].addParam(dxl_id):
                        logger.error(f"Failed to add parameter for Dynamixel ID {dxl_id}")
                
                logger.info(f"Connected '{arm_type}' on {port_handler.getPortName()}")
            
            self.is_connected = True
            self._load_calibration()
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._cleanup_connections()
            return False
    
    def _load_calibration(self):
        """Load calibration data from file"""
        path = os.path.join(CALIBRATION_DIR, 'follower_arm.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.calibration = json.load(f)
                logger.info("Calibration loaded")
            except Exception as e:
                logger.error(f"Calibration error: {e}")
    
    def _cleanup_connections(self):
        """Close all port connections"""
        for arm_type, port_handler in self.port_handlers.items():
            try:
                port_handler.closePort()
            except:
                pass
        
        self.port_handlers = {}
        self.packet_handlers = {}
        self.sync_writers = {}
        self.sync_readers = {}
        self.is_connected = False
    
    @contextmanager
    def arm_context(self, arm_type):
        """Context manager for accessing arm components"""
        if not self.is_connected or arm_type not in self.arms:
            raise ValueError(f"Arm {arm_type} not connected")
            
        yield (
            self.port_handlers[arm_type], 
            self.packet_handlers[arm_type],
            self.sync_writers[arm_type],
            self.sync_readers[arm_type],
            self.arms[arm_type]['ids']
        )
    
    def disable_torque(self, arm_type):
        """Disable torque for all motors in an arm"""
        if not self.is_connected:
            return False
            
        try:
            with self.arm_context(arm_type) as (port, packet, _, _, ids):
                for dxl_id in ids:
                    packet.write1ByteTxRx(
                        port, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                    )
            return True
        except Exception as e:
            logger.error(f"Torque disable error: {e}")
            return False
    
    def setup_control(self, arm_type):
        """Setup control mode for all motors in an arm"""
        if not self.is_connected:
            return False
            
        try:
            with self.arm_context(arm_type) as (port, packet, _, _, ids):
                for dxl_id in ids:
                    # Disable torque to change operating mode
                    packet.write1ByteTxRx(
                        port, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                    )
                    
                    # Set position control mode (3)
                    packet.write1ByteTxRx(
                        port, dxl_id, self.ADDR_OPERATING_MODE, 3
                    )
                    
                    # Enable torque
                    packet.write1ByteTxRx(
                        port, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
                    )
                    
                    # Set Position P Gain
                    packet.write2ByteTxRx(
                        port, dxl_id, self.ADDR_POSITION_P_GAIN, 200
                    )
            return True
        except Exception as e:
            logger.error(f"Control setup error: {e}")
            return False
    
    def move(self, positions, arm_type):
        """Move all motors in an arm to target positions"""
        if not self.is_connected:
            return False
            
        try:
            with self.arm_context(arm_type) as (port, packet, sync_writer, _, ids):
                # Clear sync write parameter storage
                sync_writer.clearParam()
                
                # Add parameters for all motors
                for i, dxl_id in enumerate(ids):
                    if i < len(positions):
                        position = int(positions[i])
                        # Allocate goal position value into byte array
                        param_goal_position = [
                            dxl_lobyte(dxl_loword(position)),
                            dxl_hibyte(dxl_loword(position)),
                            dxl_lobyte(dxl_hiword(position)),
                            dxl_hibyte(dxl_hiword(position))
                        ]
                        # Add parameter for sync write
                        sync_writer.addParam(dxl_id, param_goal_position)
                
                # Sync write goal position
                sync_writer.txPacket()
            return True
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False
    
    def set_joint_position(self, joint_index, position, arm_type):
        """Set the position of a specific joint"""
        if not self.is_connected:
            return False
            
        try:
            with self.arm_context(arm_type) as (port, packet, _, _, ids):
                if joint_index >= len(ids):
                    logger.error(f"Invalid joint index: {joint_index}")
                    return False
                    
                dxl_id = ids[joint_index]
                dxl_comm_result, dxl_error = packet.write4ByteTxRx(
                    port, dxl_id, self.ADDR_GOAL_POSITION, position
                )
                
                if dxl_comm_result != COMM_SUCCESS:
                    logger.error(f"Communication error: {packet.getTxRxResult(dxl_comm_result)}")
                    return False
                elif dxl_error != 0:
                    logger.error(f"Dynamixel error: {packet.getRxPacketError(dxl_error)}")
                    return False
                    
                logger.info(f"Joint {joint_index} moved to {position}")
            return True
        except Exception as e:
            logger.error(f"Joint position error: {e}")
            return False
    
    def get_positions(self, arm_type):
        """Get current positions of all motors in an arm"""
        if not self.is_connected:
            return [None] * len(self.arms[arm_type]['ids'])
            
        try:
            with self.arm_context(arm_type) as (port, packet, _, sync_reader, ids):
                # Sync read present position
                dxl_comm_result = sync_reader.txRxPacket()
                
                if dxl_comm_result != COMM_SUCCESS:
                    logger.error(f"Communication error: {packet.getTxRxResult(dxl_comm_result)}")
                    return [None] * len(ids)
                
                positions = []
                for dxl_id in ids:
                    # Check if data is available
                    if sync_reader.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, 4):
                        # Get present position value
                        position = sync_reader.getData(dxl_id, self.ADDR_PRESENT_POSITION, 4)
                        positions.append(position)
                    else:
                        logger.error(f"Sync read failed for Dynamixel ID {dxl_id}")
                        positions.append(None)
                
                return positions
        except Exception as e:
            logger.error(f"Position read error: {e}")
            return [None] * len(self.arms[arm_type]['ids'])
    
    def disconnect(self):
        """Disconnect from all robot arms"""
        if not self.is_connected:
            return
        
        for arm_type in self.arms:
            try:
                # Disable torque for all motors
                self.disable_torque(arm_type)
                # Close port
                if arm_type in self.port_handlers:
                    self.port_handlers[arm_type].closePort()
            except Exception as e:
                logger.error(f"Disconnect error for {arm_type}: {e}")
        
        self._cleanup_connections()
        logger.info("Robot disconnected")


class RobotController:
    """Robot controller with vision processing"""
    
    def __init__(self, robot, model_path=None, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.width, self.height = 640, 480
        self.running = False
        self.control_active = False
        self.data_lock = threading.Lock()
        self.last_frames = [None, None]
        self.last_data = [None] * 8
        self.tip = [(0, 0), (0, 0)]
        self.hand_detected = [False, False]
        self.z = 10
        self.serial_port = None
        self.cams = [None, None]
        
        # Load neural network model
        self.params = self._load_model(model_path)
        
        # Initialize MediaPipe once for better performance
        self.hands = [
            mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1,
                max_num_hands=2,
                static_image_mode=False
            ) for _ in range(2)
        ]
        
        # Warm up models with dummy image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            hand.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
    
    def _load_model(self, model_path):
        """Load the neural network model"""
        if not model_path:
            return None
            
        try:
            params = np.load(model_path)
            model_params = {k: params[k] for k in params.files}
            logger.info("Model loaded successfully")
            return model_params
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return None
    
    def start(self, cam_ids=(0, 2), serial_port='/dev/ttyUSB0'):
        """Start the controller with cameras and serial port"""
        if self.running:
            return True
        
        # Open cameras
        if not self._open_cameras(cam_ids):
            return False
        
        # Connect to serial port
        try:
            self.serial_port = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(2)
            logger.info(f"Serial port connected: {serial_port}")
        except Exception as e:
            logger.error(f"Serial port error: {e}")
            self._cleanup_cameras()
            return False
        
        # Start processing threads
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        threading.Thread(target=self._serial_listener, daemon=True).start()
        
        logger.info("Controller started")
        return True
    
    def _open_cameras(self, cam_ids):
        """Open and configure cameras"""
        for i, cid in enumerate(cam_ids):
            try:
                cap = cv2.VideoCapture(cid)
                if not cap.isOpened():
                    logger.error(f"Camera {i+1} (ID {cid}) failed to open")
                    self._cleanup_cameras()
                    return False
                
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                
                self.cams[i] = cap
                logger.info(f"Camera {i+1} (ID {cid}) initialized")
            except Exception as e:
                logger.error(f"Camera {i+1} error: {e}")
                self._cleanup_cameras()
                return False
        
        return True
    
    def _serial_listener(self):
        """Listen for commands from the serial port"""
        logger.info("Serial listener started")
        
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line.startswith("CMD:ROBOT:"):
                        mode = int(line.split(":")[-1])
                        position = BUTTON_POSITIONS.get(mode)
                        if position is not None:
                            self.robot.set_joint_position(0, position, 'leader')
                            logger.info(f"Button mode {mode}: moved to {position}")
                            
                            # Emit the mode change to WebSocket clients
                            socketio.emit('mode_change', {'mode': mode, 'position': position})
                            
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Serial error: {e}")
                time.sleep(1)
    
    def _process_frame(self, idx):
        """Process a single camera frame with hand detection"""
        if self.cams[idx] is None:
            return self._create_dummy_frame(f"Camera {idx+1} not ready")
        
        # Grab a frame, skip buffered frames
        frame = None
        for _ in range(2):  # Less aggressive buffer flushing
            ret, f = self.cams[idx].read()
            if ret:
                frame = f
        
        if frame is None:
            return self._create_dummy_frame("No frame available")
        
        try:
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands[idx].process(rgb)
            
            # Reset hand detection
            self.hand_detected[idx] = False
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                # Get index finger tip (landmark 8)
                landmark = results.multi_hand_landmarks[0].landmark[8]
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Store coordinates
                self.tip[idx] = (x, y)
                self.hand_detected[idx] = True
                
                # Use second camera for Z coordinate
                if idx == 1:
                    self.z = y
                
                # Draw circle at fingertip
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
                # Add hand tracking visualization
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
        
        return frame
    
    def _create_dummy_frame(self, message):
        """Create a blank frame with a message"""
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(
            frame, message, (70, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        return frame
    
    def _predict(self, x, y, z):
        """Predict joint positions from hand coordinates"""
        if not self.params:
            return np.zeros(4)
        
        # Normalize input
        A = np.array([[x], [y], [z]]) / 700.0
        
        # Forward pass through neural network
        L = len(self.params) // 2
        for l in range(1, L):
            A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
        
        # Scale output to motor positions
        return ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()
    
    def _process_loop(self):
        """Main processing loop for vision and control"""
        fps_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                # Process frames from both cameras
                frames = [self._process_frame(i) for i in range(2)]
                
                # Get current robot joint positions
                positions = self.robot.get_positions(self.arm_type)
                
                # Control the robot if hand is detected
                if self.control_active and self.hand_detected[0]:
                    # Predict joint positions from hand position
                    joints = self._predict(*self.tip[0], self.z).round().astype(int)
                    
                    if np.sum(np.abs(joints)) > 0:
                        self.robot.move(joints, self.arm_type)
                
                # Update data with thread safety
                with self.data_lock:
                    self.last_frames = frames
                    self.last_data = [
                        self.tip[0][0] if self.hand_detected[0] else None,
                        self.tip[0][1] if self.hand_detected[0] else None,
                        self.tip[1][0] if self.hand_detected[1] else None,
                        self.tip[1][1] if self.hand_detected[1] else None
                    ] + positions
                
                # Add FPS counter to frames
                frame_count += 1
                if time.time() - fps_time > 1.0:
                    fps = frame_count / (time.time() - fps_time)
                    fps_time = time.time()
                    frame_count = 0
                    
                    for i in range(2):
                        if frames[i] is not None:
                            cv2.putText(
                                frames[i], f"FPS: {fps:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2
                            )
                
                # Emit frames and status via WebSockets
                self._emit_status()
                
                # Adaptive sleep for consistent frame rate
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        
        self._cleanup()
    
    def _emit_status(self):
        """Emit status and frames via WebSockets"""
        try:
            # Convert frames to JPEG for transmission
            encoded_frames = []
            
            for i, frame in enumerate(self.last_frames):
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    encoded_frames.append(jpeg.tobytes())
                else:
                    encoded_frames.append(None)
            
            # Prepare data for emission
            data = {}
            
            # Only send frames if clients are connected
            if len(socketio.server.environ) > 0:
                # Convert frames to base64 for WebSocket transmission
                import base64
                data['frames'] = [
                    f"data:image/jpeg;base64,{base64.b64encode(frame).decode('utf-8')}" 
                    if frame else None for frame in encoded_frames
                ]
            
            # Send sensor data
            data['status'] = dict(zip(CSV_HEADERS, [
                v.item() if isinstance(v, np.generic) else v 
                for v in self.last_data
            ]))
            
            # Emit to connected clients
            socketio.emit('status_update', data)
            
        except Exception as e:
            logger.error(f"WebSocket emission error: {e}")
    
    def start_control(self):
        """Start robot control mode"""
        if not self.running or self.control_active:
            return False
            
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            logger.info("Robot control activated")
            socketio.emit('control_status', {'active': True})
        return success
    
    def stop_control(self):
        """Stop robot control mode"""
        if self.control_active:
            self.control_active = False
            self.robot.disable_torque(self.arm_type)
            logger.info("Robot control deactivated")
            socketio.emit('control_status', {'active': False})
        return True
    
    def save_snapshot(self):
        """Save current data to CSV file"""
        data = self.get_last_data()
        try:
            file_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
            
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(CSV_HEADERS)
                writer.writerow(data)
                
            logger.info("Snapshot saved to CSV")
            socketio.emit('snapshot_saved', {'success': True})
            return True
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            socketio.emit('snapshot_saved', {'success': False, 'error': str(e)})
            return False
    
    def get_last_frame(self, idx):
        """Thread-safe access to the last processed frame"""
        with self.data_lock:
            frame = self.last_frames[idx]
            return frame.copy() if frame is not None else self._create_dummy_frame("No frame")
    
    def get_last_data(self):
        """Thread-safe access to the last collected data"""
        with self.data_lock:
            return self.last_data.copy()
    
    def _cleanup_cameras(self):
        """Clean up camera resources"""
        for i, cam in enumerate(self.cams):
            if cam:
                try:
                    cam.release()
                except:
                    pass
                self.cams[i] = None
    
    def _cleanup(self):
        """Clean up all resources"""
        self._cleanup_cameras()
        
        # Clean up MediaPipe resources
        for hand in self.hands:
            try:
                hand.close()
            except:
                pass
        
        # Close serial port
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
    
    def stop(self):
        """Stop the controller and clean up resources"""
        if not self.running:
            return
            
        self.running = False
        self.control_active = False
        logger.info("Controller stopping")


# Initialize global objects
robot = None
controller = None


# Flask routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial status
    if controller and controller.running:
        data = controller.get_last_data()
        emit('status_update', {
            'status': dict(zip(CSV_HEADERS, [
                v.item() if isinstance(v, np.generic) else v 
                for v in data
            ])),
            'control_active': controller.control_active
        })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('take_snapshot')
def handle_take_snapshot():
    """Handle snapshot request from client"""
    if not controller or not controller.running:
        emit('snapshot_saved', {"success": False, "error": "System not ready"})
        return
    
    success = controller.save_snapshot()
    emit('snapshot_saved', {"success": success})


@socketio.on('start_control')
def handle_start_control():
    """Handle request to start robot control"""
    if not controller or not controller.running:
        emit('control_status', {"success": False, "error": "System not ready"})
        return
    
    success = controller.start_control()
    emit('control_status', {"success": success, "active": controller.control_active})


@socketio.on('stop_control')
def handle_stop_control():
    """Handle request to stop robot control"""
    if not controller or not controller.running:
        emit('control_status', {"success": False, "error": "System not ready"})
        return
    
    success = controller.stop_control()
    emit('control_status', {"success": success, "active": controller.control_active})


@socketio.on('set_mode')
def handle_set_mode(data):
    """Handle mode change request"""
    if not controller or not controller.running:
        emit('mode_change', {"success": False, "error": "System not ready"})
        return
    
    mode = data.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        emit('mode_change', {"success": False, "error": f"Invalid mode: {mode}"})
        return
    
    success = robot.set_joint_position(0, position, 'leader')
    emit('mode_change', {"success": success, "mode": mode, "position": position})


# RESTful API routes for backward compatibility
@app.route('/status')
def status():
    """Get current system status"""
    if not controller or not controller.running:
        return jsonify({"error": "System not ready"})
    
    data = controller.get_last_data()
    safe_data = []
    
    for v in data:
        if v is None:
            safe_data.append(None)
        elif isinstance(v, np.generic):
            safe_data.append(v.item())
        else:
            safe_data.append(int(v) if isinstance(v, (int, float)) else v)
    
    return jsonify(dict(zip(CSV_HEADERS, safe_data)))


@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    """Save a data snapshot via REST API"""
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.save_snapshot()})


@app.route('/control/start', methods=['POST'])
def start_control():
    """Start robot control via REST API"""
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.start_control()})


@app.route('/control/stop', methods=['POST'])
def stop_control():
    """Stop robot control via REST API"""
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.stop_control()})


@app.route('/absolute_position', methods=['POST'])
def absolute_position():
    """Set absolute position via REST API"""
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    
    mode = request.json.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        return jsonify({"success": False, "error": f"Invalid mode: {mode}"})
    
    success = robot.set_joint_position(0, position, 'leader')
    return jsonify({"success": success, "mode": mode, "position": position})


def init_system(model_path=MODEL_PATH, cam_ids=(0, 2), serial_port='/dev/ttyUSB0'):
    """Initialize the robot and controller system"""
    global robot, controller
    
    try:
        # Initialize robot
        robot = DynamixelRobot()
        if not robot.connect():
            logger.error("Robot connection failed")
            return False
        
        # Setup initial torque and control modes
        robot.disable_torque('follower')
        robot.setup_control('leader')
        
        # Initialize controller
        controller = RobotController(robot, model_path, 'follower')
        if not controller.start(cam_ids, serial_port):
            logger.error("Controller start failed")
            robot.disconnect()
            return False
        
        logger.info("=== System ready ===")
        return True
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return False


def cleanup_system():
    """Clean up system resources"""
    global controller, robot
    
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
    
    logger.info("System shutdown complete")


def main():
    """Main application entry point"""
    # Start initialization in background thread
    init_thread = threading.Thread(target=init_system, daemon=True)
    init_thread.start()
    
    try:
        # Run the Flask app with SocketIO
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("User interrupt received")
    finally:
        cleanup_system()


if __name__ == "__main__":
    main()