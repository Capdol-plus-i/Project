#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial
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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands
robot, controller = None, None

# Helper functions for Dynamixel SDK
def DXL_LOBYTE(value): return value & 0xFF
def DXL_HIBYTE(value): return (value >> 8) & 0xFF
def DXL_LOWORD(value): return value & 0xFFFF
def DXL_HIWORD(value): return (value >> 16) & 0xFFFF

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
            self.is_connected = False
        except ImportError as e:
            logger.error(f"Dynamixel SDK import error: {e}")
            self.is_connected = False

    def connect(self):
        if self.is_connected: return True
        try:
            for arm_type, port_handler in self.port_handlers.items():
                if not port_handler.openPort() or not port_handler.setBaudRate(BAUDRATE):
                    logger.error(f"Failed to open port/set baudrate for {arm_type}")
                    self._cleanup_connections()
                    return False
                
                logger.info(f"Connected '{arm_type}' on {port_handler.getPortName()}")
                
                # Initialize sync instances
                self.sync_writers[arm_type] = self.GroupSyncWrite(
                    port_handler, self.packet_handlers[arm_type], 
                    self.ADDR_GOAL_POSITION, 4)
                self.sync_readers[arm_type] = self.GroupSyncRead(
                    port_handler, self.packet_handlers[arm_type],
                    self.ADDR_PRESENT_POSITION, 4)
                
                # Add parameters for sync read
                for dxl_id in self.motor_ids[arm_type]:
                    self.sync_readers[arm_type].addParam(dxl_id)
            
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._cleanup_connections()
            return False

    def _cleanup_connections(self):
        for arm_type, port_handler in self.port_handlers.items():
            try: port_handler.closePort()
            except: pass
        self.is_connected = False

    def disable_torque(self, arm_type='follower'):
        if not self.is_connected: return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            for dxl_id in self.motor_ids[arm_type]:
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            return True
        except Exception as e:
            logger.error(f"Torque disable error: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if not self.is_connected: return False
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
            return True
        except Exception as e:
            logger.error(f"Control setup error: {e}")
            return False

    def move(self, positions, arm_type='follower'):
        if not self.is_connected: return False
        try:
            sync_writer = self.sync_writers[arm_type]
            sync_writer.clearParam()
            
            for i, dxl_id in enumerate(self.motor_ids[arm_type]):
                if i < len(positions):
                    position = int(positions[i])
                    param_goal_position = [
                        DXL_LOBYTE(DXL_LOWORD(position)),
                        DXL_HIBYTE(DXL_LOWORD(position)),
                        DXL_LOBYTE(DXL_HIWORD(position)),
                        DXL_HIBYTE(DXL_HIWORD(position))
                    ]
                    sync_writer.addParam(dxl_id, param_goal_position)
            
            sync_writer.txPacket()
            return True
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def set_joint_position(self, joint_index, position, arm_type='leader'):
        if not self.is_connected or joint_index >= len(self.motor_ids[arm_type]): 
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            dxl_id = self.motor_ids[arm_type][joint_index]
            
            dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
                port_handler, dxl_id, self.ADDR_GOAL_POSITION, position)
            
            if dxl_comm_result != 0 or dxl_error != 0:
                logger.error(f"Joint position error: {dxl_comm_result}, {dxl_error}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Joint position error: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if not self.is_connected: 
            return [None] * len(self.motor_ids[arm_type])
        try:
            sync_reader = self.sync_readers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            
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
        for arm_type in self.port_handlers:
            try:
                self.disable_torque(arm_type)
                self.port_handlers[arm_type].closePort()
            except: pass
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
        if model_path:
            try:
                params = np.load(model_path)
                self.params = {k: params[k] for k in params.files}
                logger.info("Model loaded")
            except Exception as e:
                logger.error(f"Model load error: {e}")
        
        # Initialize MediaPipe
        self.hands = [mp_hands.Hands(
            model_complexity=1, 
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1, 
            max_num_hands=2,
            static_image_mode=False
        ) for _ in range(2)]
        
        # Warm up models
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            hand.process(dummy)

    def start(self, cam_ids=(2, 0), serial_port='/dev/ttyUSB0'):
        if self.running: return True
        
        if not self._open_cams(cam_ids): return False
        
        try:
            self.serial_port = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(2)
            logger.info(f"Serial port connected")
        except Exception as e:
            logger.error(f"Serial port error: {e}")
            self._cleanup_cameras()
            return False
        
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        threading.Thread(target=self._serial_listener, daemon=True).start()
        socketio.emit('system_status', {'status': 'ready'})
        return True

    def start_control(self):
        if not self.running or self.control_active: return False
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            socketio.emit('control_status', {'active': True})
        return success

    def stop_control(self):
        if self.control_active:
            self.control_active = False
            self.robot.disable_torque(self.arm_type)
            socketio.emit('control_status', {'active': False})
        return True

    def _serial_listener(self):
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line.startswith("CMD:ROBOT:"):
                        mode = int(line.split(":")[-1])
                        position = BUTTON_POSITIONS.get(mode)
                        if position is not None:
                            self.robot.set_joint_position(0, position, 'leader')
                            socketio.emit('robot_mode', {'mode': mode, 'position': position})
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Serial error: {e}")
                time.sleep(1)

    def _open_cams(self, cam_ids):
        for i, cid in enumerate(cam_ids):
            try:
                cap = cv2.VideoCapture(cid)
                if not cap.isOpened():
                    logger.error(f"Camera {i+1} open failed")
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cams[i] = cap
            except Exception as e:
                logger.error(f"Camera error: {e}")
                return False
        return True

    def _predict(self, x, y, z):
        if not self.params: return np.zeros(4)
        A = np.array([[x],[y],[z]])/700.0
        L = len(self.params)//2
        for l in range(1, L):
            A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
        return ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()

    def _process_frame(self, idx):
        if not self.cams[idx]:
            return self._create_dummy_frame(f"Camera {idx+1} not ready")
            
        frame = None
        for _ in range(4):  # Flush buffer
            ret, f = self.cams[idx].read()
            if ret: frame = f
                
        if frame is None:
            return self._create_dummy_frame("No frame")

        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands[idx].process(rgb)
            
            self.hand_detected[idx] = False
            
            if results.multi_hand_landmarks:
                landmark = results.multi_hand_landmarks[0].landmark[8]  # Index finger tip
                x, y = int(landmark.x * w), int(landmark.y * h)
                self.tip[idx] = (x, y)
                self.hand_detected[idx] = True
                
                if idx == 1:  # Z coordinate from second camera
                    self.z = y
                    
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            
        return frame

    def _create_dummy_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, message, (70, 240), 1, 2, (255, 255, 255), 2)
        return frame

    def _process_loop(self):
        while self.running:
            try:
                frames = [self._process_frame(i) for i in range(2)]
                positions = self.robot.get_positions(self.arm_type)
                
                if self.control_active and self.hand_detected[0]:
                    joints = self._predict(*self.tip[0], self.z).round().astype(int)
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
            try: self.serial_port.close()
            except: pass
            self.serial_port = None

    def _cleanup_cameras(self):
        for hand in self.hands:
            try: hand.close()
            except: pass
        for i, cam in enumerate(self.cams):
            if cam:
                try: cam.release()
                except: pass
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

def generate_frames(cam_idx):
    while True:
        try:
            if controller and controller.running:
                frame = controller.get_last_frame(cam_idx)
            else:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "System starting...", (70, 240), 1, 2, (255, 255, 255), 2)
                
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            if not controller or not controller.running:
                time.sleep(0.5)
        except Exception:
            time.sleep(0.5)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    if controller and controller.running:
        socketio.emit('system_status', {'status': 'ready'})
        socketio.emit('control_status', {'active': controller.control_active})
    else:
        socketio.emit('system_status', {'status': 'initializing'})

@socketio.on('start_control')
def handle_start_control():
    if not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    return {'success': controller.start_control()}

@socketio.on('stop_control')
def handle_stop_control():
    if not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    return {'success': controller.stop_control()}

@socketio.on('set_robot_mode')
def handle_set_robot_mode(data):
    if not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    
    mode = data.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        return {'success': False, 'error': f'Invalid mode: {mode}'}
    
    success = robot.set_joint_position(0, position, 'leader')
    return {'success': success, 'mode': mode, 'position': position}

def init_system(model_path=MODEL_PATH, cam_ids=(2, 0), serial_port='/dev/ttyUSB0'):
    global robot, controller
    try:
        robot = ManipulatorRobot()
        if not robot.connect():
            socketio.emit('system_status', {'status': 'error', 'message': 'Robot connection failed'})
            return False
        
        robot.disable_torque('follower')
        robot.setup_control('leader')
        
        controller = RobotController(robot, model_path, 'follower')
        if not controller.start(cam_ids, serial_port):
            robot.disconnect()
            socketio.emit('system_status', {'status': 'error', 'message': 'Controller start failed'})
            return False
            
        socketio.emit('system_status', {'status': 'ready'})
        return True
    except Exception as e:
        socketio.emit('system_status', {'status': 'error', 'message': str(e)})
        return False

def cleanup_system():
    if controller:
        try: controller.stop()
        except: pass
    if robot:
        try: robot.disconnect()
        except: pass

def main():
    threading.Thread(target=init_system, daemon=True).start()
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()