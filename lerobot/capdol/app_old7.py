#!/usr/bin/env python3
import os, json, time, threading, logging, numpy as np, cv2, csv, serial
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths and constants
BASE_DIR = "lerobot/capdol"
CALIBRATION_DIR, DATA_DIR = f"{BASE_DIR}/calibration_data", f"{BASE_DIR}/collected_data"
MODEL_PATH = f"{BASE_DIR}/models/model_parameters_resnet.npz"
CSV_PATH = os.path.join(DATA_DIR, "robot_hand_data.csv")
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
               "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]
BUTTON_POSITIONS = {0: 110, 1: 1110}  # Mode positions
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
mp_hands = mp.solutions.hands
robot, controller = None, None

class ManipulatorRobot:
    def __init__(self, config):
        self.arms = {
            'follower': make_motors_buses_from_configs(config.follower_arms),
            'leader': make_motors_buses_from_configs(config.leader_arms)
        }
        self.is_connected = False
        try:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            self.TorqueMode = TorqueMode
        except ImportError:
            self.TorqueMode = None

    def connect(self):
        if self.is_connected: return True
        try:
            for arm_type, buses in self.arms.items():
                for name, bus in buses.items():
                    bus.connect()
                    logger.info(f"Connected '{arm_type}-{name}'")
            self.is_connected = True
            self._load_calibration()
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def _load_calibration(self):
        path = os.path.join(CALIBRATION_DIR, 'follower_arm.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    calib_data = json.load(f)
                for bus in self.arms['follower'].values():
                    bus.set_calibration(calib_data)
                logger.info("Calibration loaded")
            except Exception as e:
                logger.error(f"Calibration error: {e}")

    def disable_torque(self, arm_type='follower'):
        if not self.is_connected: return False
        try:
            for bus in self.arms[arm_type].values():
                if self.TorqueMode:
                    bus.write('Torque_Enable', self.TorqueMode.DISABLED.value)
            return True
        except Exception as e:
            logger.error(f"Torque disable error: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if not self.is_connected: return False
        try:
            for bus in self.arms[arm_type].values():
                if self.TorqueMode:
                    bus.write('Torque_Enable', self.TorqueMode.DISABLED.value)
                bus.write('Operating_Mode', 3)
                bus.write('Torque_Enable', 1)
                bus.write('Position_P_Gain', 200)
            return True
        except Exception as e:
            logger.error(f"Control setup error: {e}")
            return False

    def move(self, positions, arm_type='follower'):
        if not self.is_connected: return False
        try:
            idx = 0
            for bus in self.arms[arm_type].values():
                count = len(bus.motor_names)
                bus.write('Goal_Position', positions[idx:idx+count])
                idx += count
            return True
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def set_joint_position(self, joint_index, position, arm_type='leader'):
        if not self.is_connected: return False
        try:
            positions = []
            for bus in self.arms[arm_type].values():
                positions.extend(bus.read('Present_Position'))
            
            positions[joint_index] = position
            idx = 0
            for bus in self.arms[arm_type].values():
                count = len(bus.motor_names)
                bus.write('Goal_Position', np.array(positions[idx:idx+count], dtype=np.uint32))
                idx += count
            logger.info(f"Joint {joint_index} moved to {position}")
            return True
        except Exception as e:
            logger.error(f"Joint position error: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if not self.is_connected: return [None] * 4
        try:
            positions = []
            for bus in self.arms[arm_type].values():
                positions.extend(bus.read('Present_Position'))
            return positions[:4]
        except Exception as e:
            logger.error(f"Position read error: {e}")
            return [None] * 4

    def disconnect(self):
        if not self.is_connected: return
        for buses in self.arms.values():
            for bus in buses.values():
                try: bus.disconnect()
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
        
        # Warm up models with dummy image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            hand.process(dummy)

    def start(self, cam_ids=(0, 2), serial_port='/dev/ttyUSB0'):
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
        logger.info("Controller started")
        return True

    def start_control(self):
        if not self.running or self.control_active: return False
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            logger.info("Robot control activated")
        return success

    def stop_control(self):
        if self.control_active:
            self.control_active = False
            self.robot.disable_torque(self.arm_type)
            logger.info("Robot control deactivated")
        return True

    def _serial_listener(self):
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
                logger.info(f"Camera {i+1} ready")
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
                    
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        self._cleanup()

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

    def save_snapshot(self):
        data = self.get_last_data()
        try:
            file_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(CSV_HEADERS)
                writer.writerow(data)
            logger.info("Snapshot saved")
            return True
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return False

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

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.save_snapshot()})

@app.route('/status')
def status():
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

@app.route('/control/start', methods=['POST'])
def start_control():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.start_control()})

@app.route('/control/stop', methods=['POST'])
def stop_control():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    return jsonify({"success": controller.stop_control()})

@app.route('/absolute_position', methods=['POST'])
def absolute_position():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "System not ready"})
    
    mode = request.json.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        return jsonify({"success": False, "error": f"Invalid mode: {mode}"})
    
    success = robot.set_joint_position(0, position, 'leader')
    return jsonify({"success": success, "mode": mode, "position": position})

def init_system(model_path=MODEL_PATH, cam_ids=(0, 2), arm_type='follower', serial_port='/dev/ttyUSB0'):
    global robot, controller
    try:
        robot = ManipulatorRobot(KochRobotConfig())
        if not robot.connect():
            logger.error("Robot connection failed")
            return False
        
        robot.disable_torque(arm_type)
        robot.setup_control('leader')
        
        controller = RobotController(robot, model_path, arm_type)
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
    if controller:
        try: controller.stop()
        except: pass
    if robot:
        try: robot.disconnect()
        except: pass
    logger.info("System shutdown")

def main():
    threading.Thread(target=init_system, daemon=True).start()
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("User interrupt")
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()