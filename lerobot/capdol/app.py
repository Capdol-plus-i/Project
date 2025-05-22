#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial, csv
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import mediapipe as mp
import pyaudio
from six.moves import queue
from google.cloud import speech

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "lerobot/capdol/models/model_parameters_resnet.npz"
BUTTON_POSITIONS = {0: 110, 1: 1110}
PORTS = {"follower": "/dev/ttyACM0", "leader": "/dev/ttyACM1", "serial": "/dev/ttyACM2"}
BAUDRATE = 1000000
MOTOR_IDS = {"follower": [1, 2, 3, 4], "leader": [1]}
CSV_FILE = "lerobot/capdol/robot_data_snapshots.csv"
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
               "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]

# Voice Recognition
RATE, CHUNK = 16000, int(16000 / 5)
WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇"]
COMMANDS = {"왼쪽": ["왼쪽", "왼"], "오른쪽": ["오른쪽", "오른"], "위": ["위", "위로"], "아래": ["아래", "아래로"], "종료": ["종료", "끝내"]}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands

# Helper functions
def DXL_LOBYTE(v): return v & 0xFF
def DXL_HIBYTE(v): return (v >> 8) & 0xFF
def DXL_LOWORD(v): return v & 0xFFFF
def DXL_HIWORD(v): return (v >> 16) & 0xFFFF

class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate, self._chunk = rate, chunk
        self._buff, self.closed = queue.Queue(), True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16, channels=1, rate=self._rate, input=True,
            frames_per_buffer=self._chunk, stream_callback=self._fill_buffer)
        self.closed = False
        return self

    def __exit__(self, *args):
        self._stream.stop_stream()
        self._stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None: return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None: return
                    data.append(chunk)
                except queue.Empty: break
            yield b"".join(data)

class RobotController:
    def __init__(self, model_path=None):
        self.setup_dynamixel()
        self.load_model(model_path)
        self.init_vision()
        self.init_state()

    def setup_dynamixel(self):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
            self.PortHandler, self.PacketHandler = PortHandler, PacketHandler
            self.GroupSyncWrite, self.GroupSyncRead = GroupSyncWrite, GroupSyncRead
            
            self.ports = {k: self.PortHandler(v) for k, v in PORTS.items() if k != 'serial'}
            self.handlers = {k: self.PacketHandler(2.0) for k in self.ports}
            self.sync_writers, self.sync_readers = {}, {}
            
            # Control table addresses
            self.ADDR_TORQUE = 64
            self.ADDR_GOAL_POS = 116
            self.ADDR_PRESENT_POS = 132
            self.ADDR_OP_MODE = 11
            self.ADDR_POSITION_P_GAIN = 84
            
        except ImportError as e:
            logger.error(f"Dynamixel SDK import error: {e}")
            self.ports = {}

    def load_model(self, model_path):
        self.params = None
        if model_path and os.path.exists(model_path):
            try:
                params = np.load(model_path)
                self.params = {k: params[k] for k in params.files}
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Model load error: {e}")

    def init_vision(self):
        self.hands = [mp_hands.Hands(model_complexity=1, min_detection_confidence=0.1,
                                   min_tracking_confidence=0.1, max_num_hands=2,
                                   static_image_mode=False) for _ in range(2)]
        # Warm up
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            hand.process(dummy)

    def init_state(self):
        self.running = False
        self.control_active = False
        self.cams = [None, None]
        self.serial_port = None
        self.width, self.height = 640, 480
        self.tip = [(0, 0), (0, 0)]
        self.hand_detected = [False, False]
        self.z = 10
        self.data_lock = threading.Lock()
        self.last_frames = [None, None]
        self.last_data = [None] * 8

    def connect(self):
        if not self.ports:
            return False
            
        try:
            for arm_type, port in self.ports.items():
                if not port.openPort() or not port.setBaudRate(BAUDRATE):
                    logger.error(f"Failed to connect {arm_type}")
                    return False
                
                # Initialize sync instances
                self.sync_writers[arm_type] = self.GroupSyncWrite(port, self.handlers[arm_type], self.ADDR_GOAL_POS, 4)
                self.sync_readers[arm_type] = self.GroupSyncRead(port, self.handlers[arm_type], self.ADDR_PRESENT_POS, 4)
                
                # Add parameters
                for motor_id in MOTOR_IDS[arm_type]:
                    self.sync_readers[arm_type].addParam(motor_id)
            
            logger.info("Robot connected")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if arm_type not in self.ports:
            return False
            
        port, handler = self.ports[arm_type], self.handlers[arm_type]
        
        for motor_id in MOTOR_IDS[arm_type]:
            # Disable torque, set mode, enable torque
            handler.write1ByteTxRx(port, motor_id, self.ADDR_TORQUE, 0)
            handler.write1ByteTxRx(port, motor_id, self.ADDR_OP_MODE, 3)
            handler.write2ByteTxRx(port, motor_id, self.ADDR_POSITION_P_GAIN, 100)
            handler.write1ByteTxRx(port, motor_id, self.ADDR_TORQUE, 1)
        return True

    def move_joints(self, positions, arm_type='follower'):
        if arm_type not in self.sync_writers:
            return False
            
        try:
            writer = self.sync_writers[arm_type]
            writer.clearParam()
            
            for i, motor_id in enumerate(MOTOR_IDS[arm_type]):
                if i < len(positions):
                    pos = int(positions[i])
                    param = [DXL_LOBYTE(DXL_LOWORD(pos)), DXL_HIBYTE(DXL_LOWORD(pos)),
                            DXL_LOBYTE(DXL_HIWORD(pos)), DXL_HIBYTE(DXL_HIWORD(pos))]
                    writer.addParam(motor_id, param)
            
            writer.txPacket()
            return True
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if arm_type not in self.sync_readers:
            return [None] * len(MOTOR_IDS[arm_type])
            
        try:
            reader = self.sync_readers[arm_type]
            if reader.txRxPacket() != 0:
                return [None] * len(MOTOR_IDS[arm_type])
            
            positions = []
            for motor_id in MOTOR_IDS[arm_type]:
                if reader.isAvailable(motor_id, self.ADDR_PRESENT_POS, 4):
                    positions.append(reader.getData(motor_id, self.ADDR_PRESENT_POS, 4))
                else:
                    positions.append(None)
            return positions
        except:
            return [None] * len(MOTOR_IDS[arm_type])

    def set_joint_position(self, joint_idx, position, arm_type='leader'):
        if arm_type not in self.ports or joint_idx >= len(MOTOR_IDS[arm_type]):
            return False
            
        try:
            port, handler = self.ports[arm_type], self.handlers[arm_type]
            motor_id = MOTOR_IDS[arm_type][joint_idx]
            result, error = handler.write4ByteTxRx(port, motor_id, self.ADDR_GOAL_POS, position)
            return result == 0 and error == 0
        except:
            return False

    def predict_positions(self, x, y, z):
        if not self.params:
            return np.zeros(4)
            
        A = np.array([[x], [y], [z]]) / 700.0
        L = len(self.params) // 2
        
        for l in range(1, L):
            A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
        
        return ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()

    def start(self, cam_ids=(0, 2)):
        if self.running:
            return True
            
        if not self._open_cameras(cam_ids) or not self._open_serial():
            return False
            
        self.running = True
        threading.Thread(target=self._main_loop, daemon=True).start()
        threading.Thread(target=self._serial_loop, daemon=True).start()
        return True

    def _open_cameras(self, cam_ids):
        for i, cam_id in enumerate(cam_ids):
            try:
                cap = cv2.VideoCapture(cam_id)
                if not cap.isOpened():
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cams[i] = cap
            except:
                return False
        return True

    def _open_serial(self):
        try:
            self.serial_port = serial.Serial(PORTS['serial'], 9600, timeout=1)
            time.sleep(3)  # Arduino boot wait
            self.serial_port.reset_input_buffer()
            return True
        except:
            return False

    def _process_frame(self, idx):
        if not self.cams[idx]:
            return self._create_dummy_frame(f"Camera {idx+1} not ready")
            
        # Flush buffer and get latest frame
        frame = None
        for _ in range(1):
            ret, f = self.cams[idx].read()
            if ret:
                frame = f
                
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
                
                if idx == 1:  # Z from second camera
                    self.z = y
                    
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            
        return frame

    def _create_dummy_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, message, (70, 240), 1, 2, (255, 255, 255), 2)
        return frame

    def _main_loop(self):
        while self.running:
            try:
                # Process frames
                frames = [self._process_frame(i) for i in range(2)]
                positions = self.get_positions('follower')
                
                # Control robot if active and hand detected
                if self.control_active and self.hand_detected[0]:
                    joints = self.predict_positions(*self.tip[0], self.z).round().astype(int)
                    if np.sum(np.abs(joints)) > 0:
                        self.move_joints(joints, 'follower')
                
                # Update shared data
                with self.data_lock:
                    self.last_frames = frames
                    self.last_data = [
                        self.tip[0][0] if self.hand_detected[0] else None,
                        self.tip[0][1] if self.hand_detected[0] else None,
                        self.tip[1][0] if self.hand_detected[1] else None,
                        self.tip[1][1] if self.hand_detected[1] else None
                    ] + positions
                
                # Emit status
                self._emit_status()
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.1)

    def _serial_loop(self):
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.startswith("CMD:ROBOT:"):
                        mode = int(line.split(":")[-1])
                        position = BUTTON_POSITIONS.get(mode)
                        if position and self.set_joint_position(0, position, 'leader'):
                            socketio.emit('robot_mode', {'mode': mode, 'position': position})
                    
                    elif line.startswith("CMD:LED:"):
                        brightness = int(line.split(":")[-1])
                        socketio.emit('led_brightness', {'level': brightness})
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Serial error: {e}")
                time.sleep(1)

    def _emit_status(self):
        data = self.get_last_data()
        safe_data = [v.item() if isinstance(v, np.generic) else v for v in data]
        socketio.emit('status_update', dict(zip(CSV_HEADERS, safe_data)))

    def get_last_frame(self, idx):
        with self.data_lock:
            frame = self.last_frames[idx]
            return frame.copy() if frame is not None else self._create_dummy_frame("No frame")

    def get_last_data(self):
        with self.data_lock:
            return self.last_data.copy()

    def start_control(self):
        if self.control_active:
            return False
        success = self.setup_control('follower')
        if success:
            self.control_active = True
            socketio.emit('control_status', {'active': True})
        return success

    def stop_control(self):
        if self.control_active:
            self.control_active = False
            # Disable torque
            if 'follower' in self.ports:
                for motor_id in MOTOR_IDS['follower']:
                    self.handlers['follower'].write1ByteTxRx(
                        self.ports['follower'], motor_id, self.ADDR_TORQUE, 0)
            socketio.emit('control_status', {'active': False})
        return True

    def disconnect(self):
        self.running = False
        self.control_active = False
        
        # Close cameras
        for cam in self.cams:
            if cam:
                cam.release()
        
        # Close serial
        if self.serial_port:
            self.serial_port.close()
        
        # Close robot ports
        for port in self.ports.values():
            try:
                port.closePort()
            except:
                pass

# Global controller instance
controller = None
voice_running = False

# Voice Recognition
def create_speech_stream(is_command_mode=False):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE, language_code="ko-KR",
        speech_contexts=[speech.SpeechContext(
            phrases=WAKE_WORDS + sum(COMMANDS.values(), []), boost=20.0)],
        model="command_and_search")
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True, single_utterance=is_command_mode)
    
    stream = MicrophoneStream()
    stream.__enter__()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) 
               for chunk in stream.generator())
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses

def process_voice_command(cmd):
    if not controller or not controller.running:
        return
        
    logger.info(f"Processing voice command: {cmd}")
    socketio.emit('voice_command', {'command': cmd})
    
    if not controller.control_active:
        return
        
    positions = controller.get_positions('follower')
    if not positions[0]:
        return
        
    # Simple movement commands
    moves = {
        "왼쪽": [positions[0] - 500] + positions[1:],
        "오른쪽": [positions[0] + 500] + positions[1:],
        "위": [positions[0], positions[1] + 500] + positions[2:] if len(positions) > 1 else positions,
        "아래": [positions[0], positions[1] - 500] + positions[2:] if len(positions) > 1 else positions
    }
    
    if cmd in moves:
        controller.move_joints(moves[cmd], 'follower')

def voice_recognition_loop():
    global voice_running
    voice_running = True
    socketio.emit('voice_status', {'status': 'ready', 'message': '웨이크워드를 말해주세요'})

    while voice_running:
        try:
            stream, responses = create_speech_stream()

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript.strip()
                
                # Check for wake word
                if any(wake in transcript for wake in WAKE_WORDS):
                    if result.is_final or result.stability > 0.8:
                        socketio.emit('voice_recognition', {'text': '명령을 말씀하세요', 'type': 'wake_word'})
                        stream.__exit__(None, None, None)
                        
                        # Command mode
                        cmd_stream, cmd_responses = create_speech_stream(is_command_mode=True)
                        start_time = time.time()
                        
                        for cmd_response in cmd_responses:
                            if time.time() - start_time > 5:  # Timeout
                                break
                                
                            if not cmd_response.results:
                                continue
                                
                            cmd_result = cmd_response.results[0]
                            if not cmd_result.alternatives:
                                continue

                            cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                            
                            for cmd, variations in COMMANDS.items():
                                if any(v in cmd_transcript for v in variations):
                                    if cmd == "종료":
                                        voice_running = False
                                        cmd_stream.__exit__(None, None, None)
                                        return
                                    else:
                                        process_voice_command(cmd)
                                    cmd_stream.__exit__(None, None, None)
                                    break
                            break
                        break
                        
            try:
                stream.__exit__(None, None, None)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Voice recognition error: {e}")
            time.sleep(2)

# Flask Routes
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
        except:
            time.sleep(0.5)

# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    if controller and controller.running:
        socketio.emit('system_status', {'status': 'ready'})
        socketio.emit('control_status', {'active': controller.control_active})
    else:
        socketio.emit('system_status', {'status': 'initializing'})

@socketio.on('start_control')
def handle_start_control(data=None):
    return {'success': controller.start_control() if controller else False}

@socketio.on('stop_control')
def handle_stop_control(data=None):
    return {'success': controller.stop_control() if controller else False}

@socketio.on('set_robot_mode')
def handle_set_robot_mode(data):
    if not controller:
        return {'success': False}
    mode = data.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    if position:
        success = controller.set_joint_position(0, position, 'leader')
        return {'success': success, 'mode': mode, 'position': position}
    return {'success': False}

@socketio.on('take_snapshot')
def handle_take_snapshot(data=None):
    if not controller:
        return {'success': False}
    
    try:
        snapshot_data = controller.get_last_data()
        file_exists = os.path.exists(CSV_FILE)
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADERS)
            writer.writerow(snapshot_data)
        
        return {'success': True, 'data': dict(zip(CSV_HEADERS, snapshot_data))}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('get_csv_info')
def handle_get_csv_info(data=None):
    try:
        if os.path.exists(CSV_FILE):
            file_size = os.path.getsize(CSV_FILE)
            with open(CSV_FILE, 'r') as f:
                row_count = sum(1 for row in csv.reader(f)) - 1  # Exclude header
            return {'success': True, 'exists': True, 'file_path': CSV_FILE, 
                   'file_size': file_size, 'row_count': row_count}
        else:
            return {'success': True, 'exists': False}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('clear_csv')
def handle_clear_csv(data=None):
    try:
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('send_serial_command')
def handle_send_serial_command(data):
    if not controller or not controller.serial_port:
        return {'success': False, 'error': 'Serial port not ready'}
    
    command = data.get('command', '')
    try:
        controller.serial_port.write(f"{command}\n".encode('utf-8'))
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('start_voice_recognition')
def handle_start_voice(data=None):
    global voice_running
    if voice_running:
        return {'success': False, 'message': '이미 실행 중'}
    threading.Thread(target=voice_recognition_loop, daemon=True).start()
    return {'success': True}

@socketio.on('stop_voice_recognition')
def handle_stop_voice(data=None):
    global voice_running
    voice_running = False
    return {'success': True}

def init_system():
    global controller
    try:
        controller = RobotController(MODEL_PATH)
        if not controller.connect():
            socketio.emit('system_status', {'status': 'error', 'message': 'Robot connection failed'})
            return False
        
        controller.setup_control('leader')
        
        if not controller.start():
            controller.disconnect()
            socketio.emit('system_status', {'status': 'error', 'message': 'Controller start failed'})
            return False
            
        socketio.emit('system_status', {'status': 'ready'})
        return True
    except Exception as e:
        socketio.emit('system_status', {'status': 'error', 'message': str(e)})
        return False

def cleanup():
    global voice_running
    voice_running = False
    if controller:
        controller.disconnect()

def main():
    threading.Thread(target=init_system, daemon=True).start()
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()