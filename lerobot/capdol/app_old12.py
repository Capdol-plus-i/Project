#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
import pyaudio
from six.moves import queue
from google.cloud import speech

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = f"lerobot/capdol/models/model_parameters_resnet.npz"
BUTTON_POSITIONS = {0: 110, 1: 1110}
PORT_ACM0, PORT_ACM1 = "/dev/ttyACM0", "/dev/ttyACM1"
BAUDRATE = 1000000
DYNAMIXEL_IDS_ACM0, DYNAMIXEL_IDS_ACM1 = [1, 2, 3, 4], [1]

# 개선된 음성 인식 설정
RATE = 16000
CHUNK = int(RATE / 5)  # 100ms 단위로 변경
WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이"]
CMD_MAP = {
    "왼쪽": ["왼쪽", "왼 쪽", "왼"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레"],
    "종료": ["종료", "끝내", "종료해"]
}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands
robot, controller = None, None
voice_recognition_running = False

# Helper functions for Dynamixel SDK
def DXL_LOBYTE(value): return value & 0xFF
def DXL_HIBYTE(value): return (value >> 8) & 0xFF
def DXL_LOWORD(value): return value & 0xFFFF
def DXL_HIWORD(value): return (value >> 16) & 0xFFFF

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._stream.stop_stream()
        self._stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

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
                                             self.ADDR_POSITION_P_GAIN, 100)
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

    def start(self, cam_ids=(0, 2), serial_port='/dev/ttyACM2'):
        if self.running: return True
        
        if not self._open_cams(cam_ids): return False
        
        try:
            self.serial_port = serial.Serial(
                serial_port, 
                9600, 
                timeout=1,
                write_timeout=1,
                inter_byte_timeout=0.1
            )
            logger.info(f"Serial port {serial_port} opened")
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
        # 시리얼 버퍼 초기화 (Arduino 부팅 노이즈 제거)
        if self.serial_port:
            time.sleep(3)  # Arduino 부팅 대기
            self.serial_port.reset_input_buffer()
            logger.info("Serial buffer cleared after Arduino boot")
        
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    # UTF-8 디코딩 오류를 안전하게 처리
                    raw_line = self.serial_port.readline()
                    try:
                        line = raw_line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        # 디코딩 실패 시 에러를 무시하고 계속 진행
                        logger.debug(f"Unicode decode error, skipping: {raw_line}")
                        continue
                    
                    # 유효한 명령어만 처리
                    if line and line.startswith("CMD:"):
                        logger.info(f"Received command: {line}")
                        
                        if line.startswith("CMD:ROBOT:"):
                            try:
                                mode = int(line.split(":")[-1])
                                position = BUTTON_POSITIONS.get(mode)
                                if position is not None:
                                    success = self.robot.set_joint_position(0, position, 'leader')
                                    if success:
                                        socketio.emit('robot_mode', {'mode': mode, 'position': position})
                                        logger.info(f"Robot mode changed to {mode}, position: {position}")
                                    else:
                                        logger.error(f"Failed to set robot position")
                                else:
                                    logger.warning(f"Invalid robot mode: {mode}")
                            except (ValueError, IndexError) as e:
                                logger.error(f"Invalid robot command format: {line}")
                        
                        elif line.startswith("CMD:LED:"):
                            try:
                                brightness = int(line.split(":")[-1])
                                socketio.emit('led_brightness', {'level': brightness})
                                logger.info(f"LED brightness changed to level {brightness}")
                            except (ValueError, IndexError) as e:
                                logger.error(f"Invalid LED command format: {line}")
                
                time.sleep(0.1)
            except serial.SerialException as e:
                logger.error(f"Serial connection error: {e}")
                time.sleep(2)
                # 시리얼 재연결 시도
                try:
                    if self.serial_port and not self.serial_port.is_open:
                        self.serial_port.open()
                        logger.info("Serial port reconnected")
                except:
                    logger.error("Failed to reconnect serial port")
            except Exception as e:
                logger.error(f"Unexpected serial error: {e}")
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

# 개선된 음성 인식 관련 함수들
def start_stream(is_command_mode=False):
    client = speech.SpeechClient()

    # 인식 우선순위 키워드 등록
    speech_context = speech.SpeechContext(
        phrases=WAKE_WORDS + sum(CMD_MAP.values(), []),
        boost=20.0
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        model="command_and_search"
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=is_command_mode  # 명령어 모드일 때는 말 끝나면 자동 종료
    )

    stream = MicrophoneStream(RATE, CHUNK)
    stream.__enter__()
    audio_generator = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses

def process_voice_command(cmd):
    global controller
    
    if not controller or not controller.running:
        logger.warning("Voice command received but robot not ready")
        return
    
    logger.info(f"Processing voice command: {cmd}")
    socketio.emit('voice_command', {'command': cmd})
    
    # 명령어에 따른 로봇 제어
    if cmd == "왼쪽":
        logger.info("Moving robot left")
        if controller.control_active:
            current_positions = controller.robot.get_positions(controller.arm_type)
            if current_positions[0] is not None:
                new_position = [current_positions[0] - 500] + current_positions[1:]
                controller.robot.move(new_position, controller.arm_type)
    elif cmd == "오른쪽":
        logger.info("Moving robot right")
        if controller.control_active:
            current_positions = controller.robot.get_positions(controller.arm_type)
            if current_positions[0] is not None:
                new_position = [current_positions[0] + 500] + current_positions[1:]
                controller.robot.move(new_position, controller.arm_type)
    elif cmd == "위":
        logger.info("Moving robot up")
        if controller.control_active:
            current_positions = controller.robot.get_positions(controller.arm_type)
            if len(current_positions) > 1 and current_positions[1] is not None:
                new_position = [current_positions[0], current_positions[1] + 500] + current_positions[2:]
                controller.robot.move(new_position, controller.arm_type)
    elif cmd == "아래":
        logger.info("Moving robot down")
        if controller.control_active:
            current_positions = controller.robot.get_positions(controller.arm_type)
            if len(current_positions) > 1 and current_positions[1] is not None:
                new_position = [current_positions[0], current_positions[1] - 500] + current_positions[2:]
                controller.robot.move(new_position, controller.arm_type)

def voice_recognition_loop():
    global voice_recognition_running
    voice_recognition_running = True
    logger.info("🎙️ 음성 인식 시스템이 준비되었습니다. 웨이크워드를 말해주세요 (예: '하이봇')")
    socketio.emit('voice_status', {'status': 'ready', 'message': '웨이크워드를 말해주세요 (예: 하이봇)'})

    while voice_recognition_running:
        try:
            stream, responses = start_stream()

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript.strip()
                logger.info(f"📝 인식됨: {transcript}")
                socketio.emit('voice_recognition', {'text': transcript, 'type': 'listening'})

                # 웨이크워드 감지
                if any(wake in transcript for wake in WAKE_WORDS):
                    if result.is_final or result.stability > 0.8:
                        logger.info("✅ 웨이크워드 감지됨! 명령을 말씀하세요.")
                        socketio.emit('voice_recognition', {'text': '웨이크워드 감지됨! 명령을 말씀하세요.', 'type': 'wake_word'})
                        stream.__exit__(None, None, None)  # 기존 세션 종료
                        time.sleep(0.1)  # 스트림 안정화 대기

                        # 명령어 모드 진입
                        command_stream, command_responses = start_stream(is_command_mode=True)
                        time.sleep(0.1)  # 스트림 안정화 대기
                        start_time = time.time()
                        MAX_COMMAND_DURATION = 5  # 초과 시 자동 종료

                        for cmd_response in command_responses:
                            if time.time() - start_time > MAX_COMMAND_DURATION:
                                logger.info("⏰ 명령어 대기 시간 초과")
                                socketio.emit('voice_recognition', {'text': '명령어 대기 시간 초과', 'type': 'timeout'})
                                command_stream.__exit__(None, None, None)
                                break

                            if not cmd_response.results:
                                continue
                            cmd_result = cmd_response.results[0]
                            if not cmd_result.alternatives:
                                continue

                            cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                            logger.info(f"🎯 명령 인식: {cmd_transcript}")
                            socketio.emit('voice_recognition', {'text': cmd_transcript, 'type': 'command'})

                            for cmd, variations in CMD_MAP.items():
                                if any(v in cmd_transcript for v in variations):
                                    logger.info(f"✅ 명령어: {cmd}")
                                    socketio.emit('voice_recognition', {'text': f'명령 실행: {cmd}', 'type': 'executed'})
                                    
                                    if cmd == "종료":
                                        logger.info("🔚 음성 인식 종료")
                                        command_stream.__exit__(None, None, None)
                                        voice_recognition_running = False
                                        return
                                    else:
                                        # 로봇 명령 처리
                                        process_voice_command(cmd)

                                    command_stream.__exit__(None, None, None)
                                    break
                            break  # 한 번의 명령 처리 후 종료
                        break  # 웨이크워드 감지 후 루프 종료
                        
            # stream.__exit__가 호출되지 않은 경우 여기서 처리
            try:
                stream.__exit__(None, None, None)
            except:
                pass
                
        except Exception as e:
            logger.error(f"음성 인식 오류: {e}")
            socketio.emit('voice_status', {'status': 'error', 'message': f'음성 인식 오류: {str(e)}'})
            time.sleep(2)

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

@socketio.on('send_serial_command')
def handle_send_serial_command(data):
    """시리얼 명령 전송 (테스트용)"""
    if not controller or not controller.running or not controller.serial_port:
        return {'success': False, 'error': 'Serial port not ready'}
    
    command = data.get('command', '')
    try:
        controller.serial_port.write(f"{command}\n".encode('utf-8'))
        logger.info(f"Sent serial command: {command}")
        return {'success': True, 'message': f'Command sent: {command}'}
    except Exception as e:
        logger.error(f"Failed to send serial command: {e}")
        return {'success': False, 'error': str(e)}

@socketio.on('start_voice_recognition')
def handle_start_voice_recognition():
    global voice_recognition_running
    if voice_recognition_running:
        return {'success': False, 'message': '음성 인식이 이미 실행 중입니다.'}
    
    threading.Thread(target=voice_recognition_loop, daemon=True).start()
    return {'success': True, 'message': '음성 인식이 시작되었습니다.'}

@socketio.on('stop_voice_recognition')
def handle_stop_voice_recognition():
    global voice_recognition_running
    if not voice_recognition_running:
        return {'success': False, 'message': '음성 인식이 실행 중이 아닙니다.'}
    
    voice_recognition_running = False
    return {'success': True, 'message': '음성 인식이 중지되었습니다.'}

def init_system(model_path=MODEL_PATH, cam_ids=(0, 2), serial_port='/dev/ttyACM2'):
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
    global voice_recognition_running
    
    # 음성 인식 중지
    voice_recognition_running = False
    
    # 컨트롤러 및 로봇 정리
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