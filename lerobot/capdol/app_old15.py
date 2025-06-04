#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial, signal, sys, csv, re
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import mediapipe as mp

# Voice recognition imports
try:
    import pyaudio
    from six.moves import queue
    from google.cloud import speech
    VOICE_AVAILABLE = True
    print("✅ Voice recognition libraries loaded")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"⚠️ Voice recognition not available: {e}")

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = f"models/model_parameters_resnet.npz"
BUTTON_POSITIONS = {0: 110, 1: 1110}
PORT_ACM0, PORT_ACM1 = "/dev/ttyACM0", "/dev/ttyACM1"
ARDUINO_PORT = "/dev/ttyACM3"
BAUDRATE = 1000000
ARDUINO_BAUDRATE = 9600
DYNAMIXEL_IDS_ACM0, DYNAMIXEL_IDS_ACM1 = [1, 2, 3, 4], [1]

# Voice recognition configuration (참고 코드 기반)
VOICE_RATE = 16000
VOICE_CHUNK = int(VOICE_RATE / 5)  # 100ms 단위

WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이", "로봇아"]
VOICE_CMD_MAP = {
    "시작": ["시작", "시작해", "제어 시작", "컨트롤 시작"],
    "정지": ["정지", "멈춰", "스톱", "중지", "끝"],
    "모드0": ["모드 0", "모드0", "모드 영", "모드 제로"],
    "모드1": ["모드 1", "모드1", "모드 일", "모드 원"],
    "스냅샷": ["스냅샷", "사진", "캡처", "저장"],
    "밝게": ["밝게", "밝기 올려", "더 밝게", "라이트 업"],
    "어둡게": ["어둡게", "밝기 내려", "어둡게 해", "라이트 다운"],
    "리셋": ["리셋", "초기화", "재설정"],
    "왼쪽": ["왼쪽", "왼 쪽", "왼", "좌측"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른", "우측"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레"],
    "종료": ["종료", "끝내", "종료해", "시스템 종료"]
}

# CSV Configuration
CSV_FILENAME = "robot_snapshots.csv"
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y", 
               "joint_1", "joint_2", "joint_3", "joint_4"]
CSV_LOCK = threading.Lock()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
mp_hands = mp.solutions.hands
robot, controller, arduino_controller, voice_controller = None, None, None, None

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
            row_data = data
            with open(CSV_FILENAME, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            with open(CSV_FILENAME, 'r') as csvfile:
                total_snapshots = sum(1 for line in csvfile) - 1
            
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
            total_snapshots = max(0, total_lines - 1)
            
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

class MicrophoneStream:
    """참고 코드와 동일한 MicrophoneStream 클래스"""
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

class VoiceController:
    def __init__(self):
        self.enabled = VOICE_AVAILABLE
        self.running = False
        self.listening = False
        self.current_stream = None
        self.client = None
        
        # 음성 상태 추적
        self.voice_status = {
            'enabled': self.enabled,
            'listening': False,
            'last_transcript': '',
            'last_command': '',
            'wake_word_detected': False
        }
        
        if not self.enabled:
            logger.warning("Voice control disabled - missing dependencies")
            return
            
        try:
            self.client = speech.SpeechClient()
            logger.info("Voice controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice controller: {e}")
            self.enabled = False
    
    def start_stream(self, is_command_mode=False):
        """참고 코드와 동일한 스트림 생성 방식"""
        if not self.enabled:
            return None, None
            
        try:
            # 인식 우선순위 키워드 등록
            speech_context = speech.SpeechContext(
                phrases=WAKE_WORDS + [phrase for phrases in VOICE_CMD_MAP.values() for phrase in phrases],
                boost=20.0
            )

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=VOICE_RATE,
                language_code="ko-KR",
                speech_contexts=[speech_context],
                model="command_and_search"
            )

            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=is_command_mode  # 명령어 모드일 때는 말 끝나면 자동 종료
            )

            stream = MicrophoneStream(VOICE_RATE, VOICE_CHUNK)
            stream.__enter__()
            self.current_stream = stream
            
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
            responses = self.client.streaming_recognize(streaming_config, requests)
            
            return stream, responses
            
        except Exception as e:
            logger.error(f"Failed to create voice stream: {e}")
            return None, None
    
    def start(self):
        """음성 인식 시작"""
        if not self.enabled:
            logger.warning("Voice controller not available")
            return False
            
        if self.running:
            return True
            
        try:
            self.running = True
            self.listening = True
            self.voice_status['listening'] = True
            
            threading.Thread(target=self._voice_recognition_loop, daemon=True).start()
            logger.info("🎙️ Voice controller started - Say wake word to activate")
            
            # 웹 클라이언트에 상태 전송
            socketio.emit('voice_status_update', {
                'listening': True,
                'message': '웨이크워드를 말해주세요 (예: "하이봇")'
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to start voice controller: {e}")
            return False
    
    def stop(self):
        """음성 인식 중지"""
        self.running = False
        self.listening = False
        self.voice_status['listening'] = False
        
        if self.current_stream:
            try:
                self.current_stream.__exit__(None, None, None)
            except:
                pass
            self.current_stream = None
            
        socketio.emit('voice_status_update', {
            'listening': False,
            'message': 'Voice recognition stopped'
        })
        
        logger.info("Voice controller stopped")
    
    def _voice_recognition_loop(self):
        """참고 코드를 기반으로 한 음성 인식 메인 루프"""
        logger.info("🎙️ 시스템이 준비되었습니다. 웨이크워드를 말해주세요 (예: '하이봇')")

        while self.running:
            try:
                stream, responses = self.start_stream()
                if not stream or not responses:
                    time.sleep(1)
                    continue

                for response in responses:
                    if not self.running:
                        break
                        
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript.strip()
                    logger.debug(f"인식된 내용: {transcript}")
                    
                    # 웹 클라이언트에 인식 내용 전송
                    socketio.emit('voice_transcript', {'text': transcript})
                    
                    self.voice_status['last_transcript'] = transcript

                    # 웨이크워드 감지
                    if any(wake in transcript for wake in WAKE_WORDS):
                        if result.is_final or result.stability > 0.8:
                            logger.info(f"✅ 웨이크워드 감지됨: {transcript}")
                            self.voice_status['wake_word_detected'] = True
                            
                            # Arduino LED 효과
                            if arduino_controller and arduino_controller.is_connected():
                                arduino_controller.trigger_led_effect(2)  # 파란색 페이드
                            
                            # 웹 클라이언트에 웨이크워드 감지 알림
                            socketio.emit('voice_wake_word', {
                                'transcript': transcript,
                                'message': '웨이크워드 감지됨! 명령을 말씀하세요.'
                            })
                            
                            stream.__exit__(None, None, None)
                            time.sleep(0.1)
                            
                            # 명령어 모드 진입
                            self._command_mode()
                            break
            except Exception as e:
                logger.error(f"Voice recognition error: {e}")
                time.sleep(1)
    
    def _command_mode(self):
        """명령어 모드 - 참고 코드 패턴 적용"""
        try:
            logger.info("명령어 모드 진입")
            socketio.emit('voice_command_mode', {'active': True})
            
            # 명령어 모드 진입
            command_stream, command_responses = self.start_stream(is_command_mode=True)
            if not command_stream or not command_responses:
                return
                
            time.sleep(0.1)  # 스트림 안정화 대기
            start_time = time.time()
            MAX_COMMAND_DURATION = 5  # 초과 시 자동 종료

            for cmd_response in command_responses:
                if not self.running:
                    break
                    
                if time.time() - start_time > MAX_COMMAND_DURATION:
                    logger.info("명령어 대기 시간 초과")
                    socketio.emit('voice_timeout', {'message': '명령어 대기 시간 초과'})
                    command_stream.__exit__(None, None, None)
                    break

                if not cmd_response.results:
                    continue
                    
                cmd_result = cmd_response.results[0]
                if not cmd_result.alternatives:
                    continue

                cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                logger.info(f"말한 내용: {cmd_transcript}")
                
                # 웹 클라이언트에 명령어 전송
                socketio.emit('voice_command_transcript', {'text': cmd_transcript})
                
                # 명령어 처리
                command_found = False
                for cmd, variations in VOICE_CMD_MAP.items():
                    if any(v in cmd_transcript for v in variations):
                        logger.info(f"✅ 명령어 실행: {cmd}")
                        self.voice_status['last_command'] = cmd
                        
                        # 명령어 실행
                        success = self._execute_command(cmd, cmd_transcript)
                        
                        # 웹 클라이언트에 명령어 실행 결과 전송
                        socketio.emit('voice_command_executed', {
                            'command': cmd,
                            'transcript': cmd_transcript,
                            'success': success
                        })
                        
                        command_stream.__exit__(None, None, None)
                        command_found = True
                        break
                        
                if command_found:
                    break
                    
        except Exception as e:
            logger.error(f"Command mode error: {e}")
        finally:
            socketio.emit('voice_command_mode', {'active': False})
    
    def _execute_command(self, cmd, transcript):
        """음성 명령어 실행"""
        try:
            if cmd == "시작":
                if controller and controller.running:
                    success = controller.start_control()
                    if success:
                        logger.info("✅ 제스처 제어 시작")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(3)  # 초록색 페이드
                    return success
                    
            elif cmd == "정지":
                if controller and controller.running:
                    success = controller.stop_control()
                    if success:
                        logger.info("✅ 제어 정지")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(1)  # 깜빡임
                    return success
                    
            elif cmd == "모드0":
                success = self._set_robot_mode(0)
                if success:
                    logger.info("✅ 로봇 모드 0 설정")
                return success
                
            elif cmd == "모드1":
                success = self._set_robot_mode(1)
                if success:
                    logger.info("✅ 로봇 모드 1 설정")
                return success
                
            elif cmd == "스냅샷":
                success = self._take_snapshot()
                if success:
                    logger.info("✅ 스냅샷 저장")
                return success
                
            elif cmd == "밝게":
                if arduino_controller and arduino_controller.is_connected():
                    current_brightness = arduino_controller.get_status().get('brightness_level', 0)
                    new_brightness = min(5, current_brightness + 1)
                    success = arduino_controller.set_brightness(new_brightness)
                    if success:
                        logger.info(f"✅ 밝기 증가: {new_brightness}")
                    return success
                    
            elif cmd == "어둡게":
                if arduino_controller and arduino_controller.is_connected():
                    current_brightness = arduino_controller.get_status().get('brightness_level', 0)
                    new_brightness = max(0, current_brightness - 1)
                    success = arduino_controller.set_brightness(new_brightness)
                    if success:
                        logger.info(f"✅ 밝기 감소: {new_brightness}")
                    return success
                    
            elif cmd == "리셋":
                if arduino_controller and arduino_controller.is_connected():
                    success = arduino_controller.reset_arduino()
                    if success:
                        logger.info("✅ 아두이노 리셋")
                    return success
                    
            elif cmd in ["왼쪽", "오른쪽", "위", "아래"]:
                logger.info(f"✅ 방향 명령: {cmd}")
                socketio.emit('voice_direction_command', {'direction': cmd})
                if arduino_controller and arduino_controller.is_connected():
                    arduino_controller.trigger_led_effect(1)  # 깜빡임
                return True
                
            elif cmd == "종료":
                logger.info("⚠️ 시스템 종료 요청")
                socketio.emit('voice_shutdown_request', {'transcript': transcript})
                return True
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False
            
        return False
    
    def _set_robot_mode(self, mode):
        """로봇 모드 설정"""
        success = False
        
        # Arduino로 모드 설정
        if arduino_controller and arduino_controller.is_connected():
            success = arduino_controller.set_robot_mode(mode)
        
        # 로봇 팔로 모드 설정
        if robot and robot.is_arm_connected('leader'):
            position = BUTTON_POSITIONS.get(mode)
            if position:
                robot_success = robot.set_joint_position(0, position, 'leader')
                success = success or robot_success
        
        if success:
            socketio.emit('robot_mode', {'mode': mode, 'source': 'voice'})
            if arduino_controller and arduino_controller.is_connected():
                arduino_controller.trigger_led_effect(2 if mode == 0 else 3)
                
        return success
    
    def _take_snapshot(self):
        """스냅샷 저장"""
        if controller and controller.running:
            current_data = controller.get_last_data()
            if len(current_data) == 8:
                success, total = save_snapshot_to_csv(current_data)
                if success:
                    socketio.emit('snapshot_saved', {
                        'success': True,
                        'source': 'voice',
                        'total_snapshots': total,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    if arduino_controller and arduino_controller.is_connected():
                        arduino_controller.trigger_led_effect(2)  # 파란색 페이드
                    return True
        return False
    
    def get_status(self):
        """음성 컨트롤러 상태 반환"""
        return self.voice_status.copy()
    
    def is_enabled(self):
        """음성 컨트롤러 사용 가능 여부"""
        return self.enabled

# Arduino Controller (기존 코드와 동일)
class ArduinoController:
    def __init__(self, port, baudrate=ARDUINO_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.connected = False
        self.running = False
        self.status_lock = threading.Lock()
        
        self.arduino_status = {
            'connected': False,
            'brightness_level': 0,
            'robot_mode': 0,
            'last_heartbeat': 0,
            'last_status_update': 0
        }
        
        self.heartbeat_interval = 5.0
        self.last_heartbeat_sent = 0
        
    def connect(self):
        if self.connected:
            return True
            
        try:
            if not os.path.exists(self.port):
                logger.warning(f"Arduino port {self.port} not found")
                return False
                
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            
            logger.info(f"Connected to Arduino on {self.port}")
            self.connected = True
            self.running = True
            
            threading.Thread(target=self._communication_loop, daemon=True).start()
            self._send_heartbeat()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def _communication_loop(self):
        while self.running and self.connected:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat_sent >= self.heartbeat_interval:
                    self._send_heartbeat()
                
                if self.serial_port and self.serial_port.in_waiting > 0:
                    try:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        if line:
                            self._process_arduino_message(line)
                    except UnicodeDecodeError:
                        self.serial_port.reset_input_buffer()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Arduino communication error: {e}")
                self.connected = False
                break
    
    def _send_heartbeat(self):
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.write(b"HEARTBEAT\n")
                self.last_heartbeat_sent = time.time()
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    def _process_arduino_message(self, message):
        try:
            if message.startswith("CMD:ROBOT:"):
                mode = int(message.split(":")[-1])
                position = BUTTON_POSITIONS.get(mode)
                if position is not None and robot and robot.is_arm_connected('leader'):
                    robot.set_joint_position(0, position, 'leader')
                    socketio.emit('robot_mode', {'mode': mode, 'position': position, 'source': 'arduino'})
                    logger.info(f"Arduino robot command: Mode {mode}, Position {position}")
                    
                with self.status_lock:
                    self.arduino_status['robot_mode'] = mode
                    
            elif message.startswith("CMD:LED:"):
                level = int(message.split(":")[-1])
                logger.info(f"Arduino LED command: Level {level}")
                socketio.emit('arduino_led', {'brightness_level': level, 'source': 'arduino'})
                
                with self.status_lock:
                    self.arduino_status['brightness_level'] = level
                    
            elif message.startswith("STATUS:"):
                self._parse_status_message(message)
                
            elif message.startswith("ACK:HEARTBEAT"):
                with self.status_lock:
                    self.arduino_status['last_heartbeat'] = time.time()
                    self.arduino_status['connected'] = True
                    
            else:
                logger.info(f"Arduino: {message}")
                
        except Exception as e:
            logger.error(f"Error processing Arduino message '{message}': {e}")
    
    def _parse_status_message(self, message):
        try:
            parts = message.split(':')
            if len(parts) >= 7:
                brightness = int(parts[2])
                mode = int(parts[4])
                connected = int(parts[6])
                
                with self.status_lock:
                    self.arduino_status.update({
                        'brightness_level': brightness,
                        'robot_mode': mode,
                        'connected': bool(connected),
                        'last_status_update': time.time()
                    })
                
                socketio.emit('arduino_status', self.arduino_status.copy())
                
        except Exception as e:
            logger.error(f"Error parsing Arduino status: {e}")
    
    def send_command(self, command):
        try:
            if self.serial_port and self.serial_port.is_open:
                cmd = f"{command}\n"
                self.serial_port.write(cmd.encode('utf-8'))
                logger.info(f"Sent to Arduino: {command}")
                return True
        except Exception as e:
            logger.error(f"Failed to send Arduino command '{command}': {e}")
        return False
    
    def set_brightness(self, level):
        if 0 <= level <= 5:
            return self.send_command(f"SET_BRIGHTNESS:{level}")
        return False
    
    def set_robot_mode(self, mode):
        if mode in [0, 1]:
            return self.send_command(f"SET_MODE:{mode}")
        return False
    
    def trigger_led_effect(self, effect_type):
        if 0 <= effect_type <= 3:
            return self.send_command(f"LED_EFFECT:{effect_type}")
        return False
    
    def reset_arduino(self):
        return self.send_command("RESET")
    
    def get_status(self):
        with self.status_lock:
            return self.arduino_status.copy()
    
    def is_connected(self):
        return self.connected and self.arduino_status.get('connected', False)
    
    def disconnect(self):
        self.running = False
        self.connected = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        logger.info("Disconnected from Arduino")

# Simplified Robot and Controller classes for demo
class ManipulatorRobot:
    def __init__(self):
        self.connected_arms = set()
        self.is_connected = False
    
    def connect(self):
        # Simplified implementation
        return False
    
    def is_arm_connected(self, arm_type):
        return arm_type in self.connected_arms
    
    def set_joint_position(self, joint_index, position, arm_type='leader'):
        return False
    
    def disconnect(self):
        pass

class RobotController:
    def __init__(self, robot, model_path=None, arm_type='follower'):
        self.robot = robot
        self.running = False
        self.control_active = False
        
    def start(self, cam_ids=(0, 2)):
        self.running = True
        return True
    
    def get_last_data(self):
        return [None] * 8
    
    def start_control(self):
        self.control_active = True
        return True
    
    def stop_control(self):
        self.control_active = False
        return True
    
    def stop(self):
        self.running = False

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/voice/status')
def api_voice_status():
    """Get voice controller status"""
    if voice_controller:
        return jsonify({
            'success': True,
            'voice_enabled': voice_controller.is_enabled(),
            'status': voice_controller.get_status()
        })
    else:
        return jsonify({
            'success': False,
            'voice_enabled': False,
            'status': {}
        })

@app.route('/api/voice/commands')
def api_voice_commands():
    """Get available voice commands"""
    return jsonify({
        'success': True,
        'wake_words': WAKE_WORDS,
        'commands': VOICE_CMD_MAP
    })

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    
    # Send voice status
    if voice_controller:
        socketio.emit('voice_status', voice_controller.get_status(), to=request.sid)

@socketio.on('voice_start')
def handle_voice_start(data=None):
    """Start voice recognition"""
    if not voice_controller or not voice_controller.is_enabled():
        return {'success': False, 'error': 'Voice recognition not available'}
    
    try:
        success = voice_controller.start()
        return {'success': success}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('voice_stop')
def handle_voice_stop(data=None):
    """Stop voice recognition"""
    if not voice_controller:
        return {'success': False, 'error': 'Voice controller not available'}
    
    try:
        voice_controller.stop()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@socketio.on('manual_command')
def handle_manual_command(data):
    """Manual command execution for testing"""
    if not voice_controller:
        return {'success': False, 'error': 'Voice controller not available'}
    
    command = data.get('command', '')
    if command in VOICE_CMD_MAP:
        success = voice_controller._execute_command(command, f"Manual: {command}")
        return {'success': success, 'command': command}
    else:
        return {'success': False, 'error': 'Invalid command'}

# Global variables
system_initialized = False
system_status_data = {'status': 'initializing', 'message': 'System starting...'}

def init_system():
    """Initialize system with voice controller"""
    global robot, controller, arduino_controller, voice_controller, system_initialized, system_status_data
    
    try:
        logger.info("Initializing system with voice control...")
        
        # Initialize voice controller
        if VOICE_AVAILABLE:
            voice_controller = VoiceController()
            logger.info("Voice controller initialized")
        else:
            logger.warning("Voice controller not available")
        
        # Initialize other components
        arduino_controller = ArduinoController(ARDUINO_PORT)
        arduino_controller.connect()
        
        robot = ManipulatorRobot()
        controller = RobotController(robot)
        controller.start()
        
        system_initialized = True
        system_status_data = {'status': 'ready', 'message': 'System ready with voice control'}
        
        logger.info("System initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        return False

def cleanup_system():
    """Cleanup system"""
    global system_initialized
    logger.info("Cleaning up system...")
    system_initialized = False
    
    if voice_controller:
        voice_controller.stop()
    if arduino_controller:
        arduino_controller.disconnect()
    if controller:
        controller.stop()

def signal_handler(sig, frame):
    logger.info('Received shutdown signal, cleaning up...')
    cleanup_system()
    sys.exit(0)

def main():
    """Main application entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    init_csv_file()
    threading.Thread(target=init_system, daemon=True).start()
    
    try:
        logger.info("Starting Flask-SocketIO server with voice control on http://0.0.0.0:5000")
        logger.info("음성 명령어:")
        logger.info("  웨이크워드: " + ", ".join(WAKE_WORDS))
        logger.info("  명령어: " + ", ".join(VOICE_CMD_MAP.keys()))
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()