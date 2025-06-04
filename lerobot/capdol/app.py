#!/usr/bin/env python3
import os, time, threading, logging, numpy as np, cv2, serial, signal, sys, csv, re
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
import pyaudio
from six.moves import queue
from google.cloud import speech

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = f"models/model_parameters_resnet.npz"
BUTTON_POSITIONS = {0: 2124, 1: 3148}
PORT_ACM0, PORT_ACM1 = "/dev/ttyACM0", "/dev/ttyACM1"
ARDUINO_PORT = "/dev/ttyACM2"  # 아두이노 포트
BAUDRATE = 1000000
ARDUINO_BAUDRATE = 9600
DYNAMIXEL_IDS_ACM0, DYNAMIXEL_IDS_ACM1 = [1, 2, 3, 4], [1]
CAM_IDS = [0, 2]
ATTEMPTS_READ_FRAME_COUNT = 1
DEFAULT_JOINTS = [3000, 2470, 1530, 1530]
HANDS_TIMEOUT = 3.0  # seconds
POSITION_P_GAIN = [200, 200, 400, 200]  # P-gain for each joint

VOICE_RATE = 16000
VOICE_CHUNK = int(VOICE_RATE / 5)  # 100ms chunks
WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "로봇아"]
VOICE_CMD_MAP = {
    "시작": ["시작", "시작해", "제어 시작", "컨트롤 시작"],
    "정지": ["정지", "멈춰", "스톱", "중지"],
    "모드0": ["모드 0", "모드0", "모드 영", "모드 제로"],
    "모드1": ["모드 1", "모드1", "모드 일", "모드 원"],
    #"스냅샷": ["스냅샷", "사진", "캡처", "저장"],
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

# CSV Management Functions (unchanged)
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
        self.running = False
        self.listening = False
        self.current_stream = None
        self.client = None
        self.enabled = True  # 기본적으로 활성화
        # 음성 상태 추적
        self.voice_status = {
            'enabled': self.enabled,
            'listening': False,
            'last_transcript': '',
            'last_command': '',
            'wake_word_detected': False
        }

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
                    
            elif cmd == "종료":
                if controller and controller.running:
                    success = controller.stop_control()
                    if success:
                        logger.info("✅ 제어 종료")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(1)  # 깜빡임
                    return success
            
            elif cmd == "정지":
                if controller and controller.running:
                    success = controller.pause_control()
                    if success:
                        logger.info("✅ 제스처 제어 정지")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(1)

            # elif cmd == "모드0":
            #     success = self._set_robot_mode(0)
            #     if success:
            #         logger.info("✅ 로봇 모드 0 설정")
            #     return success
                
            # elif cmd == "모드1":
            #     success = self._set_robot_mode(1)
            #     if success:
            #         logger.info("✅ 로봇 모드 1 설정")
            #     return success
                
            # elif cmd == "스냅샷":
            #     success = self._take_snapshot()
            #     if success:
            #         logger.info("✅ 스냅샷 저장")
            #     return success
                
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
                
            # elif cmd == "종료":
            #     logger.info("⚠️ 시스템 종료 요청")
            #     socketio.emit('voice_shutdown_request', {'transcript': transcript})
            #     return True
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False
            
        return False
    
    # def _set_robot_mode(self, mode):
    #     """로봇 모드 설정"""
    #     success = False
        
    #     # Arduino로 모드 설정
    #     if arduino_controller and arduino_controller.is_connected():
    #         success = arduino_controller.set_robot_mode(mode)
        
    #     # 로봇 팔로 모드 설정
    #     if robot and robot.is_arm_connected('leader'):
    #         position = BUTTON_POSITIONS.get(mode)
    #         if position:
    #             robot_success = robot.set_joint_position(0, position, 'leader')
    #             success = success or robot_success
        
    #     if success:
    #         socketio.emit('robot_mode', {'mode': mode, 'source': 'voice'})
    #         if arduino_controller and arduino_controller.is_connected():
    #             arduino_controller.trigger_led_effect(2 if mode == 0 else 3)
                
    #     return success
    
    # def _take_snapshot(self):
    #     """스냅샷 저장"""
    #     if controller and controller.running:
    #         current_data = controller.get_last_data()
    #         if len(current_data) == 8:
    #             success, total = save_snapshot_to_csv(current_data)
    #             if success:
    #                 socketio.emit('snapshot_saved', {
    #                     'success': True,
    #                     'source': 'voice',
    #                     'total_snapshots': total,
    #                     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                 })
    #                 if arduino_controller and arduino_controller.is_connected():
    #                     arduino_controller.trigger_led_effect(2)  # 파란색 페이드
    #                 return True
    #     return False
    
    def get_status(self):
        """음성 컨트롤러 상태 반환"""
        return self.voice_status.copy()
    
    def is_enabled(self):
        """음성 컨트롤러 사용 가능 여부"""
        return self.enabled

class ArduinoController:
    def __init__(self, port, baudrate=ARDUINO_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.connected = False
        self.running = False
        self.status_lock = threading.Lock()
        
        # Arduino status
        self.arduino_status = {
            'connected': False,
            'brightness_level': 0,
            'robot_mode': 0,
            'last_heartbeat': 0,
            'last_status_update': 0
        }
        
        # Heartbeat settings
        self.heartbeat_interval = 5.0  # seconds
        self.last_heartbeat_sent = 0
        
    def connect(self):
        """Connect to Arduino"""
        if self.connected:
            return True
            
        try:
            if not os.path.exists(self.port):
                logger.warning(f"Arduino port {self.port} not found")
                return False
                
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino reset delay
            
            logger.info(f"Connected to Arduino on {self.port}")
            self.connected = True
            self.running = True
            
            # Start communication thread
            threading.Thread(target=self._communication_loop, daemon=True).start()
            
            # Send initial heartbeat
            self._send_heartbeat()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def _communication_loop(self):
        """Arduino communication loop"""
        while self.running and self.connected:
            try:
                # Send periodic heartbeat
                current_time = time.time()
                if current_time - self.last_heartbeat_sent >= self.heartbeat_interval:
                    self._send_heartbeat()
                
                # Read incoming data
                if self.serial_port and self.serial_port.in_waiting > 0:
                    try:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        if line:
                            self._process_arduino_message(line)
                    except UnicodeDecodeError:
                        # Clear buffer on decode error
                        self.serial_port.reset_input_buffer()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Arduino communication error: {e}")
                self.connected = False
                break
    
    def _send_heartbeat(self):
        """Send heartbeat to Arduino"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.write(b"HEARTBEAT\n")
                self.last_heartbeat_sent = time.time()
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    def _process_arduino_message(self, message):
        """Process messages from Arduino"""
        try:
            # Handle robot commands (existing functionality)
            if message.startswith("CMD:ROBOT:"):
                mode = int(message.split(":")[-1])
                position = BUTTON_POSITIONS.get(mode)
                if position is not None and robot and robot.is_arm_connected('leader'):
                    robot.set_joint_position(0, position, 'leader')
                    socketio.emit('robot_mode', {'mode': mode, 'position': position})
                    logger.info(f"Arduino robot command: Mode {mode}, Position {position}")
                    
                    # Update Arduino status
                    with self.status_lock:
                        self.arduino_status['robot_mode'] = mode
                        
            # Handle LED commands
            elif message.startswith("CMD:LED:"):
                level = int(message.split(":")[-1])
                logger.info(f"Arduino LED command: Level {level}")
                socketio.emit('arduino_led', {'brightness_level': level})
                
                # Update Arduino status
                with self.status_lock:
                    self.arduino_status['brightness_level'] = level
                    
            # Handle status updates
            elif message.startswith("STATUS:"):
                self._parse_status_message(message)
                
            # Handle heartbeat acknowledgment
            elif message.startswith("ACK:HEARTBEAT"):
                with self.status_lock:
                    self.arduino_status['last_heartbeat'] = time.time()
                    self.arduino_status['connected'] = True
                    
            # Log other messages
            else:
                logger.info(f"Arduino: {message}")
                
        except Exception as e:
            logger.error(f"Error processing Arduino message '{message}': {e}")
    
    def _parse_status_message(self, message):
        """Parse Arduino status message: STATUS:BRIGHTNESS:X:MODE:X:CONNECTED:X"""
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
                
                # Emit status to web clients
                socketio.emit('arduino_status', self.arduino_status.copy())
                
        except Exception as e:
            logger.error(f"Error parsing Arduino status: {e}")
    
    def send_command(self, command):
        """Send command to Arduino"""
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
        """Set Arduino LED brightness level (0-5)"""
        if 0 <= level <= 5:
            return self.send_command(f"SET_BRIGHTNESS:{level}")
        return False
    
    def set_robot_mode(self, mode):
        """Set Arduino robot mode (0 or 1)"""
        if mode in [0, 1]:
            return self.send_command(f"SET_MODE:{mode}")
        return False
    
    def trigger_led_effect(self, effect_type):
        """Trigger LED effect (0: none, 1: flash, 2: blue fade, 3: green fade)"""
        if 0 <= effect_type <= 3:
            return self.send_command(f"LED_EFFECT:{effect_type}")
        return False
    
    def reset_arduino(self):
        """Reset Arduino to default state"""
        return self.send_command("RESET")
    
    def get_status(self):
        """Get current Arduino status"""
        with self.status_lock:
            return self.arduino_status.copy()
    
    def is_connected(self):
        """Check if Arduino is connected"""
        return self.connected and self.arduino_status.get('connected', False)
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.reset_arduino()
        self.running = False
        self.connected = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        logger.info("Disconnected from Arduino")

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
            
            for i, dxl_id in enumerate(self.motor_ids[arm_type]):
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
                                             self.ADDR_POSITION_P_GAIN, POSITION_P_GAIN[i])
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
                    position = max(1, min(4094, position))
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
            position = max(1, min(4094, int(position)))
            
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
        self.last_status_update = 0
        self.status_update_interval = 0.1
        #self.last_hand_detected_time = 0
        
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5, 
            max_num_hands=2,  # Changed from 2 to 1 for better performance
            static_image_mode=False
        ) for _ in range(2)]
        
        # Warm up models
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            try:
                hand.process(dummy)
            except Exception as e:
                logger.warning(f"MediaPipe warmup warning: {e}")

    def start(self):
        if self.running: return True
        
        logger.info(f"Starting controller with cameras {CAM_IDS}")
        
        if not self._open_cams(): 
            logger.error("Failed to open cameras")
            return False
        
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        
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

    def pause_control(self):
        """Pause control without stopping it completely"""
        if self.control_active:
            logger.info("Pausing gesture control...")
            self.control_active = False
            logger.info("Gesture control paused")
        return True

    def _open_cams(self):
        """Open cameras with better error handling"""
        for i, cid in enumerate(CAM_IDS):
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
            x_norm = max(0, min(1, x / 650.0))
            y_norm = max(0, min(1, y / 650.0))
            z_norm = max(0, min(1, z / 650.0))
            
            A = np.array([[x_norm], [y_norm], [z_norm]])
            L = len(self.params) // 2
            
            for l in range(1, L):
                A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
            
            output = ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()
            
            # Safety clamp outputs to reasonable joint limits
            output = np.clip(output, 1, 4094)
            
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
        for attempt in range(ATTEMPTS_READ_FRAME_COUNT):
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
                        self.z = max(0, min(y, self.height))  # Clamp Z value
                    
                    # Draw detection
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    # cv2.putText(frame, f"Hand", (x-30, y-20), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                               
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
                
                # Emit status updates at controlled intervals
                current_time = time.time()
                if current_time - self.last_status_update >= self.status_update_interval:
                    self._emit_status_update()
                    self.last_status_update = current_time

                if self.control_active and (self.hand_detected[0] and self.hand_detected[1]) and self.robot.is_arm_connected(self.arm_type):
                    joints = self._predict(*self.tip[0], self.z)
                    if np.sum(np.abs(joints)) > 0:
                        self.robot.move(joints, self.arm_type)

                elif self.control_active and (not self.hand_detected[0] or not self.hand_detected[1]) and self.robot.is_arm_connected(self.arm_type):
                    
                    if not hasattr(self, 'last_hand_detected_time'):
                        self.last_hand_detected_time = current_time

                    if current_time - self.last_hand_detected_time >= HANDS_TIMEOUT:
                        self.robot.move(DEFAULT_JOINTS, self.arm_type)
                        self.last_hand_detected_time = current_time

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

# Flask routes (unchanged)
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
        arduino_status = arduino_controller.get_status() if arduino_controller else {}
        return jsonify({
            'system_ready': True,
            'control_active': controller.control_active,
            'robot_connected': robot.is_connected,
            'connected_arms': list(robot.connected_arms),
            'hand_detected': controller.hand_detected,
            'data': controller.get_last_data(),
            'arduino': arduino_status
        })
    else:
        return jsonify({
            'system_ready': False,
            'control_active': False,
            'robot_connected': robot.is_connected if robot else False,
            'connected_arms': list(robot.connected_arms) if robot else [],
            'hand_detected': [False, False],
            'data': [None] * 8,
            'arduino': {}
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
    
    # Send Arduino status
    if arduino_controller:
        socketio.emit('arduino_status', arduino_controller.get_status(), to=request.sid)

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
        # Trigger Arduino LED effect
        if arduino_controller and arduino_controller.is_connected():
            arduino_controller.trigger_led_effect(1)  # Flash effect
    return {'success': success}

@socketio.on('stop_control')
def handle_stop_control(data=None):
    logger.info("Received stop_control command")
    if not system_initialized or not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    success = controller.pause_control()
    if success:
        socketio.emit('control_status', {'active': False})
        # Trigger Arduino LED effect
        if arduino_controller and arduino_controller.is_connected():
            arduino_controller.trigger_led_effect(0)  # No effect (normal)
    return {'success': success}

@socketio.on('set_robot_mode')
def handle_set_robot_mode(data):
    logger.info(f"Received set_robot_mode command: {data}")
    if not system_initialized or not controller or not controller.running:
        return {'success': False, 'error': 'System not ready'}
    
    mode = data.get('mode', 0)
    position = BUTTON_POSITIONS.get(mode)
    
    if position is None:
        return {'success': False, 'error': f'Invalid mode: {mode}'}
    
    # Try Arduino first, then robot
    arduino_success = False
    robot_success = False
    
    if arduino_controller and arduino_controller.is_connected():
        arduino_success = arduino_controller.set_robot_mode(mode)
    
    if robot.is_arm_connected('leader'):
        robot_success = robot.set_joint_position(0, position, 'leader')
    
    if arduino_success or robot_success:
        socketio.emit('robot_mode', {'mode': mode, 'position': position})
        return {'success': True, 'mode': mode, 'position': position}
    else:
        return {'success': False, 'error': 'No control interfaces available'}

# New Arduino WebSocket handlers
@socketio.on('arduino_set_brightness')
def handle_arduino_set_brightness(data):
    """Set Arduino LED brightness"""
    if not arduino_controller or not arduino_controller.is_connected():
        return {'success': False, 'error': 'Arduino not connected'}
    
    level = data.get('level', 0)
    if not isinstance(level, int) or not (0 <= level <= 5):
        return {'success': False, 'error': 'Invalid brightness level (0-5)'}
    
    success = arduino_controller.set_brightness(level)
    return {'success': success}

@socketio.on('arduino_led_effect')
def handle_arduino_led_effect(data):
    """Trigger Arduino LED effect"""
    if not arduino_controller or not arduino_controller.is_connected():
        return {'success': False, 'error': 'Arduino not connected'}
    
    effect = data.get('effect', 0)
    if not isinstance(effect, int) or not (0 <= effect <= 3):
        return {'success': False, 'error': 'Invalid effect type (0-3)'}
    
    success = arduino_controller.trigger_led_effect(effect)
    return {'success': success}

@socketio.on('arduino_reset')
def handle_arduino_reset(data=None):
    """Reset Arduino to default state"""
    if not arduino_controller or not arduino_controller.is_connected():
        return {'success': False, 'error': 'Arduino not connected'}
    
    success = arduino_controller.reset_arduino()
    return {'success': success}

@socketio.on('get_status')
def handle_get_status(data=None):
    """Get current system status via WebSocket"""
    arduino_status = arduino_controller.get_status() if arduino_controller else {}
    
    if system_initialized and controller and controller.running:
        return {
            'system_ready': True,
            'control_active': controller.control_active,
            'robot_connected': robot.is_connected,
            'connected_arms': list(robot.connected_arms),
            'hand_detected': controller.hand_detected,
            'last_data': controller.get_last_data(),
            'arduino': arduino_status
        }
    else:
        return {
            'system_ready': False,
            'control_active': False,
            'robot_connected': robot.is_connected if robot else False,
            'connected_arms': list(robot.connected_arms) if robot else [],
            'hand_detected': [False, False],
            'last_data': [None] * 8,
            'arduino': arduino_status
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
            # Trigger Arduino LED effect for snapshot feedback
            if arduino_controller and arduino_controller.is_connected():
                arduino_controller.trigger_led_effect(2)  # Blue fade effect
            
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
def init_system(model_path=MODEL_PATH):
    """Enhanced system initialization with Arduino support"""
    global robot, controller, arduino_controller, voice_controller, system_initialized, system_status_data, robot_status_data
    try:
        logger.info("Initializing robot system...")
        system_status_data = {'status': 'initializing', 'message': 'Connecting to Arduino...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize Arduino controller
        arduino_controller = ArduinoController(ARDUINO_PORT)
        if arduino_controller.connect():
            logger.info("Arduino controller connected")
        else:
            logger.warning("Arduino controller not connected - continuing without it")
        
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

        voice_controller = VoiceController()

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
        if not controller.start():
            robot.disconnect()
            if arduino_controller:
                arduino_controller.disconnect()
            error_msg = 'Controller start failed - check cameras'
            logger.error(error_msg)
            system_status_data = {'status': 'error', 'message': error_msg}
            socketio.emit('system_status', system_status_data)
            return False
        
        logger.info("System initialization complete!")
        status_message = f'System ready - Robot: {connected_arms}, Arduino: {"Connected" if arduino_controller.is_connected() else "Disconnected"}'
        system_status_data = {'status': 'ready', 'message': status_message}
        system_initialized = True
        
        # Broadcast updated status to ALL connected clients
        socketio.emit('system_status', system_status_data)
        socketio.emit('robot_status', robot_status_data)
        if arduino_controller:
            socketio.emit('arduino_status', arduino_controller.get_status())
        
        # Arduino startup effect
        if arduino_controller and arduino_controller.is_connected():
            arduino_controller.trigger_led_effect(3)  # Green fade for successful init
        
        if voice_controller:
            voice_controller.start()

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
    if arduino_controller:
        try:
            arduino_controller.disconnect()
        except Exception as e:
            logger.error(f"Arduino cleanup error: {e}")
    if voice_controller:
        try:
            voice_controller.stop()
        except Exception as e:
            logger.error(f"Voice controller cleanup error: {e}")
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