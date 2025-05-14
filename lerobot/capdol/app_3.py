#!/usr/bin/env python3
"""
로봇 암 및 손 좌표 데이터 수집 프로그램 - Flask 웹 인터페이스 버전

이 스크립트는 다음을 수행합니다:
1. leader arm과 follower arm의 캘리브레이션 데이터 로드 및 적용
2. leader arm의 토크를 해제하여 사용자가 직접 움직일 수 있게 함
3. 카메라로 손의 검지 끝 좌표 추적
4. leader arm과 follower arm의 모터 위치와 검지 끝 좌표를 동시에 기록
5. 수집된 데이터를 CSV 파일로 저장
6. 웹 브라우저를 통해 카메라 피드 확인 및 제어 가능
"""

import os
import logging
import numpy as np
import argparse
import time
import cv2
import csv
import json
import threading
import datetime
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.motors.dynamixel import (
    CalibrationMode,
    TorqueMode,
    convert_degrees_to_steps,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 캘리브레이션 디렉토리 설정
CALIBRATION_DIR = "lerobot/capdol/calibration_data"

# Flask 앱 초기화
app = Flask(__name__)

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False
        self.calibration_loaded = False
        
        # 토크 모드 가져오기 (클래스 초기화 시)
        try:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            self.TorqueMode = TorqueMode
        except ImportError:
            logger.error("Failed to import TorqueMode")
            self.TorqueMode = None

    def connect(self):
        if self.is_connected:
            return
            
        # Connect arms
        for name in self.follower_arms:
            logger.info(f"Connecting {name} follower arm")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            logger.info(f"Connecting {name} leader arm")
            self.leader_arms[name].connect()

        self.is_connected = True
        logger.info("Robot connected successfully")
        
        # 캘리브레이션 로드 및 적용
        self.load_calibration()
        
    def load_calibration(self):
        """캘리브레이션 파일을 로드하고 로봇 팔에 적용"""
        logger.info("Loading calibration data...")
        
        # 캘리브레이션 파일 경로
        leader_calib_path = os.path.join(CALIBRATION_DIR, "leader_arm.json")
        follower_calib_path = os.path.join(CALIBRATION_DIR, "follower_arm.json")
        
        # leader arm 캘리브레이션 로드
        leader_calibration = None
        if os.path.exists(leader_calib_path):
            try:
                with open(leader_calib_path, 'r') as f:
                    leader_calibration = json.load(f)
                logger.info(f"Loaded leader arm calibration from {leader_calib_path}")
            except Exception as e:
                logger.error(f"Error loading leader arm calibration: {e}")
        else:
            logger.warning(f"Leader arm calibration file not found: {leader_calib_path}")
        
        # follower arm 캘리브레이션 로드
        follower_calibration = None
        if os.path.exists(follower_calib_path):
            try:
                with open(follower_calib_path, 'r') as f:
                    follower_calibration = json.load(f)
                logger.info(f"Loaded follower arm calibration from {follower_calib_path}")
            except Exception as e:
                logger.error(f"Error loading follower arm calibration: {e}")
        else:
            logger.warning(f"Follower arm calibration file not found: {follower_calib_path}")
        
        # 캘리브레이션 데이터 적용
        if leader_calibration:
            for name in self.leader_arms:
                self.leader_arms[name].set_calibration(leader_calibration)
                logger.info(f"Applied calibration to leader arm '{name}'")
        
        if follower_calibration:
            for name in self.follower_arms:
                self.follower_arms[name].set_calibration(follower_calibration)
                logger.info(f"Applied calibration to follower arm '{name}'")
        
        self.calibration_loaded = leader_calibration is not None and follower_calibration is not None
        
        if self.calibration_loaded:
            logger.info("Calibration data loaded and applied successfully")
        else:
            logger.warning("Calibration data not fully loaded. Some robot movements may be inaccurate.")
        
    def setup_for_data_collection(self):
        """데이터 수집을 위한 설정: leader arm 토크 해제, follower arm 위치 제어 모드 설정"""
        if not self.is_connected:
            logger.error("Robot is not connected. Cannot setup for data collection.")
            return False
        
        try:
            # leader arm의 토크 해제 (사용자가 직접 움직일 수 있게)
            for name in self.leader_arms:
                logger.info(f"Disabling torque on {name} leader arm for manual movement")
                self.leader_arms[name].write("Torque_Enable", self.TorqueMode.DISABLED.value)
            
            # follower arm 설정
            for name in self.follower_arms:
                # 먼저 토크 비활성화
                self.follower_arms[name].write("Torque_Enable", self.TorqueMode.DISABLED.value)
                
                # Extended Position Control Mode 설정
                self.follower_arms[name].write("Operating_Mode", 3)
                
                # 토크 다시 활성화
                self.follower_arms[name].write("Torque_Enable", 1)
            
            logger.info("Robot setup for data collection completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up robot for data collection: {e}")
            return False

    def send_action(self, action, arm_type='follower'):
        """
        로봇 팔에 명령을 전송합니다.
        
        Args:
            action: 관절 위치를 담은 배열 또는 텐서
            arm_type: 제어할 팔 유형 ('follower' 또는 'leader')
        """
        if not self.is_connected:
            logger.error("Robot is not connected. Please connect first.")
            return False
            
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
            try:
                arms[name].write("Goal_Position", safe_goal_pos)
                return True
            except Exception as e:
                logger.error(f"Error sending action: {e}")
                return False

    def get_positions(self, arm_type):
        """
        특정 팔의 현재 관절 위치를 반환합니다.
        
        Args:
            arm_type: 읽을 팔 유형 ('follower' 또는 'leader')
        
        Returns:
            numpy.ndarray: 현재 관절 위치 값
        """
        if not self.is_connected:
            logger.error("Robot is not connected. Please connect first.")
            return None
            
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        positions = []
        
        for name in arms:
            pos = arms[name].read("Present_Position")
            positions.extend(pos)
            
        return np.array(positions)

    def copy_leader_to_follower(self):
        """leader arm의 현재 위치를 읽어 follower arm으로 복사합니다."""
        if not self.is_connected:
            logger.error("Robot is not connected")
            return False
        
        try:
            # leader arm 위치 읽기
            leader_positions = self.get_positions('leader')
            
            if leader_positions is None:
                return False
            
            # follower arm으로 위치 복사
            return self.send_action(leader_positions, arm_type='follower')
        
        except Exception as e:
            logger.error(f"Error copying leader to follower: {e}")
            return False

    def disconnect(self):
        if not self.is_connected:
            return
            
        logger.info("Disconnecting robot")
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False
        logger.info("Robot disconnected successfully")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class RobotDataCollector:
    """
    로봇 관절 위치와 손 좌표를 동시에 기록하는 데이터 수집기
    """
    def __init__(self, robot, output_dir="collected_data"):
        """
        초기화
        
        Args:
            robot: ManipulatorRobot 인스턴스
            output_dir: 데이터 저장 디렉토리
        """
        self.robot = robot
        self.output_dir = output_dir
        self.camera_id = 0  # 카메라 ID 기본값
        self.width = 640
        self.height = 480
        self.running = False
        self.recording = False
        self.tip_x = 0
        self.tip_y = 0
        self.csv_writer = None
        self.csv_file = None
        self.frame_count = 0
        self.record_count = 0
        self.current_frame = None
        
        # 손 인식 모델 초기화
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            max_num_hands=1
        )
        
        # 데이터 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 상태 정보
        self.status_info = {
            "recording": False,
            "calibrated": False,
            "samples": 0,
            "tip_coords": {"x": None, "y": None},
            "hand_detected": False
        }
    
    def start_camera(self):
        """카메라 스트림 시작"""
        logger.info(f"Opening camera with ID {self.camera_id}")
        
        # Windows나 macOS에서는 다양한 카메라 API 시도
        if os.name == 'posix':  # Linux/Mac
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        else:  # Windows 또는 기타
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        # 첫 번째 방법이 실패하면 기본 API 시도
        if not self.cap.isOpened():
            logger.warning(f"Failed to open camera with specific API, trying default")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)
        
        # 그래도 실패하면 에러 반환
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera with ID {self.camera_id}")
            return False
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 테스트 프레임 읽기
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            logger.error("Failed to read test frame from camera")
            self.cap.release()
            return False
            
        logger.info(f"Camera opened successfully: {self.width}x{self.height}")
        return True
    
    def start_recording(self):
        """데이터 기록 시작"""
        if self.recording:
            logger.warning("Recording is already in progress")
            return False
        
        try:
            # 파일명 생성 (날짜_시간.csv)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"robot_hand_data_{timestamp}.csv")
            
            # CSV 파일 열기
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 헤더 작성
            header = [
                "frame", "timestamp", 
                "hand_detected", "tip_x", "tip_y",
                "leader_joint1", "leader_joint2", "leader_joint3", "leader_joint4",
                "follower_joint1", "follower_joint2", "follower_joint3", "follower_joint4"
            ]
            self.csv_writer.writerow(header)
            
            self.recording = True
            self.record_count = 0
            self.status_info["recording"] = True
            logger.info(f"Recording started. Saving data to {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.stop_recording()
            return False
    
    def stop_recording(self):
        """데이터 기록 중지"""
        if not self.recording:
            return
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        self.recording = False
        self.status_info["recording"] = False
        logger.info(f"Recording stopped. Collected {self.record_count} samples.")
    
    def process_frame(self):
        """한 프레임 처리 및 데이터 기록"""
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return None
        
        self.frame_count += 1
        timestamp = time.time()
        hand_detected = False
        
        # OpenCV는 BGR 형식, MediaPipe는 RGB 형식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 손 감지
        results = self.hands.process(frame_rgb)
        
        # 프레임 수정 가능하게 변경
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 기본값 설정
        self.tip_x = None
        self.tip_y = None
        self.status_info["hand_detected"] = False
        self.status_info["tip_coords"] = {"x": None, "y": None}
        
        # 손이 감지된 경우
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness if results.multi_handedness else [None] * len(results.multi_hand_landmarks)
            ):
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 검지 끝(8번 랜드마크) 좌표 추출
                index_finger_tip = hand_landmarks.landmark[8]
                self.tip_x = int(index_finger_tip.x * self.width)
                self.tip_y = int(index_finger_tip.y * self.height)
                hand_detected = True
                self.status_info["hand_detected"] = True
                self.status_info["tip_coords"] = {"x": self.tip_x, "y": self.tip_y}
                
                # 손 타입 표시 (있는 경우)
                if handedness:
                    hand_type = handedness.classification[0].label
                    cv2.putText(
                        frame,
                        f"Hand: {hand_type}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # 검지 끝 위치 표시
                cv2.putText(
                    frame,
                    f"Tip: x={self.tip_x}, y={self.tip_y}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 검지 끝 위치에 원 그리기
                cv2.circle(
                    frame,
                    (self.tip_x, self.tip_y),
                    10,
                    (0, 0, 255),
                    -1
                )
        
        # leader arm 위치 읽기
        leader_positions = self.robot.get_positions('leader')
        leader_step_positions = convert_degrees_to_steps(leader_positions)
        
        # follower arm 위치로 복사 (및 읽기)
        self.robot.copy_leader_to_follower()
        follower_positions = self.robot.get_positions('follower')
        follower_step_positions = convert_degrees_to_steps(follower_positions)
        
        # 데이터 기록 (recording 모드인 경우)
        if self.recording and self.csv_writer:
            row = [self.frame_count, timestamp, hand_detected]
            
            # 손 좌표 추가
            if hand_detected:
                row.extend([self.tip_x, self.tip_y])
            else:
                row.extend([None, None])
            
            # leader arm 위치 추가
            if leader_step_positions is not None:
                row.extend(leader_step_positions)
            else:
                row.extend([None] * 4)  # 4개 관절 위치
            
            # follower arm 위치 추가
            if follower_step_positions is not None:
                row.extend(follower_step_positions)
            else:
                row.extend([None] * 4)  # 4개 관절 위치
            
            # CSV에 기록
            self.csv_writer.writerow(row)
            self.record_count += 1
            self.status_info["samples"] = self.record_count
        
        # 캘리브레이션 상태 표시
        calibration_status = "CALIBRATED" if self.robot.calibration_loaded else "NOT CALIBRATED"
        self.status_info["calibrated"] = self.robot.calibration_loaded
        cv2.putText(
            frame,
            calibration_status,
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.robot.calibration_loaded else (0, 0, 255),
            2
        )
        
        # 녹화 상태 표시
        recording_status = "RECORDING" if self.recording else "NOT RECORDING"
        cv2.putText(
            frame,
            recording_status,
            (self.width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if self.recording else (200, 200, 200),
            2
        )
        
        # 샘플 수 표시
        if self.recording:
            cv2.putText(
                frame,
                f"Samples: {self.record_count}",
                (self.width - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        self.current_frame = frame
        return frame
    
    def generate_frames(self):
        """비디오 스트림 생성을 위한 제너레이터 함수"""
        while self.running:
            if self.current_frame is not None:
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if not ret:
                    continue
                
                # HTTP 응답 메시지 구성
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # 약 30fps 제한
    
    def start(self):
        """데이터 수집 시작"""
        # 로봇 설정
        if not self.robot.setup_for_data_collection():
            logger.error("Failed to setup robot for data collection")
            return False
        
        # 카메라 시작
        if not self.start_camera():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _processing_loop(self):
        """프레임 처리 루프 (별도 스레드)"""
        logger.info("Starting frame processing loop")
        
        try:
            while self.running:
                # 프레임 처리
                frame = self.process_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # 약간의 대기 시간 (CPU 사용량 감소)
                time.sleep(0.03)  # 약 30fps 제한
        
        except Exception as e:
            logger.error(f"Error in processing loop: {e}", exc_info=True)
        finally:
            logger.info("Processing loop stopped")
    
    def take_snapshot(self):
        """단일 샘플 기록 (스냅샷)"""
        if self.recording:
            logger.warning("Cannot take snapshot while recording is active")
            return False
        
        # 일시적으로 녹화 시작 및 즉시 중지
        if self.start_recording():
            self.stop_recording()
            return True
        return False
    
    def stop(self):
        """모든 처리 중지 및 리소스 해제"""
        self.running = False
        
        # 녹화 중이면 중지
        self.stop_recording()
        
        # 스레드 종료 대기
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # 리소스 해제
        if hasattr(self, 'hands'):
            self.hands.close()
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        logger.info("Data collector stopped")


# 전역 객체
robot = None
collector = None

# Flask 라우트 설정
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트림 라우트"""
    if collector is None or not collector.running:
        # 더미 프레임 반환
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera not started", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', dummy_frame)
        frame_bytes = buffer.tobytes()
        
        def generate_dummy():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        return Response(generate_dummy(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(collector.generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """녹화 시작 API"""
    if collector is None or not collector.running:
        return jsonify({"success": False, "message": "Collector not running"}), 400
    
    success = collector.start_recording()
    return jsonify({"success": success})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """녹화 중지 API"""
    if collector is None or not collector.running:
        return jsonify({"success": False, "message": "Collector not running"}), 400
    
    collector.stop_recording()
    return jsonify({"success": True})

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    """스냅샷 촬영 API"""
    if collector is None or not collector.running:
        return jsonify({"success": False, "message": "Collector not running"}), 400
    
    success = collector.take_snapshot()
    return jsonify({"success": success})

@app.route('/status')
def get_status():
    """현재 상태 정보 API"""
    if collector is None:
        return jsonify({"running": False})
    
    return jsonify({
        "running": collector.running,
        **collector.status_info
    })

def create_app(output_dir='lerobot/capdol/collected_data', camera_id=0):
    """Flask 앱 및 로봇 초기화"""
    global robot, collector
    
    try:
        # 로봇 초기화
        logger.info("Initializing robot...")
        robot_config = KochRobotConfig()
        robot = ManipulatorRobot(robot_config)
        
        # 로봇 연결
        logger.info("Connecting to robot...")
        robot.connect()
        
        # 데이터 수집기 초기화 및 시작
        logger.info("Starting data collector...")
        collector = RobotDataCollector(robot, output_dir)
        collector.camera_id = camera_id
        collector.start()
        
        # 사용 설명
        print("\n=== Robot Data Collection Web Interface ===")
        print(f"- Using calibration files from: {CALIBRATION_DIR}")
        print(f"- Saving data to: {output_dir}")
        print(f"- Using camera ID: {camera_id}")
        print("\nOpen your browser at http://localhost:5000 to view and control the data collection.")
        print("Move the leader arm manually to collect data.")
        print("The follower arm will automatically follow.")
        print("Make sure to position your hand so the camera can see it.\n")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        if 'robot' in globals() and robot and robot.is_connected:
            robot.disconnect()
        
    return app

def shutdown_app():
    """앱 종료 시 리소스 정리"""
    global robot, collector
    
    if collector:
        collector.stop()
    
    if robot and robot.is_connected:
        robot.disconnect()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Robot and Hand Data Collector Web Interface')
    parser.add_argument('--output', type=str, default='lerobot/capdol/collected_data',
                        help='Directory to save collected data (default: lerobot/capdol/collected_data)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID to use (default: 1)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on (default: 0.0.0.0, accessible from any network interface)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    
    try:
        # 앱 생성 및 실행
        app_instance = create_app(args.output, args.camera)
        app_instance.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nStopping by user request...")
    finally:
        # 리소스 정리
        shutdown_app()

if __name__ == "__main__":
    main()