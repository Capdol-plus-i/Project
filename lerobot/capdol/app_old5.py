#!/usr/bin/env python3
import os
import json
import time
import threading
import logging
import numpy as np
import cv2
import csv
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = "lerobot/capdol"
CALIBRATION_DIR = f"{BASE_DIR}/calibration_data"
DATA_DIR = f"{BASE_DIR}/collected_data"
MODEL_PATH = f"{BASE_DIR}/models/model_parameters_resnet.npz"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "robot_hand_data.csv")
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
               "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]

# Flask 및 MediaPipe 설정
app = Flask(__name__)
mp_hands = mp.solutions.hands

class ManipulatorRobot:
    def __init__(self, config):
        self.arms = {'follower': make_motors_buses_from_configs(config.follower_arms)}
        self.is_connected = False
        self.calibrated = False
        try:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            self.TorqueMode = TorqueMode
        except ImportError:
            logger.warning("TorqueMode import 실패")
            self.TorqueMode = None

    def connect(self):
        if self.is_connected: return True
        try:
            for buses in self.arms.values():
                for name, bus in buses.items():
                    logger.info(f"로봇 암 '{name}' 연결")
                    bus.connect()
            self.is_connected = True
            self._load_calibration()
            return True
        except Exception as e:
            logger.error(f"연결 오류: {e}")
            return False

    def _load_calibration(self):
        path = os.path.join(CALIBRATION_DIR, 'follower_arm.json')
        if not os.path.exists(path):
            logger.warning("캘리브레이션 파일 없음")
            return
            
        try:
            with open(path) as f:
                calib_data = json.load(f)
            for bus in self.arms['follower'].values():
                bus.set_calibration(calib_data)
            self.calibrated = True
            logger.info("캘리브레이션 로드 완료")
        except Exception as e:
            logger.error(f"캘리브레이션 오류: {e}")

    def disable_torque(self, arm_type='follower'):
        if not self.is_connected: return False
        try:
            for bus in self.arms[arm_type].values():
                if self.TorqueMode:
                    bus.write('Torque_Enable', self.TorqueMode.DISABLED.value)
            return True
        except Exception as e:
            logger.error(f"토크 비활성화 오류: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if not self.is_connected: return False
        try:
            for bus in self.arms[arm_type].values():
                if self.TorqueMode:
                    bus.write('Torque_Enable', self.TorqueMode.DISABLED.value)
                bus.write('Operating_Mode', 3)
                bus.write('Torque_Enable', 1)
            return True
        except Exception as e:
            logger.error(f"제어 설정 오류: {e}")
            return False

    def move(self, positions, arm_type='follower'):
        if not self.is_connected or len(positions) != 4: return False
        try:
            idx = 0
            for bus in self.arms[arm_type].values():
                count = len(bus.motor_names)
                bus.write('Goal_Position', positions[idx:idx+count])
                idx += count
            return True
        except Exception as e:
            logger.error(f"이동 오류: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if not self.is_connected: return [None] * 4
        try:
            positions = []
            for bus in self.arms[arm_type].values():
                positions.extend(bus.read('Present_Position'))
            return positions[:4]
        except Exception as e:
            logger.error(f"위치 읽기 오류: {e}")
            return [None] * 4

    def disconnect(self):
        if not self.is_connected: return
        for buses in self.arms.values():
            for bus in buses.values():
                try: bus.disconnect()
                except: pass
        self.is_connected = False
        logger.info("로봇 연결 해제")

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
        
        # 모델 로드
        self.params = None
        if model_path:
            try:
                params = np.load(model_path)
                self.params = {k: params[k] for k in params.files}
                logger.info("모델 로드 완료")
            except Exception as e:
                logger.error(f"모델 로드 오류: {e}")
        
        # MediaPipe 초기화
        self.hands = [mp_hands.Hands(
            model_complexity=0, 
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3, 
            max_num_hands=1,
            static_image_mode=False
        ) for _ in range(2)]

    def start(self, cam_ids=(0, 2)):
        if self.running: return True
        if not self._open_cams(cam_ids): return False
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        logger.info("컨트롤러 시작")
        return True

    def start_control(self):
        if not self.running or self.control_active: return False
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            logger.info("로봇 제어 활성화")
        return success

    def stop_control(self):
        if self.control_active:
            self.control_active = False
            self.robot.disable_torque(self.arm_type)
            logger.info("로봇 제어 비활성화")
        return True

    def _open_cams(self, cam_ids):
        for i, cid in enumerate(cam_ids):
            try:
                cap = cv2.VideoCapture(cid)
                if not cap.isOpened():
                    logger.error(f"카메라 {i+1} 열기 실패")
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cams[i] = cap
                logger.info(f"카메라 {i+1} 준비")
            except Exception as e:
                logger.error(f"카메라 오류: {e}")
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
            return self._create_dummy_frame(f"카메라 {idx+1} 준비 안됨")
            
        # 최신 프레임 가져오기
        frame = None
        for _ in range(4):  # 버퍼 플러시
            ret, f = self.cams[idx].read()
            if ret: frame = f
                
        if frame is None:
            return self._create_dummy_frame(f"프레임 없음")

        # 프레임 처리
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands[idx].process(rgb)
            
            self.hand_detected[idx] = False
            
            if results.multi_hand_landmarks:
                landmark = results.multi_hand_landmarks[0].landmark[8]  # 검지 손가락 끝
                x, y = int(landmark.x * w), int(landmark.y * h)
                self.tip[idx] = (x, y)
                self.hand_detected[idx] = True
                
                if idx == 1: # Z 좌표 (두 번째 카메라에서)
                    self.z = y
                    
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        except Exception as e:
            logger.error(f"프레임 처리 오류: {e}")
            
        return frame

    def _create_dummy_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, message, (70, 240), 1, 2, (255, 255, 255), 2)
        return frame

    def _process_loop(self):
        while self.running:
            try:
                # 프레임 처리 및 로봇 제어
                frames = [self._process_frame(i) for i in range(2)]
                positions = self.robot.get_positions(self.arm_type)
                
                if self.control_active and self.hand_detected[0]:
                    joints = self._predict(*self.tip[0], self.z).round().astype(int)
                    if np.sum(np.abs(joints)) > 0:
                        self.robot.move(joints, self.arm_type)
                
                # 데이터 업데이트
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
                logger.error(f"처리 오류: {e}")
                time.sleep(0.1)
        self._cleanup()

    def _cleanup(self):
        for hand in self.hands:
            try: hand.close()
            except: pass
        for i, cam in enumerate(self.cams):
            if cam:
                try: 
                    cam.release()
                    logger.info(f"카메라 {i+1} 해제")
                except: pass
        self.cams = [None, None]

    def get_last_frame(self, idx):
        with self.data_lock:
            frame = self.last_frames[idx]
            return frame.copy() if frame is not None else self._create_dummy_frame("프레임 없음")

    def get_last_data(self):
        with self.data_lock:
            return self.last_data.copy()

    def save_snapshot(self):
        data = self.get_last_data()
        try:
            # 필요시 CSV 헤더 생성
            if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
                with open(CSV_PATH, 'w', newline='') as f:
                    csv.writer(f).writerow(CSV_HEADERS)
            
            # 데이터 저장
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow(data)
            logger.info("스냅샷 저장 완료")
            return True
        except Exception as e:
            logger.error(f"스냅샷 오류: {e}")
            return False

    def stop(self):
        if not self.running: return
        self.running = False
        self.control_active = False

# 글로벌 인스턴스
robot = None
controller = None

def generate_frames(cam_idx):
    while True:
        try:
            if controller and controller.running:
                frame = controller.get_last_frame(cam_idx)
            else:
                # 시스템 초기화 메시지
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "시스템 시작 중...", (70, 240), 1, 2, (255, 255, 255), 2)
                
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            if not controller or not controller.running:
                time.sleep(0.5)
        except Exception:
            time.sleep(0.5)

# Flask 라우트
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "시스템 준비 안됨"})
    return jsonify({"success": controller.save_snapshot()})

@app.route('/status')
def status():
    if not controller or not controller.running:
        return jsonify({"error": "시스템 준비 안됨"})
        
    # 데이터 변환
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
        return jsonify({"success": False, "error": "시스템 준비 안됨"})
    return jsonify({"success": controller.start_control()})

@app.route('/control/stop', methods=['POST'])
def stop_control():
    if not controller or not controller.running:
        return jsonify({"success": False, "error": "시스템 준비 안됨"})
    return jsonify({"success": controller.stop_control()})

def init_system(model_path=MODEL_PATH, cam_ids=(0, 2), arm_type='follower'):
    global robot, controller
    try:
        # 로봇 및 컨트롤러 초기화
        robot = ManipulatorRobot(KochRobotConfig())
        if not robot.connect():
            logger.error("로봇 연결 실패")
            return False
        
        robot.disable_torque(arm_type)
        
        controller = RobotController(robot, model_path, arm_type)
        if not controller.start(cam_ids):
            logger.error("컨트롤러 시작 실패")
            robot.disconnect()
            return False
            
        logger.info("=== 시스템 준비 완료 ===")
        return True
    except Exception as e:
        logger.error(f"초기화 오류: {e}")
        return False

def cleanup_system():
    if controller:
        try: controller.stop()
        except: pass
    if robot:
        try: robot.disconnect()
        except: pass
    logger.info("시스템 종료")

def main():
    # 백그라운드에서 시스템 초기화
    threading.Thread(target=init_system, daemon=True).start()
    
    # Flask 웹 서버 시작
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("사용자에 의한 중단")
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()