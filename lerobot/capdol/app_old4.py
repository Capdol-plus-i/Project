#!/usr/bin/env python3
import os
import numpy as np
import csv
import threading
import time
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

# 기본 설정
DATA_DIR = "lerobot/capdol/collected_data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "robot_hand_data.csv")
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
               "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]

app = Flask(__name__)
mp_hands = mp.solutions.hands

class ManipulatorRobot:
    def __init__(self, config):
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.connected = False
    
    def connect(self):
        for arm in self.follower_arms.values():
            arm.connect()
        self.connected = True
    
    def disable_follower_torque(self):
        for arm in self.follower_arms.values():
            arm.write("Torque_Enable", TorqueMode.DISABLED.value)
    
    def get_follower_positions(self):
        positions = []
        for arm in self.follower_arms.values():
            positions.extend(arm.read("Present_Position"))
        return positions[:4]  # 앞 4개만 반환
    
    def disconnect(self):
        for arm in self.follower_arms.values():
            arm.disconnect()

class DualCameraCollector:
    def __init__(self, robot, cam1=0, cam2=2):
        self.robot = robot
        self.cam_ids = [cam1, cam2]
        self.caps = [None, None]
        self.mp_hands = [mp_hands.Hands(min_detection_confidence=0.4, max_num_hands=1) for _ in range(2)]
        self.data_lock = threading.Lock()
        self.last_frames = [None, None]
        self.last_data = [None] * 8
        self.running = False
    
    def start(self):
        for i, cam_id in enumerate(self.cam_ids):
            self.caps[i] = cv2.VideoCapture(cam_id)
            self.caps[i].set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()
    
    def stop(self):
        self.running = False
        for cap in self.caps:
            if cap: cap.release()
        for hand in self.mp_hands:
            hand.close()
    
    def process_loop(self):
        while self.running:
            # 카메라 프레임 캡처
            frames = [None, None]
            for i, cap in enumerate(self.caps):
                if cap:
                    for _ in range(4):  # 버퍼 플러시
                        ret, frame = cap.read()
                        if ret:
                            frames[i] = frame
            
            # 손 끝 감지 및 로봇 관절 위치 획득
            tips = [self.detect_hand_tip(frame, self.mp_hands[i]) 
                    if frame is not None else (None, None) 
                    for i, frame in enumerate(frames)]
            
            joints = self.robot.get_follower_positions() if self.robot else [None] * 4
            
            # 데이터 병합 및 저장
            data = [tips[0][0], tips[0][1], tips[1][0], tips[1][1]] + joints
            
            with self.data_lock:
                self.last_frames = frames
                self.last_data = data
            
            time.sleep(0.01)
    
    @staticmethod
    def detect_hand_tip(frame, hand_module):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_module.process(rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[8]  # 검지 손가락 끝
            h, w = frame.shape[:2]
            return int(lm.x * w), int(lm.y * h)
        return None, None
    
    def get_last_frame(self, idx):
        with self.data_lock:
            frame = self.last_frames[idx]
            if frame is None:
                # 더미 프레임 생성
                dummy = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(dummy, f"Camera {idx+1} not ready", (70, 240), 1, 2, (255, 255, 255), 2)
                return dummy
            return frame.copy()
    
    def get_last_data(self):
        with self.data_lock:
            return self.last_data.copy()
    
    def save_snapshot(self):
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.get_last_data())

# 글로벌 변수
robot, collector = None, None

# 프레임 인코딩 및 응답 생성 함수 (두 라우트에서 공유)
def generate_frames(cam_idx):
    while True:
        frame = collector.get_last_frame(cam_idx)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
               jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(0), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(1), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    collector.save_snapshot()
    return jsonify({"success": True})

@app.route('/status')
def status():
    data = collector.get_last_data()
    
    # NumPy 값을 Python 기본 타입으로 변환
    safe_data = []
    for v in data:
        if v is None:
            safe_data.append(None)
        elif isinstance(v, np.generic):
            safe_data.append(v.item())
        else:
            safe_data.append(int(v) if isinstance(v, (int, float)) else v)
    
    return jsonify(dict(zip(CSV_HEADERS, safe_data)))

def main():
    global robot, collector
    
    # 로봇 초기화
    robot = ManipulatorRobot(KochRobotConfig())
    robot.connect()
    robot.disable_follower_torque()
    
    # 수집기 초기화
    collector = DualCameraCollector(robot)
    collector.start()
    
    # CSV 파일 생성 (필요시)
    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    
    # 웹 서버 시작
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()