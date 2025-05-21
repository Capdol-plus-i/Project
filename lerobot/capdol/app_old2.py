#!/usr/bin/env python3
"""
로봇 암 및 손 좌표 데이터 수집 프로그램 - 듀얼 카메라 Flask 웹 인터페이스 버전
단일 CSV 파일에 스냅샷 기록
"""

import os, logging, numpy as np, argparse, time, cv2, csv, json, threading, datetime
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.motors.dynamixel import CalibrationMode, TorqueMode

# 기본 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CALIBRATION_DIR = "lerobot/capdol/calibration_data"
app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False
        self.calibration_loaded = False
        
        try:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            self.TorqueMode = TorqueMode
        except ImportError:
            self.TorqueMode = None

    def connect(self):
        if self.is_connected: return
        
        for name in self.follower_arms:
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            self.leader_arms[name].connect()
            
        self.is_connected = True
        self.load_calibration()
        
    def load_calibration(self):
        leader_calib_path = os.path.join(CALIBRATION_DIR, "leader_arm.json")
        follower_calib_path = os.path.join(CALIBRATION_DIR, "follower_arm.json")
        
        # 캘리브레이션 파일 로드
        leader_calibration = None
        follower_calibration = None
        
        if os.path.exists(leader_calib_path):
            with open(leader_calib_path, 'r') as f:
                leader_calibration = json.load(f)
        
        if os.path.exists(follower_calib_path):
            with open(follower_calib_path, 'r') as f:
                follower_calibration = json.load(f)
        
        # 캘리브레이션 적용
        if leader_calibration:
            for name in self.leader_arms:
                self.leader_arms[name].set_calibration(leader_calibration)
        
        if follower_calibration:
            for name in self.follower_arms:
                self.follower_arms[name].set_calibration(follower_calibration)
        
        self.calibration_loaded = leader_calibration is not None and follower_calibration is not None
        
    def setup_for_data_collection(self):
        if not self.is_connected: return False
        
        # leader arm 토크 해제 (사용자가 직접 움직일 수 있게)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", self.TorqueMode.DISABLED.value)
        
        # follower arm 설정
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", self.TorqueMode.DISABLED.value)
            self.follower_arms[name].write("Operating_Mode", 3)
            self.follower_arms[name].write("Torque_Enable", 1)
        
        return True

    def get_positions(self, arm_type):
        if not self.is_connected: return None
        
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        positions = []
        
        for name in arms:
            pos = arms[name].read("Present_Position")
            positions.extend(pos)
            
        return np.array(positions)

    def copy_leader_to_follower(self):
        if not self.is_connected: return False
        
        leader_positions = self.get_positions('leader')
        if leader_positions is None: return False
        
        return self.send_action(leader_positions, arm_type='follower')
        
    def send_action(self, action, arm_type='follower'):
        if not self.is_connected: return False
        
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        
        from_idx, to_idx = 0, 0
        for name in arms:
            motor_count = len(arms[name].motor_names)
            to_idx += motor_count
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            
            present_pos = np.array(arms[name].read("Present_Position"), dtype=np.float32)
            
            if hasattr(goal_pos, 'numpy'):
                goal_pos = goal_pos.numpy()
            goal_pos = np.array(goal_pos, dtype=np.int32)
            
            # 안전 제한 적용
            max_delta = 150.0
            diff = goal_pos - present_pos
            diff = np.clip(diff, -max_delta, max_delta)
            safe_goal_pos = present_pos + diff
            safe_goal_pos = np.round(safe_goal_pos).astype(np.int32)
            
            try:
                arms[name].write("Goal_Position", safe_goal_pos)
                return True
            except Exception as e:
                return False

    def disconnect(self):
        if not self.is_connected: return
        
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class DualCameraDataCollector:
    def __init__(self, robot, output_dir="collected_data"):
        self.robot = robot
        self.output_dir = output_dir
        self.camera1_id = 0
        self.camera2_id = 2
        self.width = 640
        self.height = 480
        self.running = False
        
        # 카메라 및 데이터 관련 변수 초기화
        self.cap1 = None
        self.cap2 = None
        self.current_frame1 = None
        self.current_frame2 = None
        
        # CSV 파일 초기화 - 한 번만 생성
        self.csv_writer = None
        self.csv_file = None
        self.snapshot_count = 0
        
        # 손 좌표 변수
        self.tip1_x = None
        self.tip1_y = None
        self.tip2_x = None
        self.tip2_y = None
        
        # MediaPipe 손 인식 모델 초기화
        self.hands1 = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3,
            max_num_hands=1
        )
        
        self.hands2 = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3,
            max_num_hands=1
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV 파일 초기화 - 프로그램 시작 시 한 번만
        self.initialize_csv_file()
        
        self.status_info = {
            "snapshots": 0,
            "calibrated": False,
            "cam1_hand_detected": False,
            "cam2_hand_detected": False,
            "cam1_tip_coords": {"x": None, "y": None},
            "cam2_tip_coords": {"x": None, "y": None}
        }
    
    def initialize_csv_file(self):
        """CSV 파일을 한 번만 초기화하고 헤더를 작성"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"robot_hand_snapshots_{timestamp}.csv")
        
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 헤더 작성
        header = [
            "snapshot_id", "timestamp", 
            "cam1_hand_detected", "cam1_tip_x", "cam1_tip_y",
            "cam2_hand_detected", "cam2_tip_x", "cam2_tip_y",
            "leader_joint1", "leader_joint2", "leader_joint3", "leader_joint4",
            "follower_joint1", "follower_joint2", "follower_joint3", "follower_joint4"
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()  # 즉시 파일에 쓰기
    
    def start_cameras(self):
        # 카메라 초기화 및 설정
        if os.name == 'posix':
            self.cap1 = cv2.VideoCapture(self.camera1_id, cv2.CAP_V4L2)
            if not self.cap1.isOpened():
                self.cap1.release()
                self.cap1 = cv2.VideoCapture(self.camera1_id)
                
            self.cap2 = cv2.VideoCapture(self.camera2_id, cv2.CAP_V4L2)
            if not self.cap2.isOpened():
                self.cap2.release()
                self.cap2 = cv2.VideoCapture(self.camera2_id)
        else:
            self.cap1 = cv2.VideoCapture(self.camera1_id, cv2.CAP_DSHOW)
            if not self.cap1.isOpened():
                self.cap1.release()
                self.cap1 = cv2.VideoCapture(self.camera1_id)
                
            self.cap2 = cv2.VideoCapture(self.camera2_id, cv2.CAP_DSHOW)
            if not self.cap2.isOpened():
                self.cap2.release()
                self.cap2 = cv2.VideoCapture(self.camera2_id)
        
        if not self.cap1.isOpened():
            return False
            
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if self.cap2 and self.cap2.isOpened():
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        return True
    
    def take_snapshot(self):
        """현재 상태의 스냅샷을 CSV 파일에 기록"""
        if not self.csv_writer:
            return False
        
        timestamp = time.time()
        self.snapshot_count += 1
        
        # 로봇 데이터 수집
        leader_positions = self.robot.get_positions('leader')
        self.robot.copy_leader_to_follower()
        follower_positions = self.robot.get_positions('follower')
        
        # 스냅샷 데이터 구성
        row = [self.snapshot_count, timestamp]
        
        # 카메라 1 데이터
        if self.status_info["cam1_hand_detected"]:
            row.extend([True, self.tip1_x, self.tip1_y])
        else:
            row.extend([False, None, None])
        
        # 카메라 2 데이터
        if self.status_info["cam2_hand_detected"]:
            row.extend([True, self.tip2_x, self.tip2_y])
        else:
            row.extend([False, None, None])
        
        # 로봇 팔 데이터
        if leader_positions is not None:
            row.extend(leader_positions)
        else:
            row.extend([None] * 4)
        
        if follower_positions is not None:
            row.extend(follower_positions)
        else:
            row.extend([None] * 4)
        
        # CSV에 기록
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 즉시 파일에 쓰기
        
        self.status_info["snapshots"] = self.snapshot_count
        logger.info(f"Snapshot {self.snapshot_count} recorded")
        
        return True
    
    def process_frames(self):
        # 카메라 프레임 처리 (실시간 비디오 스트림용)
        frame1 = None
        frame2 = None
        
        if self.cap1 and self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if ret:
                frame1 = self.process_hand_detection(frame, 0)
                
        if self.cap2 and self.cap2.isOpened():
            ret, frame = self.cap2.read()
            if ret:
                frame2 = self.process_hand_detection(frame, 2)
        
        self.current_frame1 = frame1
        self.current_frame2 = frame2
        
        return frame1, frame2
        
    def process_hand_detection(self, frame, camera_idx):
        if frame is None: return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 손 감지
        results = self.hands1.process(frame_rgb) if camera_idx == 0 else self.hands2.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 기본값 설정
        if camera_idx == 0:
            self.tip1_x = None
            self.tip1_y = None
            self.status_info["cam1_hand_detected"] = False
            self.status_info["cam1_tip_coords"] = {"x": None, "y": None}
        else:
            self.tip2_x = None
            self.tip2_y = None
            self.status_info["cam2_hand_detected"] = False
            self.status_info["cam2_tip_coords"] = {"x": None, "y": None}
        
        # 손이 감지된 경우 검지 끝 좌표 추출
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 검지 끝(8번 랜드마크) 좌표 추출
                index_finger_tip = hand_landmarks.landmark[8]
                tip_x = int(index_finger_tip.x * self.width)
                tip_y = int(index_finger_tip.y * self.height)
                
                if camera_idx == 0:
                    self.tip1_x = tip_x
                    self.tip1_y = tip_y
                    self.status_info["cam1_hand_detected"] = True
                    self.status_info["cam1_tip_coords"] = {"x": tip_x, "y": tip_y}
                else:
                    self.tip2_x = tip_x
                    self.tip2_y = tip_y
                    self.status_info["cam2_hand_detected"] = True
                    self.status_info["cam2_tip_coords"] = {"x": tip_x, "y": tip_y}
                
                # 검지 끝 위치에 원 그리기
                cv2.circle(frame, (tip_x, tip_y), 10, (0, 0, 255), -1)
        
        # 상태 정보 표시
        cv2.putText(frame, f"CAM {camera_idx}", (self.width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # 스냅샷 카운트 표시
        cv2.putText(frame, f"Snapshots: {self.snapshot_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def generate_frames(self, camera_idx=0):
        while self.running:
            frame = self.current_frame1 if camera_idx == 0 else self.current_frame2
                
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    
    def start(self):
        if not self.robot.setup_for_data_collection():
            return False
        
        if not self.start_cameras():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _processing_loop(self):
        try:
            while self.running:
                self.process_frames()
                time.sleep(0.03)
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
    
    def stop(self):
        self.running = False
        
        # CSV 파일 종료
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        if hasattr(self, 'hands1'):
            self.hands1.close()
        if hasattr(self, 'hands2'):
            self.hands2.close()
        
        if hasattr(self, 'cap1') and self.cap1:
            self.cap1.release()
        if hasattr(self, 'cap2') and self.cap2:
            self.cap2.release()


# 전역 객체
robot = None
collector = None

# Flask 라우트
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    if collector is None or not collector.running:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera 1 not started", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', dummy_frame)
        frame_bytes = buffer.tobytes()
        
        def generate_dummy():
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return Response(generate_dummy(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(collector.generate_frames(0), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    if collector is None or not collector.running or collector.cap2 is None:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera 2 not available", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', dummy_frame)
        frame_bytes = buffer.tobytes()
        
        def generate_dummy():
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return Response(generate_dummy(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(collector.generate_frames(2), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    """스냅샷을 찍어 CSV에 기록"""
    if collector is None or not collector.running:
        return jsonify({"success": False, "message": "Collector not running"}), 400
    
    success = collector.take_snapshot()
    return jsonify({
        "success": success, 
        "snapshot_count": collector.snapshot_count if success else 0
    })

@app.route('/status')
def get_status():
    if collector is None:
        return jsonify({"running": False})
    
    return jsonify({
        "running": collector.running,
        **collector.status_info
    })

def create_app(output_dir='lerobot/capdol/collected_data', camera1_id=0, camera2_id=2):
    global robot, collector
    
    try:
        robot_config = KochRobotConfig()
        robot = ManipulatorRobot(robot_config)
        robot.connect()
        
        collector = DualCameraDataCollector(robot, output_dir)
        collector.camera1_id = camera1_id
        collector.camera2_id = camera2_id
        collector.start()
    except Exception as e:
        logger.error(f"Error initializing: {e}")
        if robot and robot.is_connected:
            robot.disconnect()
        
    return app

def main():
    parser = argparse.ArgumentParser(description='Robot and Hand Data Collector Web Interface')
    parser.add_argument('--output', type=str, default='lerobot/capdol/collected_data',
                      help='Directory to save collected data')
    parser.add_argument('--camera1', type=int, default=4,
                      help='Camera 1 ID to use')
    parser.add_argument('--camera2', type=int, default=0,
                      help='Camera 2 ID to use')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run the server on')
    args = parser.parse_args()
    
    try:
        app_instance = create_app(args.output, args.camera1, args.camera2)
        app_instance.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping by user request...")
    finally:
        if collector:
            collector.stop()
        if robot and robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    main()