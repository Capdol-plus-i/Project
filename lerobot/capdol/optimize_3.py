#!/usr/bin/env python3
import os
import logging
import numpy as np
import argparse
import time
import cv2
import json
import threading
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 캘리브레이션 디렉토리 설정
CALIBRATION_DIR = "lerobot/capdol/calibration_data"

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False
        self.calibration_loaded = False
        
        # 토크 모드 가져오기
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

    def setup_robot_control(self, arm_type='follower'):
        """로봇 제어를 위한 설정"""
        if not self.is_connected:
            logger.error("Robot is not connected. Cannot setup for control.")
            return False
        
        try:
            # Configure specified arm
            arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
            
            for name in arms:
                # 먼저 토크 비활성화
                arms[name].write("Torque_Enable", self.TorqueMode.DISABLED.value)
                
                # Position Control Mode 설정
                arms[name].write("Operating_Mode", 3)
                
                # 토크 다시 활성화
                arms[name].write("Torque_Enable", 1)
            
            logger.info(f"Robot setup for {arm_type} control completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up robot for control: {e}")
            return False
    
    def move_to_joint_positions(self, joint_positions, arm_type='follower'):
        """지정된 관절 위치로 이동"""
        if len(joint_positions) != 4:
            logger.error(f"Expected 4 joint positions, got {len(joint_positions)}")
            return False
        return self.send_action(np.array(joint_positions, dtype=np.int32), arm_type)

    def send_action(self, action, arm_type='follower'):
        """로봇 팔에 동작 명령 전송"""
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
            
            # 현재 위치 읽기
            present_pos = np.array(arms[name].read("Present_Position"), dtype=np.float32)
            
            # 목표 위치를 numpy 배열로 변환
            if hasattr(goal_pos, 'numpy'):
                goal_pos = goal_pos.numpy()
            goal_pos = np.array(goal_pos, dtype=np.int32)
            
            # 안전 제한 적용
            max_delta = 150.0
            diff = goal_pos - present_pos
            diff = np.clip(diff, -max_delta, max_delta)
            safe_goal_pos = present_pos + diff
            safe_goal_pos = np.round(safe_goal_pos).astype(np.int32)
            
            # 명령 전송
            try:
                arms[name].write("Goal_Position", safe_goal_pos)
                return True
            except Exception as e:
                logger.error(f"Error sending action: {e}")
                return False

    def get_current_positions(self, arm_type='follower'):
        """현재 관절 위치 가져오기"""
        if not self.is_connected:
            logger.error("Robot not connected")
            return None
            
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        positions = []
        
        for name in arms:
            positions.extend(arms[name].read("Present_Position"))
            
        return np.array(positions)

    def disconnect(self):
        """로봇 연결 종료"""
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


class DualCameraHandRobotController:
    """두 개의 카메라를 사용하여 손 위치로 로봇 제어"""
    def __init__(self, robot, model_parameters_path="lerobot/capdol/models/model_parameters_resnet.npz", arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.camera1_id = 2  # X, Y 카메라
        self.camera2_id = 0  # Z 카메라
        self.width = 640
        self.height = 480
        self.running = False
        
        # 카메라 관련 변수
        self.caps = [None, None]
        self.hand_detected = [False, False]
        self.tip_coords = [(0, 0), (0, 0)]  # [(x1, y1), (x2, y2)]
        self.display_frames = [None, None]
        
        # MediaPipe 손 인식 모델 초기화
        self.hands_models = [
            mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, 
                         min_tracking_confidence=0.3, max_num_hands=1),
            mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, 
                         min_tracking_confidence=0.3, max_num_hands=1)
        ]
        
        # 신경망 파라미터 로드
        try:
            self.load_model_parameters(model_parameters_path)
            logger.info(f"Neural network parameters loaded successfully")
        except Exception as e:
            logger.error(f"Error loading neural network parameters: {e}")
            self.parameters = None
        
        # 상태 변수
        self.predicted_joints = np.zeros(4)
        self.z_value = 10  # 기본 Z 값
    
    def load_model_parameters(self, model_path):
        """신경망 파라미터 로드"""
        param_data = np.load(model_path)
        self.parameters = {k: param_data[k] for k in param_data.files}
    
    def predict_joints(self, x, y, z):
        """손 좌표에서 관절 위치 예측"""
        if self.parameters is None:
            return np.zeros(4)
        
        # 입력 벡터
        X = np.array([[x], [y], [z]]) / 700.0
        
        # 순방향 전파
        A = X
        L = len(self.parameters) // 2
        
        # 은닉층 (ReLU)
        for l in range(1, L):
            W, b = self.parameters[f'W{l}'], self.parameters[f'b{l}']
            A = np.maximum(0, W.dot(A) + b)
            
        # 출력층 (선형)
        W, b = self.parameters[f'W{L}'], self.parameters[f'b{L}']
        predictions = (W.dot(A) + b) * 400
        
        return predictions.flatten()
    
    def start_cameras(self):
        """두 카메라 시작"""
        for i, cam_id in enumerate([self.camera1_id, self.camera2_id]):
            logger.info(f"Opening camera {i+1} with ID {cam_id}")
            
            # 카메라 초기화 시도 (플랫폼별 처리)
            if os.name == 'posix':  # Linux/Mac
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cam_id)
            else:  # Windows
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # 테스트 프레임 읽기
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.caps[i] = cap
                    logger.info(f"Camera {i+1} opened successfully: {self.width}x{self.height}")
                else:
                    logger.error(f"Failed to read test frame from camera {i+1}")
                    cap.release()
                    if i == 0:  # 첫 번째 카메라는 필수
                        return False
            else:
                logger.error(f"Failed to open camera {i+1}")
                if i == 0:  # 첫 번째 카메라는 필수
                    return False
        
        return self.caps[0] is not None
    
    def process_frame(self, cam_idx):
        """카메라 프레임 처리"""
        cap = self.caps[cam_idx]
        if cap is None or not cap.isOpened():
            # 카메라가 없으면 빈 프레임 생성
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {cam_idx+1} not available", 
                      (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 255), 2)
            self.display_frames[cam_idx] = frame
            return frame
            
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame from camera {cam_idx+1}")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {cam_idx+1} read error", 
                      (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 0, 0), 2)
            self.display_frames[cam_idx] = frame
            return frame
        
        # 이전 좌표 저장
        old_x, old_y = self.tip_coords[cam_idx]
        self.hand_detected[cam_idx] = False
        
        # MediaPipe로 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_models[cam_idx].process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 손 랜드마크 처리
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # 검지 끝(8번 랜드마크) 좌표 추출
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)
                self.tip_coords[cam_idx] = (x, y)
                self.hand_detected[cam_idx] = True
                
                # Z 값 업데이트 (카메라 2에서)
                if cam_idx == 1:
                    self.z_value = y
                
                # 손가락 위치 표시
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
                # 텍스트 정보
                info_text = f"x={x}, y={y}" if cam_idx == 0 else f"z={y}"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 손이 감지되지 않으면 이전 값 유지
            self.tip_coords[cam_idx] = (old_x, old_y)
        
        # 추가 정보 표시
        cv2.putText(frame, f"CAM {cam_idx+1}", (self.width - 100, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # 캘리브레이션 상태 표시
        calibration_status = "CALIBRATED" if self.robot.calibration_loaded else "NOT CALIBRATED"
        cv2.putText(
            frame,
            calibration_status,
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.robot.calibration_loaded else (0, 0, 255),
            2
        )
        
        self.display_frames[cam_idx] = frame
        return frame
    
    def create_combined_display(self):
        """두 카메라 화면을 합쳐서 표시"""
        # 프레임이 없으면 처리
        if any(frame is None for frame in self.display_frames):
            for i in range(2):
                if self.display_frames[i] is None:
                    self.process_frame(i)
        
        # 결합된 디스플레이 생성
        combined = np.zeros((self.height, self.width*2 + 20, 3), dtype=np.uint8)
        
        # 카메라 프레임 추가
        if self.display_frames[0] is not None:
            combined[0:self.height, 0:self.width] = self.display_frames[0]
        
        if self.display_frames[1] is not None:
            combined[0:self.height, self.width+20:] = self.display_frames[1]
        
        # 구분선
        cv2.line(combined, (self.width+10, 0), (self.width+10, self.height), 
               (200, 200, 200), 1)
        
        # 관절 위치 및 좌표 정보 추가
        x, y = self.tip_coords[0]
        z = self.z_value
        
        joint_text = f"Joints: [{', '.join([str(int(j)) for j in self.predicted_joints])}]"
        coord_text = f"Input: X={x}, Y={y}, Z={z}"
        control_text = f"Controlling: {self.arm_type} arm"
        
        cv2.putText(combined, joint_text, (10, self.height-20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(combined, coord_text, (10, self.height-40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(combined, control_text, (10, self.height-60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return combined
    
    def start(self):
        """컨트롤러 시작"""
        # 로봇 설정
        if not self.robot.setup_robot_control(self.arm_type):
            logger.error(f"Failed to setup robot for {self.arm_type} arm control")
            return False
        
        # 카메라 시작
        if not self.start_cameras():
            logger.error("Failed to start cameras")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _main_loop(self):
        """메인 처리 루프"""
        logger.info("Starting hand control loop")
        
        try:
            while self.running:
                # 두 카메라 프레임 처리
                self.process_frame(0)
                self.process_frame(1)
                
                # 손이 감지되면 관절 위치 예측 및 로봇 제어
                if self.hand_detected[0]:
                    x, y = self.tip_coords[0]
                    z = self.z_value
                    
                    # 예측 및 로봇 제어
                    self.predicted_joints = self.predict_joints(x, y, z)
                    if np.sum(self.predicted_joints) > 0:
                        self.robot.move_to_joint_positions(
                            self.predicted_joints.round().astype(int),
                            arm_type=self.arm_type
                        )
                
                # 결합된 화면 표시
                #cv2.imshow('Hand Controlled Robot', self.create_combined_display())
                
                # 종료 키 확인
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # CPU 사용량 감소를 위한 짧은 대기
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in control loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """정지 및 리소스 해제"""
        self.running = False
        
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # 손 모델 해제
        for hand_model in self.hands_models:
            hand_model.close()
        
        # 카메라 해제
        for i, cap in enumerate(self.caps):
            if cap and cap.isOpened():
                logger.info(f"Releasing camera {i+1}")
                cap.release()
        
        cv2.destroyAllWindows()
        logger.info("Hand controller stopped")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Dual Camera Hand Controlled Robot')
    parser.add_argument('--arm', type=str, default='follower', choices=['follower', 'leader'],
                      help='Control arm type (follower or leader)')
    parser.add_argument('--model', type=str, default='lerobot/capdol/models/model_parameters_resnet.npz',
                      help='Neural network model parameters file')
    parser.add_argument('--camera1', type=int, default=4,
                      help='Camera 1 ID (for X,Y coordinates)')
    parser.add_argument('--camera2', type=int, default=0,
                      help='Camera 2 ID (for Z value)')
    args = parser.parse_args()
    
    try:
        # 로봇 초기화
        logger.info("Initializing robot...")
        robot = ManipulatorRobot(KochRobotConfig())
        robot.connect()
        
        # 컨트롤러 시작
        logger.info("Starting hand controller...")
        controller = DualCameraHandRobotController(robot, args.model, args.arm)
        controller.camera1_id = args.camera1
        controller.camera2_id = args.camera2
        controller.start()
        
        # 상태 표시
        print("\n=== 듀얼 카메라 핸드 로봇 컨트롤러 ===")
        print(f"- 카메라 1 (ID: {args.camera1}): X, Y 좌표")
        print(f"- 카메라 2 (ID: {args.camera2}): Z 값")
        print(f"- 제어 대상: {args.arm} 암")
        print(f"- 캘리브레이션: {'로드됨' if robot.calibration_loaded else '로드되지 않음'}")
        print("종료하려면 'q'를 누르세요\n")
        
        # 사용자 종료 대기
        try:
            while controller.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n사용자에 의한 중지...")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # 정리
        if 'controller' in locals():
            controller.stop()
        
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()