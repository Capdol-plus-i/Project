#!/usr/bin/env python3
import logging
import numpy as np
import argparse
import time
import cv2
import threading
import os
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False

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

        # Import dynamically to avoid dependency issues
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        
        # First disable torque for configuration
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        
        # Configure operating mode properly for each motor type
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                # Use integer 4 (Extended Position Control Mode)
                arms[name].write("Operating_Mode", 4)
                # Re-enable torque
                arms[name].write("Torque_Enable", 1)

        self.is_connected = True
        logger.info("Robot connected successfully")

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
                #logger.info(f"Sent action to {arm_type} arm: {safe_goal_pos}")
                return True
            except Exception as e:
                logger.error(f"Error sending action: {e}")
                return False

    def move_to_joint_positions(self, joint_positions, arm_type='follower'):
        """
        4개의 관절 위치로 로봇 팔을 이동시킵니다.
        
        Args:
            joint_positions: 4개의 관절 위치 값 (리스트 또는 NumPy 배열)
            arm_type: 제어할 팔 유형 ('follower' 또는 'leader')
        
        Returns:
            bool: 명령 전송 성공 여부
        """
        # 4개의 값이 입력되었는지 확인
        if len(joint_positions) != 4:
            logger.error(f"Expected 4 joint positions, but got {len(joint_positions)}")
            return False
        
        # NumPy 배열로 변환
        action = np.array(joint_positions, dtype=np.float32)
        
        # 관절 위치로 이동
        return self.send_action(action, arm_type=arm_type)

    def get_current_positions(self, arm_type='follower'):
        """
        현재 로봇 팔의 관절 위치를 반환합니다.
        
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


class DualCameraHandRobotController:
    """
    두 개의 카메라로 손 움직임을 인식하여 로봇을 제어하는 클래스
    """
    def __init__(self, robot, model_parameters_path="model_parameters.npz", arm_type='follower'):
        """
        초기화
        
        Args:
            robot: ManipulatorRobot 인스턴스
            model_parameters_path: 신경망 파라미터 파일 경로
            arm_type: 제어할 팔 유형 ('follower' 또는 'leader')
        """
        self.robot = robot
        self.arm_type = arm_type
        self.camera1_id = 0  # 첫 번째 카메라 ID
        self.camera2_id = 2  # 두 번째 카메라 ID
        self.width = 640
        self.height = 480
        self.running = False
        
        # 카메라 1 관련 변수
        self.cap1 = None
        self.tip1_x = 0
        self.tip1_y = 0
        self.display_frame1 = None
        self.cam1_hand_detected = False
        
        # 카메라 2 관련 변수
        self.cap2 = None
        self.tip2_x = 0
        self.tip2_y = 0
        self.display_frame2 = None
        self.cam2_hand_detected = False
        
        # 두 개의 손 인식 모델 초기화
        self.hands1 = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.2,
            max_num_hands=1
        )
        
        self.hands2 = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.2,
            max_num_hands=1
        )
        
        # 신경망 파라미터 로드
        try:
            self.load_model_parameters(model_parameters_path)
            logger.info(f"Neural network parameters loaded from {model_parameters_path}")
        except Exception as e:
            logger.error(f"Error loading neural network parameters: {e}")
            self.parameters = None
        
        # 화면 표시용 데이터
        self.predicted_joints = np.zeros(4)
        
        # 디스플레이 창 크기
        self.display_width = self.width * 2 + 20  # 두 카메라 화면과 간격
        self.display_height = self.height
        
        # z값 기본 설정
        self.z_value = 10  # 기본 z 값 (두 번째 카메라의 tip_y로 대체됨)
    
    def load_model_parameters(self, model_path):
        """신경망 파라미터 로드"""
        param_data = np.load(model_path)
        self.parameters = {k: param_data[k] for k in param_data.files}
    
    def predict_joints(self, x, y, z):
        """
        손 위치로부터 관절 위치 예측
        
        Args:
            x, y: 첫 번째 카메라의 손 검지 끝 좌표
            z: 두 번째 카메라의 손 검지 끝 y좌표 (대신 z값으로 사용)
        
        Returns:
            numpy.ndarray: 4개의 관절 위치 예측값
        """
        if self.parameters is None:
            return np.zeros(4)
        
        # 입력 벡터 생성
        #print(f"Input coordinates: x={x}, y={y}, z={z}")
        X = np.array([[x], [y], [z]])
        X = (X / 600).astype(np.float64)
        
        # 순전파 계산
        A = X
        L = len(self.parameters) // 2
        
        # 은닉층 (ReLU)
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = W.dot(A) + b
            A = np.maximum(0, Z)
            
        # 출력층 (선형)
        Wl = self.parameters[f'W{L}']
        bl = self.parameters[f'b{L}']
        ZL = Wl.dot(A) + bl
        
        # 스케일 복원
        predictions = ZL * 4000
        #print(f"Predicted joints: {predictions}")
        return predictions.flatten()
    
    def start_cameras(self):
        """두 카메라 스트림 시작"""
        # 첫 번째 카메라 시작
        logger.info(f"Opening camera 1 with ID {self.camera1_id}")
        self.cap1 = cv2.VideoCapture(self.camera1_id)
        
        # 카메라 설정
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap1.isOpened():
            logger.error(f"Cannot open camera 1 with ID {self.camera1_id}")
            return False
        
        logger.info(f"Camera 1 opened successfully")
        
        # 두 번째 카메라 시작
        logger.info(f"Opening camera 2 with ID {self.camera2_id}")
        self.cap2 = cv2.VideoCapture(self.camera2_id)
        
        # 카메라 설정
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap2.isOpened():
            logger.warning(f"Cannot open camera 2 with ID {self.camera2_id}, continuing with default z value")
            self.cap2 = None
        else:
            logger.info(f"Camera 2 opened successfully")
        
        # 첫 번째 카메라가 열렸으면 성공으로 간주
        return self.cap1 is not None
    
    def process_frame1(self):
        """첫 번째 카메라 프레임 처리 (x, y 좌표용)"""
        if self.cap1 is None or not self.cap1.isOpened():
            return None
            
        ret, frame = self.cap1.read()
        if not ret:
            logger.warning("Failed to read frame from camera 1")
            return None
        
        # OpenCV는 BGR 형식, MediaPipe는 RGB 형식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 손 감지
        results = self.hands1.process(frame_rgb)
        
        # 프레임 수정 가능하게 변경
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 이전 값 저장 (손이 인식되지 않을 경우 유지)
        old_tip_x, old_tip_y = self.tip1_x, self.tip1_y
        self.cam1_hand_detected = False
        
        # 손이 감지된 경우
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness if results.multi_handedness else [None] * len(results.multi_hand_landmarks)
            ):
                # 손 랜드마크 그리기
                # mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
                
                # 검지 끝(8번 랜드마크) 좌표 추출
                index_finger_tip = hand_landmarks.landmark[8]
                self.tip1_x = int(index_finger_tip.x * self.width)
                self.tip1_y = int(index_finger_tip.y * self.height)
                self.cam1_hand_detected = True
                
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
                    f"Tip: x={self.tip1_x}, y={self.tip1_y}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 검지 끝 위치에 원 그리기
                cv2.circle(
                    frame,
                    (self.tip1_x, self.tip1_y),
                    10,
                    (0, 0, 255),
                    -1
                )
        else:
            # 손이 인식되지 않으면 이전 값 유지
            self.tip1_x, self.tip1_y = old_tip_x, old_tip_y
        
        # 카메라 표시
        # cv2.putText(
        #     frame,
        #     "Camera 1 (X, Y)",
        #     (10, self.height - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2
        # )
        
        # 손 인식 상태 표시
        # detection_status = "Hand Detected" if self.cam1_hand_detected else "No Hand"
        # cv2.putText(
        #     frame,
        #     detection_status,
        #     (self.width - 200, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 255, 0) if self.cam1_hand_detected else (0, 0, 255),
        #     2
        # )
        
        self.display_frame1 = frame
        return frame
    
    def process_frame2(self):
        """두 번째 카메라 프레임 처리 (z 값으로 사용할 y 좌표용)"""
        if self.cap2 is None or not self.cap2.isOpened():
            # 두 번째 카메라가 없는 경우 기본 표시 프레임 반환
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Camera 2 not available",
                (50, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            self.display_frame2 = frame
            return frame
            
        ret, frame = self.cap2.read()
        if not ret:
            logger.warning("Failed to read frame from camera 2")
            # 에러 표시 프레임 반환
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Camera 2 read error",
                (50, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
            self.display_frame2 = frame
            return frame
        
        # OpenCV는 BGR 형식, MediaPipe는 RGB 형식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 손 감지
        results = self.hands2.process(frame_rgb)
        
        # 프레임 수정 가능하게 변경
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 이전 값 저장 (손이 인식되지 않을 경우 유지)
        old_tip_y = self.tip2_y
        self.cam2_hand_detected = False
        
        # 손이 감지된 경우
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness if results.multi_handedness else [None] * len(results.multi_hand_landmarks)
            ):
                # 손 랜드마크 그리기
                # mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
                
                # 검지 끝(8번 랜드마크) 좌표 추출
                index_finger_tip = hand_landmarks.landmark[8]
                self.tip2_x = int(index_finger_tip.x * self.width)
                self.tip2_y = int(index_finger_tip.y * self.height)
                self.cam2_hand_detected = True
                
                # Z값으로 사용할 좌표를 업데이트
                self.z_value = self.tip2_y
                
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
                
                # Y 좌표 표시 (z값으로 사용됨)
                cv2.putText(
                    frame,
                    f"Z value (from Y): {self.tip2_y}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 검지 끝 위치에 원 그리기
                cv2.circle(
                    frame,
                    (self.tip2_x, self.tip2_y),
                    10,
                    (0, 0, 255),
                    -1
                )
        else:
            # 손이 인식되지 않으면 이전 값 유지
            self.tip2_y = old_tip_y
        
        # 카메라 표시
        # cv2.putText(
        #     frame,
        #     "Camera 2 (Z from Y)",
        #     (10, self.height - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2
        # )
        
        # 손 인식 상태 표시
        # detection_status = "Hand Detected" if self.cam2_hand_detected else "No Hand"
        # cv2.putText(
        #     frame,
        #     detection_status,
        #     (self.width - 200, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 255, 0) if self.cam2_hand_detected else (0, 0, 255),
        #     2
        # )
        
        self.display_frame2 = frame
        return frame
    
    def create_combined_display(self):
        """두 카메라 화면을 나란히 표시하는 디스플레이 생성"""
        # 두 카메라 프레임이 모두 있는지 확인
        if self.display_frame1 is None:
            self.display_frame1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                self.display_frame1,
                "Camera 1 not available",
                (50, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        if self.display_frame2 is None:
            self.display_frame2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                self.display_frame2,
                "Camera 2 not available",
                (50, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # 빈 디스플레이 생성
        combined_display = np.zeros((self.height, self.display_width, 3), dtype=np.uint8)
        
        # 첫 번째 카메라 화면 복사
        combined_display[0:self.height, 0:self.width] = self.display_frame1
        
        # 두 번째 카메라 화면 복사
        combined_display[0:self.height, self.width + 20:] = self.display_frame2
        
        # 중앙에 경계선 그리기
        cv2.line(
            combined_display,
            (self.width + 10, 0),
            (self.width + 10, self.height),
            (200, 200, 200),
            1
        )
        
        # 예측된 관절 위치 표시
        pred_text = f"Predicted Joints: [{', '.join([str(int(j)) for j in self.predicted_joints])}]"
        cv2.putText(
            combined_display,
            pred_text,
            (10, self.height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # 입력 좌표 표시
        coords_text = f"Input Coords: X={self.tip1_x}, Y={self.tip1_y}, Z={self.z_value}"
        cv2.putText(
            combined_display,
            coords_text,
            (10, self.height - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        return combined_display
    
    def start(self):
        """메인 루프 시작"""
        if not self.start_cameras():
            logger.error("Failed to start cameras")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _main_loop(self):
        """메인 처리 루프 (별도 스레드)"""
        logger.info("Starting dual camera hand control loop")
        
        try:
            while self.running:
                # 첫 번째 카메라 프레임 처리 (x, y 좌표용)
                self.process_frame1()
                
                # 두 번째 카메라 프레임 처리 (z 값으로 사용할 y 좌표용)
                self.process_frame2()
                
                # 두 카메라의 손 감지 정보가 모두 있는 경우만 신경망 예측 진행
                if self.cam1_hand_detected:
                    # 신경망으로 관절 위치 예측
                    self.predicted_joints = self.predict_joints(
                        self.tip1_x,
                        self.tip1_y,
                        self.z_value  # 두 번째 카메라의 y 좌표 또는 기본값
                    )
                    
                    # 손이 감지된 경우에만 로봇 제어
                    if np.sum(self.predicted_joints) > 0:  # 예측값이 0이 아닌 경우
                        # 로봇 관절 제어
                        self.robot.move_to_joint_positions(
                            self.predicted_joints.round().astype(int),
                            arm_type=self.arm_type
                        )
                
                # 화면 표시
                #combined_display = self.create_combined_display()
                #cv2.imshow('Dual Camera Hand Controlled Robot', combined_display)
                
                # 'q' 키 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 약간의 대기 시간 (CPU 사용량 감소)
                time.sleep(0.001)
        
        except Exception as e:
            logger.error(f"Error in dual camera hand control loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """처리 중지 및 리소스 해제"""
        self.running = False
        
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if hasattr(self, 'hands1'):
            self.hands1.close()
        
        if hasattr(self, 'hands2'):
            self.hands2.close()
        
        if hasattr(self, 'cap1') and self.cap1 and self.cap1.isOpened():
            self.cap1.release()
        
        if hasattr(self, 'cap2') and self.cap2 and self.cap2.isOpened():
            self.cap2.release()
        
        cv2.destroyAllWindows()
        logger.info("Dual camera hand controller stopped")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Dual Camera Hand Controlled Robot')
    parser.add_argument('--arm', type=str, default='follower', choices=['follower', 'leader'],
                        help='Control arm type (follower or leader)')
    parser.add_argument('--model', type=str, default='lerobot/capdol/model_parameters.npz',
                        help='Path to neural network model parameters file')
    parser.add_argument('--camera1', type=int, default=0,
                        help='Camera 1 ID (for X,Y coordinates) (default: 0)')
    parser.add_argument('--camera2', type=int, default=2,
                        help='Camera 2 ID (for Z value from Y coordinate) (default: 2)')
    args = parser.parse_args()
    
    try:
        # 로봇 초기화
        logger.info("Initializing robot...")
        robot_config = KochRobotConfig()
        robot = ManipulatorRobot(robot_config)
        
        # 로봇 연결
        logger.info("Connecting to robot...")
        robot.connect()
        
        # 듀얼 카메라 핸드 컨트롤러 초기화 및 시작
        logger.info("Starting dual camera hand controller...")
        controller = DualCameraHandRobotController(robot, args.model, args.arm)
        controller.camera1_id = args.camera1
        controller.camera2_id = args.camera2
        controller.start()
        
        # 메인 스레드는 여기서 대기
        print("\n=== Dual Camera Hand Controlled Robot ===")
        print(f"- Using camera 1 (ID: {args.camera1}) for X and Y coordinates")
        print(f"- Using camera 2 (ID: {args.camera2}) for Z value (from Y coordinate)")
        print(f"- Controlling the {args.arm} arm")
        print("Move your hand in front of both cameras to control the robot")
        print("Press 'q' in the video window to quit\n")
        
        try:
            # 사용자가 Ctrl+C를 누를 때까지 대기
            while controller.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping by user request...")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # 리소스 정리
        if 'controller' in locals():
            controller.stop()
        
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()