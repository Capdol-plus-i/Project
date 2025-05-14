# controllers.py
import sys, termios, tty, select, threading, time, logging
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)
mp_hands = mp.solutions.hands

class KalmanFilter:
    """칼만 필터 구현 - 위치와 속도 추정을 부드럽게 하기 위한 필터"""
    def __init__(self, dim, process_noise=0.001, measurement_noise=0.1):
        self.dim = dim
        self.x = np.zeros(dim)  # 상태 (위치)
        self.P = np.eye(dim)    # 오차 공분산
        self.Q = np.eye(dim) * process_noise       # 프로세스 노이즈
        self.R = np.eye(dim) * measurement_noise   # 측정 노이즈
        self.initialized = False
        
    def update(self, z):
        # 첫 호출 시 초기화
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x
            
        # 예측 단계
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 업데이트 단계
        K = P_pred @ np.linalg.inv(P_pred + self.R)  # 칼만 이득
        self.x = x_pred + K @ (z - x_pred)           # 상태 업데이트
        self.P = (np.eye(self.dim) - K) @ P_pred     # 공분산 업데이트
        
        return self.x
        
    def reset(self):
        """필터 상태 초기화"""
        self.x = np.zeros(self.dim)
        self.P = np.eye(self.dim)
        self.initialized = False


def inverse_kinematics(x, y, z, robot_type='koch'):
    """간단한 역기구학 구현 (실제 로봇 기구학에 맞게 수정 필요)"""
    if robot_type == 'koch':
        # 4축 로봇에 대한 단순화된 역기구학
        # 실제 로봇의 링크 길이와 기구학 방정식에 맞게 구현 필요
        L1 = 100  # 첫 번째 링크 길이 (mm)
        L2 = 120  # 두 번째 링크 길이 (mm)
        
        # 어깨 회전 각도 (joint 0)
        theta1 = np.arctan2(y, x)
        
        # 바닥 평면에서의 거리
        r = np.sqrt(x**2 + y**2)
        if r < 1.0:  # 특이점 방지
            r = 1.0
            
        # 남은 두 관절에 대한 기구학 계산
        d = np.sqrt(r**2 + z**2)
        if d > L1 + L2:  # 도달 범위 밖
            d = L1 + L2 - 1.0
        elif d < abs(L1 - L2):  # 최소 거리 제한
            d = abs(L1 - L2) + 1.0
            
        cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
        alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
        
        beta = np.arctan2(z, r)
        
        # 어깨 틸트 (joint 1)
        theta2 = beta + alpha
        
        # 팔꿈치 (joint 2)
        cos_gamma = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        theta3 = np.pi - gamma
        
        # 각도를 로봇 단위로 변환 (로봇에 맞게 조정 필요)
        j1 = theta1 * (2000 / np.pi)
        j2 = theta2 * (2000 / np.pi)
        j3 = theta3 * (2000 / np.pi)
        
        return np.array([j1, j2, j3, 0])
    else:
        # 다른 로봇 유형에 대한 역기구학 구현
        return np.zeros(4)


class JointController:
    def __init__(self, robot, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.arms = robot.follower_arms if arm_type == 'follower' else robot.leader_arms
        self.position_filter = KalmanFilter(3, process_noise=0.001, measurement_noise=0.1)
        self.joint_filter = KalmanFilter(4, process_noise=0.001, measurement_noise=0.1)

    def move_joint(self, joint_index, delta_value):
        """특정 관절을 상대적으로 이동"""
        try:
            # 현재 위치를 numpy 배열로 가져오기
            current_pos = np.array(self.arms['main'].read('Present_Position'), dtype=np.float32)
            
            # 목표 위치 계산
            goal_pos = current_pos.copy()
            goal_pos[joint_index] += delta_value
            
            # 목표 위치 전송
            self.robot.send_action(goal_pos, arm_type=self.arm_type)
            logger.debug(f"Moved {self.arm_type} joint {joint_index} by {delta_value}")
        except Exception as e:
            logger.error(f"Error moving joint: {e}")
            
    def move_to_position(self, position):
        """절대 위치로 이동"""
        try:
            # 로봇 자세한 타입에 따라 조정 필요
            current_pos = np.array(self.arms['main'].read('Present_Position'), dtype=np.float32)
            
            # 필터링 적용
            position = self.joint_filter.update(position)
            
            # 안전을 위한 기존 위치와 새 위치 간의 제한
            max_delta = 200.0  # 한 번에 최대 변화량
            diff = position - current_pos
            diff = np.clip(diff, -max_delta, max_delta)
            safe_position = current_pos + diff
            
            # 목표 위치 전송
            self.robot.send_action(safe_position, arm_type=self.arm_type)
            logger.debug(f"Moving {self.arm_type} to position {safe_position}")
            
            return safe_position
        except Exception as e:
            logger.error(f"Error moving to position: {e}")
            return None
            
    def reset_filters(self):
        """필터 상태 초기화"""
        self.position_filter.reset()
        self.joint_filter.reset()


class KeyboardController:
    def __init__(self, robot):
        self.robot = robot
        self.old_settings = None
        self.follower_ctrl = JointController(robot, 'follower')
        self.leader_ctrl = JointController(robot, 'leader')
        self.running = False
        
        # 키 매핑
        self.key_mapping = {
            # Follower arm
            'w': (self.follower_ctrl, 0, 100.0),   # Shoulder pan up
            's': (self.follower_ctrl, 0, -100.0),  # Shoulder pan down
            'e': (self.follower_ctrl, 1, 100.0),   # Shoulder tilt up
            'd': (self.follower_ctrl, 1, -100.0),  # Shoulder tilt down
            'r': (self.follower_ctrl, 2, 100.0),   # Elbow up
            'f': (self.follower_ctrl, 2, -100.0),  # Elbow down
            't': (self.follower_ctrl, 3, 100.0),   # Wrist up
            'g': (self.follower_ctrl, 3, -100.0),  # Wrist down
            
            # Leader arm
            'y': (self.leader_ctrl, 0, 100.0),     # Shoulder pan up
            'h': (self.leader_ctrl, 0, -100.0),    # Shoulder pan down
            'u': (self.leader_ctrl, 1, 100.0),     # Shoulder tilt up
            'j': (self.leader_ctrl, 1, -100.0),    # Shoulder tilt down
            'i': (self.leader_ctrl, 2, 100.0),     # Elbow up
            'k': (self.leader_ctrl, 2, -100.0),    # Elbow down
            'o': (self.leader_ctrl, 3, 100.0),     # Wrist up
            'l': (self.leader_ctrl, 3, -100.0),    # Wrist down
        }

    def start(self):
        """키보드 제어 시작 (별도 스레드)"""
        self.running = True
        self.thread = threading.Thread(target=self._keyboard_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Keyboard control started")

    def stop(self):
        """키보드 제어 중지"""
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        logger.info("Keyboard control stopped")

    def _keyboard_loop(self):
        """키보드 제어 메인 루프 (스레드에서 실행)"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    c = sys.stdin.read(1)
                    if c == 'q':
                        logger.info("Quit command received")
                        self.running = False
                        break
                    
                    if c in self.key_mapping:
                        # 키 처리
                        controller, joint_index, delta = self.key_mapping[c]
                        controller.move_joint(joint_index, delta)
                time.sleep(0.01)  # CPU 과부하 방지
        except Exception as e:
            logger.error(f"Keyboard control error: {e}")
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


class HandController:
    def __init__(self, robot):
        self.robot = robot
        self.follower_ctrl = JointController(robot, 'follower')
        self.leader_ctrl = JointController(robot, 'leader')
        self.is_active = False
        self.prev_landmarks = None
        self.smoothing_factor = 0.5  # 움직임 스무딩 계수
        
        # 필터 및 위치 추적
        self.position_filter = KalmanFilter(3, process_noise=0.001, measurement_noise=0.05)
        self.joint_filter = KalmanFilter(4, process_noise=0.001, measurement_noise=0.05)
        self.last_time = time.time()
        self.last_pos = None
        self.velocity = np.zeros(3)
        
        # 제어 매개변수
        self.acceleration_factor = 3.0    # 가속도에 따른 움직임 증폭
        self.joint_min = np.array([-2000, -2000, -2000, -500])  # 최소 관절 각도
        self.joint_max = np.array([2000, 2000, 2000, 500])      # 최대 관절 각도
        self.workspace_min = np.array([100, -200, 0])     # 작업 공간 최소값 (mm)
        self.workspace_max = np.array([400, 200, 300])    # 작업 공간 최대값 (mm)
        
        # 기본 위치 범위 설정
        self.position_min = np.array([0.1, 0.1, -0.1])  # 최소 위치 기록
        self.position_max = np.array([0.9, 0.9, 0.1])   # 최대 위치 기록
        
        # 손 제스처 관련
        self.grip_state = False
        self.grip_count = 0
        self.gesture_history = []
        
    def set_active(self, active):
        """핸드 컨트롤 활성화/비활성화"""
        self.is_active = active
        if active:
            # 활성화 시 필터 초기화
            self.position_filter.reset()
            self.joint_filter.reset()
            self.prev_landmarks = None
            self.last_pos = None
        logger.info(f"Hand control {'activated' if active else 'deactivated'}")
        
    def detect_gesture(self, hand_landmarks):
        """특정 손 제스처 감지 (그리퍼 제어용)"""
        # 손가락 랜드마크 인덱스
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        THUMB_MCP = 2
        WRIST = 0
        
        # 손가락 끝 위치 가져오기
        thumb_tip = hand_landmarks.landmark[THUMB_TIP]
        index_tip = hand_landmarks.landmark[INDEX_TIP]
        middle_tip = hand_landmarks.landmark[MIDDLE_TIP]
        ring_tip = hand_landmarks.landmark[RING_TIP]
        pinky_tip = hand_landmarks.landmark[PINKY_TIP]
        thumb_mcp = hand_landmarks.landmark[THUMB_MCP]
        wrist = hand_landmarks.landmark[WRIST]
        
        # 집기 제스처 감지 (엄지와 검지만 펴고 나머지 접었을 때)
        pinch_distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )
        
        # 손가락 접힘 여부 확인 (y 좌표 비교)
        middle_folded = middle_tip.y > thumb_mcp.y
        ring_folded = ring_tip.y > thumb_mcp.y
        pinky_folded = pinky_tip.y > thumb_mcp.y
        
        # 집는 제스처: 엄지-검지 가까움 + 나머지 손가락 접힘
        is_pinch_gesture = (pinch_distance < 0.05) and middle_folded and ring_folded and pinky_folded
        
        # 펴는 제스처: 모든 손가락 펴짐
        is_open_gesture = not (middle_folded or ring_folded or pinky_folded)
        
        # 상태 변경을 위한 히스토리 저장
        self.gesture_history.append(1 if is_pinch_gesture else (2 if is_open_gesture else 0))
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        # 연속된 제스처로 상태 변경
        if sum(1 for g in self.gesture_history if g == 1) >= 7:  # 7/10 프레임이 집기 제스처
            if not self.grip_state:
                self.grip_state = True
                logger.info("Gesture detected: Grip")
                return 0.0  # 그리퍼 닫기
        elif sum(1 for g in self.gesture_history if g == 2) >= 7:  # 7/10 프레임이 펴기 제스처
            if self.grip_state:
                self.grip_state = False
                logger.info("Gesture detected: Release")
                return 1.0  # 그리퍼 열기
                
        # 일반적인 경우: 집기 거리에 따른 그리퍼 위치
        return max(0, min(1, (pinch_distance - 0.03) / 0.07))
        
    def process_hand_landmarks(self, hand_landmarks, width, height, handedness):
        """손 랜드마크 처리 및 로봇 제어"""
        if not self.is_active:
            return
            
        try:
            # 손 유형 확인
            hand_type = handedness.classification[0].label
            controller = self.follower_ctrl if hand_type == "Left" else self.leader_ctrl
            
            # 손목 위치 가져오기
            wrist = hand_landmarks.landmark[0]
            raw_pos = np.array([wrist.x, wrist.y, wrist.z])
            
            # 칼만 필터로 위치 필터링
            filtered_pos = self.position_filter.update(raw_pos)
            
            # 속도 계산
            current_time = time.time()
            dt = current_time - self.last_time
            
            if self.last_pos is not None and dt > 0:
                # 속도 계산 및 필터링
                current_velocity = (filtered_pos - self.last_pos) / dt
                self.velocity = 0.8 * current_velocity + 0.2 * self.velocity  # 간단한 속도 필터링
            
            self.last_pos = filtered_pos.copy()
            self.last_time = current_time
            
            # 위치 정규화 (제한된 범위 내로)
            normalized_pos = np.clip((filtered_pos - self.position_min) / 
                                     (self.position_max - self.position_min), 0, 1)
            
            # 두 가지 매핑 방법 중 선택
            use_inverse_kinematics = True  # 역기구학 사용 여부
            
            if use_inverse_kinematics:
                # 방법 1: 역기구학 사용
                # 정규화된 위치를 실제 공간 좌표로 변환 (mm)
                xyz = self.workspace_min + normalized_pos * (self.workspace_max - self.workspace_min)
                
                # 속도 기반 추가 이동 (빠르게 움직이면 더 멀리)
                velocity_magnitude = np.linalg.norm(self.velocity)
                if velocity_magnitude > 0.1:  # 최소 속도 임계값
                    # 방향에 따라 작업 공간 확장
                    direction = self.velocity / velocity_magnitude
                    xyz += direction * velocity_magnitude * self.acceleration_factor * 100  # mm 단위로 변환
                
                # 역기구학으로 관절 각도 계산
                joint_pos = inverse_kinematics(xyz[0], xyz[1], xyz[2], robot_type='koch')
            else:
                # 방법 2: 직접 매핑
                # 정규화된 위치를 관절 각도로 변환
                joint_range = self.joint_max - self.joint_min
                joint_pos = self.joint_min + normalized_pos[:3] * joint_range[:3]
                
                # 속도 기반 추가 이동
                velocity_magnitude = np.linalg.norm(self.velocity)
                if velocity_magnitude > 0.2:  # 최소 속도 임계값
                    # 방향에 따라 관절 움직임 증폭
                    joint_pos[:3] += self.velocity * velocity_magnitude * self.acceleration_factor * 1000
            
            # 그리퍼 제어 (4번째 관절)
            gripper_val = self.detect_gesture(hand_landmarks)
            joint_pos[3] = self.joint_min[3] + gripper_val * (self.joint_max[3] - self.joint_min[3])
            
            # 관절 위치 필터링 및 스무딩
            if self.prev_landmarks is not None:
                # 칼만 필터 적용
                joint_pos = self.joint_filter.update(joint_pos)
                # 추가 스무딩
                joint_pos = self.smoothing_factor * joint_pos + (1 - self.smoothing_factor) * self.prev_landmarks
            
            # 다음 스무딩을 위해 현재 값 저장
            self.prev_landmarks = joint_pos.copy()
            
            # 로봇 관절 제어
            controller.move_to_position(joint_pos)
            
            # 디버그 로깅
            logger.debug(f"Hand control: {hand_type}, norm pos: {normalized_pos}, joints: {joint_pos}")
            
        except Exception as e:
            logger.error(f"Hand control error: {e}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())