# camera.py
import os, subprocess, time, logging
import cv2
import numpy as np
import mediapipe as mp
import queue
import threading

logger = logging.getLogger(__name__)

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_camera_info():
    """Get information about available cameras using v4l2-ctl"""
    cameras = []
    try:
        # List video devices
        if os.path.exists('/dev/video0'):  # Check if we're on a Linux system with video devices
            for i in range(10):  # Check first 10 potential cameras
                device_path = f'/dev/video{i}'
                if os.path.exists(device_path):
                    try:
                        # Try to get camera name using v4l2-ctl
                        cmd = ['v4l2-ctl', '--device', device_path, '--info']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                        name = "Unknown camera"
                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                name = line.split(':')[1].strip()
                                break
                        cameras.append({
                            'device': device_path,
                            'id': i,
                            'name': name
                        })
                    except (subprocess.SubprocessError, FileNotFoundError):
                        # Fallback if v4l2-ctl fails or isn't installed
                        cameras.append({
                            'device': device_path,
                            'id': i,
                            'name': "Camera (details unavailable)"
                        })
        else:
            # Non-Linux system or no video devices
            logger.info("No Linux video devices found or not a Linux system")
    except Exception as e:
        logger.error(f"Error getting camera info: {e}")
    
    # If no cameras found through v4l2, try using OpenCV
    if not cameras:
        logger.info("Trying to detect cameras using OpenCV")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    'device': f'Camera {i}',
                    'id': i,
                    'name': "Camera detected by OpenCV"
                })
                cap.release()
    
    return cameras

def create_error_frame(text):
    """Create a simple error frame with text"""
    img = np.zeros((480, 640, 3), np.uint8)
    # Add a more visible background for text
    cv2.rectangle(img, (20, 210), (620, 270), (50, 50, 50), -1)
    # Add text
    cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Add instructions
    cv2.putText(img, "Please check camera connections and permissions", 
                (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img

def create_test_pattern():
    """Create a test pattern image when no camera is available"""
    img = np.zeros((480, 640, 3), np.uint8)
    
    # Add color bars
    colors = [
        (255, 0, 0),    # Blue (BGR)
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 255) # White
    ]
    
    bar_width = img.shape[1] // len(colors)
    for i, color in enumerate(colors):
        x1 = i * bar_width
        x2 = (i + 1) * bar_width
        img[:240, x1:x2] = color
    
    # Add text
    cv2.putText(img, "Camera Not Available", (160, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Using Test Pattern", (180, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add frame counter box
    cv2.rectangle(img, (10, 430), (150, 470), (50, 50, 50), -1)
    
    return img

def encode_frame_jpeg(image, quality=85):
    """이미지를 JPEG 형식으로 인코딩하고 HTTP 응답에 맞는 바이트 형식으로 반환"""
    if image is None:
        return None
    
    try:
        ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            logger.error("Failed to encode frame")
            return None
            
        # HTTP 응답 형식으로 변환
        frame_data = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        return frame_data
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None

class FrameProcessor(threading.Thread):
    """별도 스레드에서 프레임 처리를 수행하는 클래스"""
    def __init__(self, camera_processor):
        super().__init__()
        self.camera_processor = camera_processor
        self.daemon = True
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)  # 최신 프레임만 유지
        
    def run(self):
        self.running = True
        while self.running:
            try:
                # 카메라 프레임 처리
                image = self.camera_processor.process_frame(
                    self.camera_processor.hand_controller, 
                    self.camera_processor.control_mode
                )
                
                # 이전 프레임 제거하고 새 프레임만 큐에 추가
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # 인코딩된 데이터 저장
                frame_data = encode_frame_jpeg(image)
                if frame_data:
                    self.frame_queue.put(frame_data)
                
                # 30fps 유지를 위한 짧은 대기
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=1.0)
    
    def get_latest_frame(self):
        """가장 최근에 처리된 프레임 데이터 반환"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

class CameraProcessor:
    def __init__(self, camera_id=8, width=640, height=480):  # 해상도 320x240으로 낮춤
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_count = 0
        self.cap = None
        self.use_test_pattern = False
        self.frame_processor = None
        self.hand_controller = None
        self.control_mode = "none"
        
        # MediaPipe Hands 초기화
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,  # 낮게 설정하여 추적 최적화
            max_num_hands=1)  # 한 손만 감지하도록 제한
            
    def start_processing(self):
        """별도 스레드에서 프레임 처리 시작"""
        self.frame_processor = FrameProcessor(self)
        self.frame_processor.start()
        return self.frame_processor
        
    def open_camera(self):
        """카메라 열기"""
        # 이전 캡처가 있으면 닫기
        if self.cap is not None and hasattr(self.cap, 'release'):
            self.cap.release()
            time.sleep(0.5)  # 재연결 전 대기
        
        logger.info(f"Connecting to camera ID: {self.camera_id}")
        
        # 다른 캡처 API 시도
        if os.name == 'posix':  # Linux/Mac
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        else:  # Windows 또는 기타
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # 백엔드 없이 시도
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera ID: {self.camera_id}")
            return False
            
        logger.info(f"Connected to camera ID: {self.camera_id}")
        
        # 카메라 매개변수 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 카메라 포맷 설정 (MJPG가 더 빠를 수 있음)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 버퍼 크기 줄여서 지연 시간 최소화
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 테스트 프레임으로 카메라 작동 확인
        for _ in range(5):  # 여러 번 시도
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                return True
            time.sleep(0.1)
        
        logger.error(f"Camera {self.camera_id} opened but failed initial frame read")
        return False
        
    def process_frame(self, hand_controller=None, control_mode="none"):
        """프레임 처리"""
        self.frame_count += 1
        
        if self.use_test_pattern:
            # 테스트 패턴 이미지 생성
            image = create_test_pattern()
            # 프레임 카운터 추가
            cv2.putText(image, f"Frame: {self.frame_count}", (20, 455), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            time.sleep(0.033)  # ~30 FPS
            return image
            
        # 카메라에서 읽기
        success, image = self.cap.read()
        
        if not success or image is None or image.size == 0:
            logger.warning(f"Failed to read frame from camera {self.camera_id}")
            return None
        
        # MediaPipe로 이미지 처리
        try:
            # 이미지 반전 - 처리 전에 먼저 적용
            #image = cv2.flip(image, 0)  # 상하 반전
            
            # RGB로 변환 (MediaPipe용)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # BGR로 다시 변환 (OpenCV용)
            image.flags.writeable = True
            
            # 손 랜드마크 처리
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 손 유형 가져오기
                    hand_type = handedness.classification[0].label
                    
                    # 손목 위치 가져오기
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist.x * self.width), int(wrist.y * self.height)
                    wrist_z = int(wrist.z * 10000000000)  # z 위치를 mm로 변환
                    
                    index_finger_tip = hand_landmarks.landmark[8]
                    tip_x = int(index_finger_tip.x * self.width)
                    tip_y = int(index_finger_tip.y * self.height)
                    tip_z = int(index_finger_tip.z * 1)

                    # 손 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # 손목 좌표 텍스트 추가
                    cv2.putText(
                        image,
                        f"Tip: x={tip_x}, y={tip_y}, z={tip_z}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # 손 유형 표시
                    cv2.putText(
                        image, 
                        hand_type, 
                        (wrist_x - 30, wrist_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # 로봇 제어를 위한 손 랜드마크 처리
                    if hand_controller and control_mode == "hand":
                        hand_controller.process_hand_landmarks(
                            hand_landmarks, self.width, self.height, handedness)
            
            # 현재 제어 모드 표시
            cv2.putText(
                image,
                f"Control: {control_mode}", 
                (self.width - 200, self.height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # 프레임 카운터 추가
            cv2.putText(
                image,
                f"Frame: {self.frame_count}", 
                (10, self.height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return create_error_frame(f"Error: {str(e)[:30]}")
    
    def get_frame_generator(self):
        """HTTP 스트리밍을 위한 프레임 제너레이터"""
        if not self.frame_processor or not self.frame_processor.is_alive():
            self.start_processing()
            
        while True:
            # 카메라 연결 확인
            if not self.cap or not self.cap.isOpened():
                if not self.open_camera():
                    # 카메라 연결 실패 시 에러 프레임 전송
                    error_img = create_error_frame("Camera connection failed")
                    error_data = encode_frame_jpeg(error_img)
                    if error_data:
                        yield error_data
                    time.sleep(1)
                    continue
            
            # 최신 프레임 가져오기
            frame_data = self.frame_processor.get_latest_frame()
            if frame_data:
                yield frame_data
            else:
                # 프레임이 없는 경우 대기
                time.sleep(0.03)  # ~30fps
    
    def close(self):
        """리소스 정리"""
        if self.frame_processor:
            self.frame_processor.stop()
            
        if self.hands:
            self.hands.close()
            
        if self.cap and hasattr(self.cap, 'release'):
            self.cap.release()