#!/usr/bin/env python3
"""
손 좌표 데이터 수집 프로그램 - 듀얼 카메라 Flask 웹 인터페이스 버전

이 스크립트는 다음을 수행합니다:
1. 두 개의 카메라로 손의 검지 끝 좌표 추적
2. 두 카메라의 검지 끝 좌표를 동시에 기록
3. 수집된 데이터를 CSV 파일로 저장
4. 웹 브라우저를 통해 두 카메라 피드 확인 및 제어 가능
"""

import os
import logging
import numpy as np
import argparse
import time
import cv2
import csv
import threading
import datetime
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Flask 앱 초기화
app = Flask(__name__)

class DualCameraDataCollector:
    """
    두 카메라의 손 좌표를 동시에 기록하는 데이터 수집기
    """
    def __init__(self, output_dir="collected_data"):
        """
        초기화
        
        Args:
            output_dir: 데이터 저장 디렉토리
        """
        self.output_dir = output_dir
        self.camera1_id = 0  # 첫 번째 카메라 ID 기본값
        self.camera2_id = 2  # 두 번째 카메라 ID 기본값
        self.width = 640
        self.height = 480
        self.running = False
        self.recording = False
        
        # 카메라 관련 변수
        self.cap1 = None
        self.cap2 = None
        self.current_frame1 = None
        self.current_frame2 = None
        
        # 손 감지 관련 변수
        self.tip1_x = None
        self.tip1_y = None
        self.tip2_x = None
        self.tip2_y = None
        
        # 기록 관련 변수
        self.csv_writer = None
        self.csv_file = None
        self.frame_count = 0
        self.record_count = 0
        
        # 두 개의 손 인식 모델 초기화
        self.hands1 = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            max_num_hands=1
        )
        
        self.hands2 = mp_hands.Hands(
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
            "samples": 0,
            "cam1_hand_detected": False,
            "cam2_hand_detected": False,
            "cam1_tip_coords": {"x": None, "y": None},
            "cam2_tip_coords": {"x": None, "y": None}
        }
    
    def start_cameras(self):
        """두 카메라 스트림 시작"""
        # 첫 번째 카메라 초기화
        logger.info(f"Opening camera 1 with ID {self.camera1_id}")
        
        if os.name == 'posix':  # Linux/Mac
            self.cap1 = cv2.VideoCapture(self.camera1_id, cv2.CAP_V4L2)
        else:  # Windows 또는 기타
            self.cap1 = cv2.VideoCapture(self.camera1_id, cv2.CAP_DSHOW)
        
        # 첫 번째 방법이 실패하면 기본 API 시도
        if not self.cap1.isOpened():
            logger.warning(f"Failed to open camera 1 with specific API, trying default")
            self.cap1.release()
            self.cap1 = cv2.VideoCapture(self.camera1_id)
        
        # 그래도 실패하면 에러 반환
        if not self.cap1.isOpened():
            logger.error(f"Cannot open camera 1 with ID {self.camera1_id}")
            return False
        
        # 카메라 설정
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 테스트 프레임 읽기
        ret, test_frame = self.cap1.read()
        if not ret or test_frame is None:
            logger.error("Failed to read test frame from camera 1")
            self.cap1.release()
            self.cap1 = None
            return False
        
        logger.info(f"Camera 1 opened successfully: {self.width}x{self.height}")
        
        # 두 번째 카메라 초기화
        logger.info(f"Opening camera 2 with ID {self.camera2_id}")
        
        if os.name == 'posix':  # Linux/Mac
            self.cap2 = cv2.VideoCapture(self.camera2_id, cv2.CAP_V4L2)
        else:  # Windows 또는 기타
            self.cap2 = cv2.VideoCapture(self.camera2_id, cv2.CAP_DSHOW)
        
        # 첫 번째 방법이 실패하면 기본 API 시도
        if not self.cap2.isOpened():
            logger.warning(f"Failed to open camera 2 with specific API, trying default")
            self.cap2.release()
            self.cap2 = cv2.VideoCapture(self.camera2_id)
        
        # 두 번째 카메라가 실패해도 첫 번째 카메라만으로 계속 진행
        if not self.cap2.isOpened():
            logger.warning(f"Cannot open camera 2 with ID {self.camera2_id}, continuing with one camera")
            self.cap2 = None
        else:
            # 카메라 설정
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # 테스트 프레임 읽기
            ret, test_frame = self.cap2.read()
            if not ret or test_frame is None:
                logger.warning("Failed to read test frame from camera 2, continuing with one camera")
                if self.cap2:
                    self.cap2.release()
                self.cap2 = None
            else:
                logger.info(f"Camera 2 opened successfully: {self.width}x{self.height}")
        
        # 적어도 하나의 카메라가 열렸으면 성공
        return self.cap1 is not None
    
    def start_recording(self):
        """데이터 기록 시작"""
        if self.recording:
            logger.warning("Recording is already in progress")
            return False
        
        try:
            # 파일명 생성 (날짜_시간.csv)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"hand_data_{timestamp}.csv")
            
            # CSV 파일 열기
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 헤더 작성 (두 카메라의 손 좌표 포함)
            header = [
                "frame", "timestamp", 
                "cam1_hand_detected", "cam1_tip_x", "cam1_tip_y",
                "cam2_hand_detected", "cam2_tip_x", "cam2_tip_y"
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
    
    def process_frames(self):
        """두 카메라의 프레임 처리 및 데이터 기록"""
        # 첫 번째 카메라 프레임 처리
        frame1 = None
        cam1_hand_detected = False
        if self.cap1 and self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if ret:
                frame1 = self.process_hand_detection(frame, 0)
                cam1_hand_detected = self.status_info["cam1_hand_detected"]
        
        # 두 번째 카메라 프레임 처리
        frame2 = None
        cam2_hand_detected = False
        if self.cap2 and self.cap2.isOpened():
            ret, frame = self.cap2.read()
            if ret:
                frame2 = self.process_hand_detection(frame, 2)
                cam2_hand_detected = self.status_info["cam2_hand_detected"]
        
        # 프레임 카운트 및 타임스탬프
        self.frame_count += 1
        timestamp = time.time()
        
        # 데이터 기록 (recording 모드인 경우)
        if self.recording and self.csv_writer:
            row = [self.frame_count, timestamp]
            
            # 첫 번째 카메라 손 좌표 추가
            if cam1_hand_detected:
                row.extend([True, self.tip1_x, self.tip1_y])
            else:
                row.extend([False, None, None])
            
            # 두 번째 카메라 손 좌표 추가
            if cam2_hand_detected:
                row.extend([True, self.tip2_x, self.tip2_y])
            else:
                row.extend([False, None, None])
            
            # CSV에 기록
            self.csv_writer.writerow(row)
            self.record_count += 1
            self.status_info["samples"] = self.record_count
        
        # 현재 프레임 저장
        self.current_frame1 = frame1
        self.current_frame2 = frame2
        
        return frame1, frame2
        
    def process_hand_detection(self, frame, camera_idx):
        """
        카메라 프레임에서 손 감지 및 좌표 추출
        
        Args:
            frame: 처리할 프레임
            camera_idx: 카메라 인덱스 (0 또는 2)
        
        Returns:
            처리된 프레임
        """
        if frame is None:
            return None
        
        # OpenCV는 BGR 형식, MediaPipe는 RGB 형식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 손 감지
        if camera_idx == 0:
            results = self.hands1.process(frame_rgb)
        else:
            results = self.hands2.process(frame_rgb)
        
        # 프레임 수정 가능하게 변경
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
                    f"Tip: x={tip_x}, y={tip_y}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # 검지 끝 위치에 원 그리기
                cv2.circle(
                    frame,
                    (tip_x, tip_y),
                    10,
                    (0, 0, 255),
                    -1
                )
        
        # 카메라 번호 표시
        cv2.putText(
            frame,
            f"CAM {camera_idx}",
            (self.width - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 165, 0),  # 오렌지색
            2
        )
        
        # 녹화 상태 표시
        recording_status = "RECORDING" if self.recording else "NOT RECORDING"
        cv2.putText(
            frame,
            recording_status,
            (self.width - 200, 60),
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
                (self.width - 200, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return frame
    
    def generate_frames(self, camera_idx=0):
        """비디오 스트림 생성을 위한 제너레이터 함수"""
        while self.running:
            frame = None
            if camera_idx == 0:
                frame = self.current_frame1
            else:
                frame = self.current_frame2
                
            if frame is not None:
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                # HTTP 응답 메시지 구성
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # 약 30fps 제한
    
    def start(self):
        """데이터 수집 시작"""
        # 카메라 시작
        if not self.start_cameras():
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
                # 두 카메라 프레임 처리
                self.process_frames()
                
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
        
        try:
            # 파일명 생성 (날짜_시간_snapshot.csv)
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"hand_data_snapshot_{timestamp_str}.csv")
            
            # 현재 타임스탬프
            timestamp = time.time()
            
            # CSV 파일 열기
            with open(filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # 헤더 작성
                header = [
                    "frame", "timestamp", 
                    "cam1_hand_detected", "cam1_tip_x", "cam1_tip_y",
                    "cam2_hand_detected", "cam2_tip_x", "cam2_tip_y"
                ]
                csv_writer.writerow(header)
                
                # 데이터 행 구성
                row = [self.frame_count, timestamp]
                
                # 첫 번째 카메라 손 좌표 추가
                cam1_hand_detected = self.status_info["cam1_hand_detected"]
                if cam1_hand_detected:
                    row.extend([True, self.tip1_x, self.tip1_y])
                else:
                    row.extend([False, None, None])
                
                # 두 번째 카메라 손 좌표 추가
                cam2_hand_detected = self.status_info["cam2_hand_detected"]
                if cam2_hand_detected:
                    row.extend([True, self.tip2_x, self.tip2_y])
                else:
                    row.extend([False, None, None])
                
                # CSV에 기록
                csv_writer.writerow(row)
            
            logger.info(f"Snapshot saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
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
        if hasattr(self, 'hands1'):
            self.hands1.close()
        
        if hasattr(self, 'hands2'):
            self.hands2.close()
        
        if hasattr(self, 'cap1') and self.cap1 and self.cap1.isOpened():
            self.cap1.release()
        
        if hasattr(self, 'cap2') and self.cap2 and self.cap2.isOpened():
            self.cap2.release()
        
        logger.info("Data collector stopped")


# 전역 객체
collector = None

# Flask 라우트 설정
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/video_feed1')
def video_feed1():
    """첫 번째 카메라 비디오 스트림 라우트"""
    if collector is None or not collector.running:
        # 더미 프레임 반환
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera 1 not started", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
    """두 번째 카메라 비디오 스트림 라우트"""
    if collector is None or not collector.running or collector.cap2 is None:
        # 더미 프레임 반환
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Camera 2 not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', dummy_frame)
        frame_bytes = buffer.tobytes()
        
        def generate_dummy():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        return Response(generate_dummy(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(collector.generate_frames(2), 
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

def create_app(output_dir='collected_data', camera1_id=0, camera2_id=2):
    """Flask 앱 초기화"""
    global collector
    
    try:
        # 데이터 수집기 초기화 및 시작
        logger.info("Starting data collector...")
        collector = DualCameraDataCollector(output_dir)
        collector.camera1_id = camera1_id
        collector.camera2_id = camera2_id
        collector.start()
        
        # 사용 설명
        print("\n=== Hand Tracking Data Collection Web Interface (Dual Camera) ===")
        print(f"- Saving data to: {output_dir}")
        print(f"- Using camera 1 ID: {camera1_id}")
        print(f"- Using camera 2 ID: {camera2_id}")
        print("\nOpen your browser at http://localhost:5000 to view and control the data collection.")
        print("Make sure to position your hand so the cameras can see it.\n")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        
    return app

def shutdown_app():
    """앱 종료 시 리소스 정리"""
    global collector
    
    if collector:
        collector.stop()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Hand Data Collector Web Interface (Dual Camera)')
    parser.add_argument('--output', type=str, default='collected_data',
                        help='Directory to save collected data (default: collected_data)')
    parser.add_argument('--camera1', type=int, default=0,
                        help='Camera 1 ID to use (default: 0)')
    parser.add_argument('--camera2', type=int, default=2,
                        help='Camera 2 ID to use (default: 2)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on (default: 0.0.0.0, accessible from any network interface)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    
    try:
        # 앱 생성 및 실행
        app_instance = create_app(args.output, args.camera1, args.camera2)
        app_instance.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nStopping by user request...")
    finally:
        # 리소스 정리
        shutdown_app()

if __name__ == "__main__":
    main()