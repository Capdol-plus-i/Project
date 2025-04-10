import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from flask import Flask, Response, render_template_string

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 전역 변수
width, height = 640, 480
output_frame = None
lock = threading.Lock()

app = Flask(__name__)

def detect_hands():
    global output_frame, lock
    
    # 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # MediaPipe Hands 모델 생성
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2)
    
    print("핸드 포즈 감지 스트림 시작")
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라에서 프레임을 받아오지 못했습니다.")
                continue
            
            # 성능 향상을 위해 이미지 쓰기 불가능하게 설정
            image.flags.writeable = False
            
            # 색상 변환 (OpenCV는 BGR, MediaPipe는 RGB 사용)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 손 감지 처리
            results = hands.process(image_rgb)
            
            # 그리기 위해 이미지 쓰기 가능하게 설정
            image.flags.writeable = True
            
            # 감지된 손이 있는 경우
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 손의 종류 확인 (왼손/오른손)
                    hand_type = "right"
                    if results.multi_handedness and len(results.multi_handedness) > idx:
                        hand_info = results.multi_handedness[idx]
                        if hand_info.classification[0].label == "Left":
                            hand_type = "left"
                    
                    # 손의 색상 설정 (왼손: 초록색, 오른손: 빨간색)
                    color = (0, 255, 0) if hand_type == "left" else (0, 0, 255)
                    
                    # 손의 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # 손 종류 텍스트 표시
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
                    cv2.putText(
                        image, 
                        hand_type, 
                        (wrist_x - 30, wrist_y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2)
                    
                    # 각 키포인트에 점 추가
                    #for landmark in hand_landmarks.landmark:
                    #    x, y = int(landmark.x * width), int(landmark.y * height)
                    #    cv2.circle(image, (x, y), 5, color, -1)
            
            # 상태 표시 텍스트 추가
            cv2.putText(
                image,
                f"Hand Tracking - Size: {width}x{height}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # 프레임 잠금 및 업데이트
            with lock:
                output_frame = image.copy()
            
            # 과도한 CPU 사용 방지
            time.sleep(0.05)
    
    finally:
        hands.close()
        cap.release()
        print("핸드 포즈 감지 종료")

def generate_frames():
    global output_frame, lock
    
    while True:
        # 프레임 확인
        with lock:
            if output_frame is None:
                continue
            
            # JPEG로 인코딩
            _, encoded_image = cv2.imencode(".jpg", output_frame)
            
            if not _:
                continue
                
        # 바이트 문자열 반환
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        
        # 과도한 CPU 사용 방지
        time.sleep(0.05)

@app.route("/")
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>핸드 포즈 감지 스트림</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
                background-color: #f0f0f0;
            }
            h1 {
                color: #333;
            }
            .video-container {
                margin: 20px auto;
                max-width: 1280px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            #video-stream {
                width: 100%;
                height: auto;
                border: 1px solid #ddd;
                background-color: #000;
            }
            .info {
                margin: 20px;
                padding: 15px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        <h1>핸드 포즈 감지 스트림</h1>
        <div class="video-container">
            <img id="video-stream" src="/video_feed" alt="Hand Tracking Stream">
        </div>
        <div class="info">
            <h2>사용 방법</h2>
            <p>카메라 앞에서 손을 움직이면 왼손(초록색)과 오른손(빨간색)이 감지됩니다.</p>
            <p>각 손가락 관절과 손목 위치가 표시됩니다.</p>
        </div>
    </body>
    </html>
    ''')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # 핸드 포즈 감지 스레드 시작
    t = threading.Thread(target=detect_hands)
    t.daemon = True
    t.start()
    
    # 웹 서버 시작 (0.0.0.0으로 설정하여 외부에서도 접근 가능)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)