import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
width, height = 640, 480
cap = cv2.VideoCapture(8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks and results.multi_handedness:
      for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        
        hand_type = handedness.classification[0].label
        index_finger_tip = hand_landmarks.landmark[8]
        tip_x = int(index_finger_tip.x * width)
        tip_y = int(index_finger_tip.y * height)
        tip_z = index_finger_tip.z
        print(f"{hand_type}hand tip coordinate: x={tip_x}, y={tip_y}, z={tip_z:.3f}")
        
        # 손의 색상 설정 (왼손: 초록색, 오른손: 빨간색)
        color = (0, 255, 0) if hand_type == "left" else (0, 0, 255)

        # 손 종류 텍스트 표시
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
        #print(wrist)
        
        # cv2.putText(
        #     image, 
        #     hand_type, 
        #     (wrist_x - 30, wrist_y - 30), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1, 
        #     (255, 255, 255), 
        #     2)
        
        '''print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) 양손 검지 끝 좌표
        print(results.multi_handedness) #-> right = 1 left = 0'''

        # 손의 랜드마크 그리기
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
        
        
        
        # 각 키포인트에 점 추가
        #for landmark in hand_landmarks.landmark:
        #    x, y = int(landmark.x * width), int(landmark.y * height)
        #    cv2.circle(image, (x, y), 5, color, -1)
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break
cap.release()

