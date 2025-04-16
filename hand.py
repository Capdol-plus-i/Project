import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
width, height = 640, 480
cap = cv2.VideoCapture(0)
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
        '''
        우선 multi_hand_landmarks는 영상에서 인식한 손에서 각 landmarks(검지 끝, 중지 중간 등등..)들의 좌표를 담고있다.
        multi_handedness는 그 인식한 손이 왼손, 오른손인지 라벨과 그 신뢰도 정보를 담고있다.
        ->mediapipe/mediapipe/python/solution/hands.py 참조
        자 그러면 양손 모두 영상에 등장하였을 때 이 새키들은 어떻게 정보를 저장해놓았을까?
        답은 생각보다 간단하다. 영상에 등장한 손만큼 multi_hand_landmarks와 multi_handedness의 배열 크기가 증가하고
        각각의 인덱싱 번호로 짝지어져있기에 multi_hand_landmarks[0]과 multi_handedness[0]은 같은 손의 정보를 담고있다는것이다.
        만약 양손 모두 영상에 등장하면 multi_hand_landmarks[0]과 multi_hand_landmarks[1]이 존재하게 되고 각각 같은 인덱싱 번호를 갖는
        multi_handedness[0], multi_hand_landmarks[1]와 묶여 각 손의 정보를 담고있다.
        여기까지 알았으면 우리가 영상에 등장한 손의 종류와 그 좌표를 알아내는것은 쉬울것이다. 

        우선 바로 이 주석문 바로 위 for문을 보면 zip(results.multi_hand_landmarks, results.multi_handedness)가 보일텐데 이는 앞 hand_landmarks 와 handedness에
        results.multi_hand_landmarks[0] ~ [1]까지 (양손이 등장했다면 1까지 아니라면 0만..) 와 results.multi_handedness[0] ~ [1] 을 차례대로 대입해가며 
        for문 안의 코드를 동작 시킨다는 것이다. 그러니까 만약 양손이 등장했다면 인덱스 0에 해당하는 hand_landmarks 와 handedness로 아래 코드를 수행하고, 이후
        인덱스 1에 해당하는 hand_landmarks 와 handedness로 코드를 동작시킨다는것!
        그러면 첫번째 인덱스 0에 해당하는 손의 정보 hand_landmarks 와 handedness를 가지고 아래 코드를 수행해보자
        '''
        hand_type = handedness.classification[0].label
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        tip_x = int(index_finger_tip.x * width)
        tip_y = int(index_finger_tip.y * height)
        tip_z = index_finger_tip.z
        print(f"{hand_type}hand tip coordinate: x={tip_x}, y={tip_y}, z={tip_z:.3f}")
          
        '''
        자 첫번째 줄  hand_type = handedness.classification[0].label을 보고 살짝 어지러웠을 것인데
        갑자기 등장하는 저 classification[0]은 뭐냐하면 학습 모델이 추론한 여러가지 가능성중 가장 신뢰성이 높은 결과를 보겠다 이 의미다.
        결국 영상을 바탕으로 모델이 추론을 하는것이기 떄문에 그 결과는 여러가지 경우의 수가 있을 수 있는데, 그중 신뢰성이 가장 높은 경우가 classification[0]이기에
        저 classification[0]의 label 즉, 너가 가장 신뢰성이 높다고 추론한 결과속 손의 왼손 오른손 여부가 무엇인지를 보는것이고 그 결과를 hand_type에 저장한다.
        
        그리고 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 이 코드를 설명하자면 아래 사이트를 먼저 보자
        https://chuoling.github.io/mediapipe/solutions/hands.html 이 사이트를 보면 각 손의 관절 위치마다 인덱싱이 되어있는데 
        hand_landmarks.landmark[ 여기에 위치정보를 알고싶은 관절의 인덱스를 넣어서 ] 그 인덱스의 해당하는 손 관절 위치의 위치정보 묶음 (x,y,z)를 가져와 
        index_finger_tip 변수에 저장하는 것이다. 참고로 사이트에서 보았듯 검지 끝의 인덱스는 8이기에 mp_hands.HandLandmark.INDEX_FINGER_TIP 대신
        8을 입력해도 잘 동작한다. 이제 데이터 가져왔으니 이제 뭐해? 잘 나눠서 출력해야지? 그래서~
        
        tip_x = int(index_finger_tip.x * width)
        tip_y = int(index_finger_tip.y * height)
        tip_z = index_finger_tip.z
        이 코드를 거쳐 각각 tip_x,y,z에 위치 정보를 저장하고
        
        print(f"{hand_type}hand tip coordinate: x={tip_x}, y={tip_y}, z={tip_z:.3f}")
        최종적으로 우리가 판단한 왼손 오른손 여부와 그 손의 왼손 검지의 위치정보를 출력하는 것이다.
        '''
        # 손의 색상 설정 (왼손: 초록색, 오른손: 빨간색)
        color = (0, 255, 0) if hand_type == "left" else (0, 0, 255)

        # 손 종류 텍스트 표시
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
        #print(wrist)
        
        cv2.putText(
            image, 
            hand_type, 
            (wrist_x - 30, wrist_y - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2)

        # 손의 랜드마크 그리기
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        
        
        # 각 키포인트에 점 추가
        #for landmark in hand_landmarks.landmark:
        #    x, y = int(landmark.x * width), int(landmark.y * height)
        #    cv2.circle(image, (x, y), 5, color, -1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
