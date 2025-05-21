from dynamixel_sdk import *
import cv2
import mediapipe as mp
import numpy as np

# dynamixel init
#=================================================================================================================
MY_DXL = 'XL330-M077'
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_POSITION       = 132
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the CW Angle Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the CCW Angle Limit of product eManual
BAUDRATE                    = 1000000   # Default Baudrate of XL-320 is 1Mbps
PROTOCOL_VERSION            = 2.0
DXL_ID                      = [1, 2, 3, 4]
DEVICENAME                  = 'COM4'
TORQUE_ENABLE               = 1     # Value for enabling the torque
TORQUE_DISABLE              = 0     # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold

index = 0
# dxl_goal_position = [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE]         # Goal position
dxl_goal_position = 3048        # Goal position

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    quit()


# Enable Dynamixel Torque
for i in range(4) :
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print(f"Dynamixel id : {i + 1} has been successfully connected")

#=================================================================================================================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
width, height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 1) 파라미터 로드
param_data = np.load("model_parameters_resnet.npz")
parameters = {k: param_data[k] for k in param_data.files}

# 2) 순전파 함수 (ReLU 은닉, 선형 출력)
def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    # 은닉층 (ReLU)
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = W.dot(A) + b
        A = np.maximum(0, Z)
    # 출력층 (선형)
    Wl = parameters[f'W{L}']
    bl = parameters[f'b{L}']
    ZL = Wl.dot(A) + bl
    return ZL


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
        x_new = np.array([[tip_x],
                  [tip_y],
                  [100.0]], dtype=np.float64) / 700.0   # 반드시 이전에 사용한 스케일 복원도 적용
        
        AL = L_model_forward(x_new, parameters)    # shape: (4, m)
        pred = (AL * 4100).astype(int)      # 일단 int로
        print(f"{hand_type}hand tip coordinate: x={tip_x}, y={tip_y}, z={tip_z:.3f}")
        print(f"pred value = {pred}")
        # for i in range(4):  
        #     dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_GOAL_POSITION, pred[i].item())
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
        
        '''print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) 양손 검지 끝 좌표
        print(results.multi_handedness) #-> right = 1 left = 0'''
        # 각 키포인트에 점 추가
        for landmark in hand_landmarks.landmark:
           x, y = int(landmark.x * width), int(landmark.y * height)
           cv2.circle(image, (x, y), 5, color, -1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()