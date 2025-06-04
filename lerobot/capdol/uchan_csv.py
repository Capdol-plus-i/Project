from dynamixel_sdk import *
import cv2
import mediapipe as mp
import numpy as np
import os, sys
import csv

sys.path.append("..\DynamixelSDK\python\src\dynamixel_sdk\*")
ESC_ASCII_VALUE             = 0x1b
CSV_FILENAME = "lerobot/capdol/robot_snapshots.csv"
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_y", 
               "joint_1", "joint_2", "joint_3", "joint_4"]

# CSV Management Functions
def init_csv_file():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)

def save_snapshot_to_csv(data):
    """Save snapshot data to CSV file"""
    try:
        # Write to CSV
        with open(CSV_FILENAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
        
        return True
            
    except Exception as e:
        return False, 0
    
# dynamixel init
#=================================================================================================================
# MY_DXL = 'XL330-M077'
# ADDR_TORQUE_ENABLE          = 64
# ADDR_GOAL_POSITION          = 116
# ADDR_PRESENT_POSITION       = 132
# ADDR_POSITION_P_GAIN        = 84
# DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the CW Angle Limit of product eManual
# DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the CCW Angle Limit of product eManual
# BAUDRATE                    = 1000000   # Default Baudrate of XL-320 is 1Mbps
# PROTOCOL_VERSION            = 2.0
# DXL_ID                      = [1, 2, 3, 4]
# DEVICENAME                  = '/dev/ttyACM0'
# TORQUE_ENABLE               = 1     # Value for enabling the torque
# TORQUE_DISABLE              = 0     # Value for disabling the torque
# DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold
#=================================================================================================================
class DXL_MOTOR:
    def __init__(self):
        self.ADDR_TORQUE_ENABLE          = 64
        self.ADDR_GOAL_POSITION          = 116
        self.ADDR_PRESENT_POSITION       = 132
        self.ADDR_POSITION_P_GAIN        = 84
        self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the CW Angle Limit of product eManual
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the CCW Angle Limit of product eManual
        self.BAUDRATE                    = 1000000   # Default Baudrate of XL-320 is 1Mbps
        self.PROTOCOL_VERSION            = 2.0
        self.DXL_ID                      = [1, 2, 3, 4]
        self.DEVICENAME                  = '/dev/ttyACM0'
        self.TORQUE_ENABLE               = 1     # Value for enabling the torque
        self.TORQUE_DISABLE              = 0     # Value for disabling the torque
        self.DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold
        self.last_goal_position = np.array([3000,2470,1530,1530]) # default goal_position
    def init(self):
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
            for i in range(4):
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_POSITION_P_GAIN, 150)
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            quit()
        
    def torque_enable(self):
        for i in range(4) :
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel id : {i + 1} has been successfully connected")

    def torque_disable(self):
        for i in range(4) :
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel id : {i + 1} has been successfully connected")
    def write_goal_pos(self, goal_pos):
        for i in range(4) :
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_GOAL_POSITION, goal_pos[i].item())
    def read_goal_pos(self):
        present_goal_pos = [0, 0, 0, 0]
        for i in range(4) :
            present_goal_pos[i], __, __ = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_PRESENT_POSITION)    
        return present_goal_pos

#=================================================================================================================
class REGRESSION_MODEL:
    def __init__(self):
        #1) 파라미터 로드
        self.param_data = np.load("lerobot/capdol/models/model_parameters_resnet.npz")
        self.parameters = {k: self.param_data[k] for k in self.param_data.files}
    # 2) 순전파 함수 (ReLU 은닉, 선형 출력)
    def L_model_forward(self, X):
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
        return ZL

#=================================================================================================================
M077 = DXL_MOTOR()
M077.init()
M077.torque_disable()
Model = REGRESSION_MODEL()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For save as csv
init_csv_file()

# For webcam input:
width, height = 640, 480
cap_1 = cv2.VideoCapture(0)
cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_2 = cv2.VideoCapture(2)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

prev_tip_x = 0
prev_tip_y = 0
prev_tip_z = 0


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1) as hands:
  while cap_1.isOpened() and cap_2.isOpened():
    success_1, image_1 = cap_1.read()
    success_2, image_2 = cap_2.read()
    if not (success_1 & success_2) :
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image_1.flags.writeable = False
    image_2.flags.writeable = False
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    results_1 = hands.process(image_1)
    results_2 = hands.process(image_2)

    # Draw the hand annotations on the image.
    image_1.flags.writeable = True
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
    image_2.flags.writeable = True
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR)
    if results_1.multi_hand_landmarks and results_1.multi_handedness and results_2.multi_hand_landmarks and results_2.multi_handedness:
      for cam1_hand_landmarks, cam1_handedness, cam2_hand_landmarks, cam2_handedness in zip(results_1.multi_hand_landmarks, results_1.multi_handedness, results_2.multi_hand_landmarks, results_2.multi_handedness):
        
        hand_type = cam1_handedness.classification[0].label        
        cam1_index_finger_tip = cam1_hand_landmarks.landmark[8]
        cam2_index_finger_tip = cam2_hand_landmarks.landmark[8]
        tip_x = int(cam1_index_finger_tip.x * width)
        tip_y = int(cam1_index_finger_tip.y * height)
        tip_z = int(cam2_index_finger_tip.y * height)
        
      
        x_new = np.array([[tip_x],
                [tip_y],
                [100.0]], dtype=np.float64) / 700.0   # 반드시 이전에 사용한 스케일 복원도 적용
        
        pred = M077.read_goal_pos()
        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'): break
        if c == 's':
            data = [tip_x, tip_y, tip_z] + pred
            save_snapshot_to_csv(data)
            print("Snapshot saved to CSV.")
        #손의 색상 설정 (왼손: 초록색, 오른손: 빨간색)
        color = (0, 255, 0) if hand_type == "left" else (0, 0, 255)

        # 손 종류 텍스트 표시
        cam1_wrist = cam1_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        wrist_x, wrist_y = int(cam1_wrist.x * width), int(cam1_wrist.y * height)
            
        cv2.putText(
            image_1, 
            hand_type, 
            (wrist_x - 30, wrist_y - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
        2)
        
        '''print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) 양손 검지 끝 좌표
        print(results.multi_handedness) #-> right = 1 left = 0'''

        #각 키포인트에 점 추가
        for landmark in cam1_hand_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image_1, (x, y), 5, color, -1)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image_1, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    else:
        #손이 인식되지 않았을 경우엔 기본 위치로 이동
        M077.write_goal_pos(M077.last_goal_position)

cap_1.release()
cap_2.release()