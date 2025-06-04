from dynamixel_sdk import *
import cv2
import mediapipe as mp
import numpy as np
import threading
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
        self.prev_goal_position = np.array([3000, 2470, 1530, 1530]) 
        self.last_goal_position = np.array([3000, 2470, 1530, 1530]) 
        self.default_position = np.array([3000, 2470, 1530, 1530])# default goal_position
        self.P_GAIN = [200, 200, 200, 700]

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
                self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_POSITION_P_GAIN, self.P_GAIN[i])
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            quit()
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            self.ADDR_GOAL_POSITION,
            4  # 4 bytes
        )
    def set_default_position(self):
        self.write_goal_pos(self.default_position)
        self.last_goal_position = self.default_position
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
    # def write_goal_pos(self, goal_pos):
    #     for i in range(4) :
    #         dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_GOAL_POSITION, goal_pos[i].item())
    def write_goal_pos(self, goal_pos):
        self.groupSyncWrite.clearParam()
        for dxl_id, val in zip(self.DXL_ID, goal_pos):
            # little-endian 4바이트로 변환
            byte_array = list(int(val).to_bytes(4, byteorder='little'))
            self.groupSyncWrite.addParam(dxl_id, bytearray(byte_array))
        self.groupSyncWrite.txPacket()

    def read_goal_pos(self):
        present_goal_pos = [0, 0, 0, 0]
        for i in range(4) :
            present_goal_pos[i] = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID[i], self.ADDR_PRESENT_POSITION)    
        return present_goal_pos
#=================================================================================================================
class REGRESSION_MODEL:
    def __init__(self):
        #1) 파라미터 로드
        self.param_data = np.load("models/model_parameters_resnet.npz")
        self.parameters = {k: self.param_data[k] for k in self.param_data.files}
        self.TIP_MIN_THRESH = -10
        self.TIP_MAX_THRESH = 10
        self.prev_tip_x = 0
        self.prev_tip_y = 0
        self.prev_tip_z = 0
        self.alpha = 0.2
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
    def clip(self, tip_x, tip_y, tip_z) :
        if tip_x < 68 :
            tip_x = 68
        elif tip_x > 610 :
            tip_x = 610
        if tip_y < 29 :
            tip_y = 29
        elif tip_y > 337 :
            tip_y = 337
        if tip_z < 41 :
            tip_z = 41
        elif tip_z > 445 :
            tip_z = 445

        # tip_x = self.alpha * tip_x + (1-self.alpha) * self.prev_tip_x
        # tip_y = self.alpha * tip_y + (1-self.alpha) * self.prev_tip_y
        # tip_z = self.alpha * tip_z + (1-self.alpha) * self.prev_tip_z

        return tip_x, tip_y, tip_z


#=================================================================================================================
M077 = DXL_MOTOR()
M077.init()
M077.torque_enable()
Model = REGRESSION_MODEL()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

timer = None
TIMER_FLAG = False
# For webcam input:
width, height = 640, 480
target_fps = 60

cap_1 = cv2.VideoCapture(0)
cap_1.set(cv2.CAP_PROP_FPS, target_fps)
cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_2 = cv2.VideoCapture(2)
cap_2.set(cv2.CAP_PROP_FPS, target_fps)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)





with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    static_image_mode = False) as hands:
  while cap_1.isOpened() and cap_2.isOpened():
    success_1, image_1 = cap_1.read()
    success_2, image_2 = cap_2.read()
    if not (success_1 and success_2) :
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
    if results_1.multi_hand_landmarks and results_1.multi_handedness and results_2.multi_hand_landmarks and results_2.multi_handedness:
      for cam1_hand_landmarks, cam1_handedness, cam2_hand_landmarks, cam2_handedness in zip(results_1.multi_hand_landmarks, results_1.multi_handedness, results_2.multi_hand_landmarks, results_2.multi_handedness):
        if len(results_1.multi_hand_landmarks)>1 and cam1_handedness.classification[0].label != 'Right':
            continue
        hand_type = cam1_handedness.classification[0].label        
        cam1_index_finger_tip = cam1_hand_landmarks.landmark[8]
        cam2_index_finger_tip = cam2_hand_landmarks.landmark[8]
        tip_x = int(cam1_index_finger_tip.x * width)
        tip_y = int(cam1_index_finger_tip.y * height)
        tip_z = int(cam2_index_finger_tip.y * height)
        
        if TIMER_FLAG and timer :
            timer.cancel()
            TIMER_FLAG = False
        tip_x, tip_y, tip_z = Model.clip(tip_x, tip_y, tip_z)
        
        x_new = np.array([[tip_x],
                [tip_y],
                [tip_z]], dtype=np.float64) / 650.0   # 반드시 이전에 사용한 스케일 복원도 적용
        
        AL = Model.L_model_forward(x_new)    # shape: (4, m)
        pred = (AL * 4100).astype(int)      # 일단 int
        #print(f"{hand_type}hand tip coordinate: x={tip_x}, y={tip_y}, z={tip_z:.3f}")
        #print(f"pred value = {pred}")
        M077.last_goal_position = pred
        #goal_pos = pred
        goal_pos = (0.8 * M077.prev_goal_position) + (0.2 * M077.last_goal_position)
        goal_pos = goal_pos.flatten()
        goal_pos = goal_pos.astype(int)
        goal_pos = np.clip(goal_pos, 1, 4095)
            
        Model.prev_tip_x = tip_x
        Model.prev_tip_y = tip_y
        Model.prev_tip_z = tip_z

        M077.write_goal_pos(goal_pos)
        M077.prev_goal_position = goal_pos
        #손의 색상 설정 (왼손: 초록색, 오른손: 빨간색)

        # color = (0, 255, 0) if hand_type == "left" else (0, 0, 255)

        # 손 종류 텍스트 표시
        # cam1_wrist = cam1_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        # wrist_x, wrist_y = int(cam1_wrist.x * width), int(cam1_wrist.y * height)
            
        # cv2.putText(
        #     image_1, 
        #     hand_type, 
        #     (wrist_x - 30, wrist_y - 30), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1, 
        #     (255, 255, 255), 
        # 2)
        
        '''print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) 양손 검지 끝 좌표
        print(results.multi_handedness) #-> right = 1 left = 0'''

        # # 각 키포인트에 점 추가
        # for landmark in cam1_hand_landmarks.landmark:
        #     x, y = int(landmark.x * width), int(landmark.y * height)
        #     cv2.circle(image_1, (x, y), 5, color, -1)
        # # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Hands', cv2.flip(image_1, 1))
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
    else:
        #손이 인식되지 않았을 경우엔 기본 위치로 이동
        if not TIMER_FLAG :
            timer = threading.Timer(3, M077.set_default_position)
            timer.start()
            TIMER_FLAG = True
        M077.write_goal_pos(M077.last_goal_position)

cap_1.release()
cap_2.release()