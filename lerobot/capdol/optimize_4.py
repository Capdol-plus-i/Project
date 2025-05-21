#!/usr/bin/env python3
import os
import json
import time
import threading
import logging
import numpy as np
import cv2
import argparse
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CALIBRATION_DIR = "lerobot/capdol/calibration_data"
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ManipulatorRobot:
    def __init__(self, config):
        self.arms = {
            'follower': make_motors_buses_from_configs(config.follower_arms)
        }
        self.is_connected = False
        self.calibrated = False
        try:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            self.TorqueMode = TorqueMode
        except ImportError:
            self.TorqueMode = None

    def connect(self):
        if self.is_connected:
            return
        for arm_type, buses in self.arms.items():
            for name, bus in buses.items():
                logger.info(f"Connecting {arm_type} arm '{name}'")
                bus.connect()
        self.is_connected = True
        logger.info("Robot connected")
        self._load_calibration()

    def _load_calibration(self):
        def load(file_name):
            path = os.path.join(CALIBRATION_DIR, file_name)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    logger.info(f"Loaded calibration from {path}")
                    return data
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
            else:
                logger.warning(f"Calibration file missing: {path}")
            return None
        follower_calib = load('follower_arm.json')
        if follower_calib:
            for bus in self.arms['follower'].values():
                bus.set_calibration(follower_calib)
        self.calibrated = bool(follower_calib)
        status = "OK" if self.calibrated else "INCOMPLETE"
        logger.info(f"Calibration status: {status}")

    def setup_control(self, arm_type='follower'):
        if not self.is_connected:
            logger.error("Robot not connected")
            return False
        for bus in self.arms[arm_type].values():
            if self.TorqueMode:
                bus.write('Torque_Enable', self.TorqueMode.DISABLED.value)
            bus.write('Operating_Mode', 3)
            bus.write('Torque_Enable', 1)
        logger.info(f"{arm_type.capitalize()} control setup done")
        return True

    def move(self, positions, arm_type='follower'):
        if not self.is_connected or len(positions) != 4:
            logger.error("Invalid move")
            return False
        idx = 0
        for bus in self.arms[arm_type].values():
            count = len(bus.motor_names)
            current = np.array(bus.read('Present_Position'), dtype=np.float32)
            delta = positions[idx:idx+count] - current
            delta = np.clip(delta, -150.0, 150.0)
            target = np.round(current + delta).astype(np.int32)
            idx += count
            try:
                bus.write('Goal_Position', target)
            except Exception as e:
                logger.error(f"Move error: {e}")
                return False
        return True

    def disconnect(self):
        if not self.is_connected:
            return
        for buses in self.arms.values():
            for bus in buses.values():
                bus.disconnect()
        self.is_connected = False
        logger.info("Robot disconnected")

class DualCameraHandRobotController:
    def __init__(self, robot, model_path, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.model_path = model_path
        self.cams = [None, None]
        self.tip = [(0,0), (0,0)]
        self.hand_detected = [False, False]
        self.z = 10
        self.width, self.height = 640, 480
        self.running = False
        self._load_model()
        self.hands = [mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4,
                                     min_tracking_confidence=0.3, max_num_hands=1) for _ in range(2)]

    def _load_model(self):
        try:
            params = np.load(self.model_path)
            self.params = {k: params[k] for k in params.files}
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            self.params = None

    def _predict(self, x, y, z):
        if not self.params:
            return np.zeros(4)
        A = np.array([[x],[y],[z]])/700.0
        L = len(self.params)//2
        for l in range(1, L):
            A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
        return ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 400).flatten()

    def start(self, cam_ids=(4,0)):
        if not (self.robot.setup_control(self.arm_type) and self._open_cams(cam_ids)):
            return False
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _open_cams(self, cam_ids):
        for i, cid in enumerate(cam_ids):
            cap = cv2.VideoCapture(cid)
            if not cap.isOpened():
                logger.error(f"Camera {i+1} open fail")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cams[i] = cap
            logger.info(f"Camera {i+1} ready")
        return True

    def _process(self, i):
        ret, frame = self.cams[i].read()
        if not ret:
            return np.zeros((self.height,self.width,3), np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands[i].process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark[8]
            x, y = int(lm.x*self.width), int(lm.y*self.height)
            self.tip[i], self.hand_detected[i] = (x,y), True
            if i == 1:
                self.z = y
            cv2.circle(frame, (x,y), 10, (0,0,255), -1)
        return frame

    def _loop(self):
        logger.info("Control loop start")
        while self.running:
            for i in (0,1):
                self._process(i)
            if self.hand_detected[0]:
                joints = self._predict(*self.tip[0], self.z).round().astype(int)
                if joints.sum():
                    self.robot.move(joints, self.arm_type)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
        self.stop()

    def stop(self):
        self.running = False
        for h in self.hands:
            h.close()
        for c in self.cams:
            if c:
                c.release()
        cv2.destroyAllWindows()
        logger.info("Controller stopped")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['follower'], default='follower')
    p.add_argument('--model', default='lerobot/capdol/models/model_parameters_resnet.npz')
    p.add_argument('--camera1', type=int, default=4)
    p.add_argument('--camera2', type=int, default=0)
    args = p.parse_args()
    robot = ManipulatorRobot(KochRobotConfig())
    robot.connect()
    controller = DualCameraHandRobotController(robot, args.model, args.arm)
    if controller.start((args.camera1, args.camera2)):
        logger.info("=== Dual Camera Hand Robot ===")
        logger.info(f"Arm: {args.arm}, Calibrated: {robot.calibrated}")
        try:
            while controller.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    controller.stop()
    robot.disconnect()