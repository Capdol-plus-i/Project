#!/usr/bin/env python3
import logging
import numpy as np
import argparse
import time
import cv2
import threading
import mediapipe as mp
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

# Setup simplified logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe once
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            return
        
        # Import here to avoid dependency issues
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        
        # Connect and configure all arms with a single loop
        for arm_type, arms in [("follower", self.follower_arms), ("leader", self.leader_arms)]:
            for name in arms:
                logger.info(f"Connecting {name} {arm_type} arm")
                arms[name].connect()
                
        # Configure all motors
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                # Disable torque, set extended position mode, re-enable torque
                arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
                arms[name].write("Operating_Mode", 4)  # Extended Position Control Mode
                arms[name].write("Torque_Enable", 1)

        self.is_connected = True
        logger.info("Robot connected successfully")

    def send_action(self, action, arm_type='follower'):
        """Send joint position commands to robot arm"""
        if not self.is_connected:
            logger.error("Robot not connected")
            return False
            
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        
        # Process each motor group
        from_idx, to_idx = 0, 0
        for name in arms:
            motor_count = len(arms[name].motor_names)
            to_idx += motor_count
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            
            # Get current position
            present_pos = np.array(arms[name].read("Present_Position"), dtype=np.float32)
            
            # Convert goal position to numpy array
            if hasattr(goal_pos, 'numpy'):
                goal_pos = goal_pos.numpy()
            goal_pos = np.array(goal_pos, dtype=np.float32)
            
            # Safety limits: max change of 150 units
            safe_goal_pos = present_pos + np.clip(goal_pos - present_pos, -150.0, 150.0)
            safe_goal_pos = np.round(safe_goal_pos).astype(np.int32)
            
            # Send command
            try:
                arms[name].write("Goal_Position", safe_goal_pos)
                return True
            except Exception as e:
                logger.error(f"Error sending action: {e}")
                return False

    def move_to_joint_positions(self, joint_positions, arm_type='follower'):
        """Simple wrapper to move to specified joint positions"""
        if len(joint_positions) != 4:
            logger.error(f"Expected 4 joint positions, got {len(joint_positions)}")
            return False
        return self.send_action(np.array(joint_positions, dtype=np.float32), arm_type)

    def get_current_positions(self, arm_type='follower'):
        """Get current joint positions"""
        if not self.is_connected:
            logger.error("Robot not connected")
            return None
            
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        positions = []
        
        for name in arms:
            positions.extend(arms[name].read("Present_Position"))
            
        return np.array(positions)

    def disconnect(self):
        """Disconnect from robot"""
        if not self.is_connected:
            return
            
        logger.info("Disconnecting robot")
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False
        logger.info("Robot disconnected")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class DualCameraHandRobotController:
    """Controller for hand-based robot manipulation using two cameras"""
    def __init__(self, robot, model_parameters_path="model_parameters.npz", arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.camera1_id = 0
        self.camera2_id = 2
        self.width = 640
        self.height = 480
        self.running = False
        
        # Camera variables
        self.caps = [None, None]
        self.hand_detected = [False, False]
        self.tip_coords = [(0, 0), (0, 0)]  # [(x1, y1), (x2, y2)]
        self.display_frames = [None, None]
        
        # Initialize hand tracking models with optimized parameters
        self.hands_models = [
            mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, 
                         min_tracking_confidence=0.2, max_num_hands=1),
            mp_hands.Hands(model_complexity=0, min_detection_confidence=0.4, 
                         min_tracking_confidence=0.2, max_num_hands=1)
        ]
        
        # Neural network parameters
        try:
            self.load_model_parameters(model_parameters_path)
            logger.info(f"Neural network parameters loaded")
        except Exception as e:
            logger.error(f"Error loading neural network parameters: {e}")
            self.parameters = None
        
        # State variables
        self.predicted_joints = np.zeros(4)
        self.z_value = 10  # Default z value
    
    def load_model_parameters(self, model_path):
        """Load neural network parameters"""
        param_data = np.load(model_path)
        self.parameters = {k: param_data[k] for k in param_data.files}
    
    def predict_joints(self, x, y, z):
        """Predict joint positions from hand coordinates"""
        if self.parameters is None:
            return np.zeros(4)
        
        # Input vector
        X = np.array([[x], [y], [z]]) / 600.0
        
        # Forward pass
        A = X
        L = len(self.parameters) // 2
        
        # Hidden layers (ReLU)
        for l in range(1, L):
            W, b = self.parameters[f'W{l}'], self.parameters[f'b{l}']
            A = np.maximum(0, W.dot(A) + b)
            
        # Output layer (linear)
        W, b = self.parameters[f'W{L}'], self.parameters[f'b{L}']
        predictions = (W.dot(A) + b) * 4000
        
        return predictions.flatten()
    
    def start_cameras(self):
        """Initialize and start both cameras"""
        for i, cam_id in enumerate([self.camera1_id, self.camera2_id]):
            logger.info(f"Opening camera {i+1} with ID {cam_id}")
            cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.caps[i] = cap
                logger.info(f"Camera {i+1} opened successfully")
            else:
                logger.error(f"Failed to open camera {i+1}")
                if i == 0:  # First camera is required
                    return False
        
        return self.caps[0] is not None
    
    def process_frame(self, cam_idx):
        """Process a frame from either camera"""
        cap = self.caps[cam_idx]
        if cap is None or not cap.isOpened():
            # Create empty frame if camera unavailable
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {cam_idx+1} not available", 
                      (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 255), 2)
            self.display_frames[cam_idx] = frame
            return frame
            
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame from camera {cam_idx+1}")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {cam_idx+1} read error", 
                      (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 0, 0), 2)
            self.display_frames[cam_idx] = frame
            return frame
        
        # Store current coordinates
        old_x, old_y = self.tip_coords[cam_idx]
        self.hand_detected[cam_idx] = False
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_models[cam_idx].process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)
                self.tip_coords[cam_idx] = (x, y)
                self.hand_detected[cam_idx] = True
                
                # Update z value from camera 2
                if cam_idx == 1:
                    self.z_value = y
                
                # Draw finger position
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                
                # Text info
                info_text = f"x={x}, y={y}" if cam_idx == 0 else f"z={y}"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Keep previous values if no hand detected
            self.tip_coords[cam_idx] = (old_x, old_y)
        
        self.display_frames[cam_idx] = frame
        return frame
    
    def create_combined_display(self):
        """Create combined display with both camera views"""
        # Process frames if they don't exist
        if any(frame is None for frame in self.display_frames):
            for i in range(2):
                if self.display_frames[i] is None:
                    self.process_frame(i)
        
        # Create combined display
        combined = np.zeros((self.height, self.width*2 + 20, 3), dtype=np.uint8)
        
        # Add camera frames
        combined[0:self.height, 0:self.width] = self.display_frames[0]
        combined[0:self.height, self.width+20:] = self.display_frames[1]
        
        # Add divider
        cv2.line(combined, (self.width+10, 0), (self.width+10, self.height), 
               (200, 200, 200), 1)
        
        # Add joint positions and coordinates
        x, y = self.tip_coords[0]
        z = self.z_value
        
        joint_text = f"Joints: [{', '.join([str(int(j)) for j in self.predicted_joints])}]"
        coord_text = f"Input: X={x}, Y={y}, Z={z}"
        
        cv2.putText(combined, joint_text, (10, self.height-20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(combined, coord_text, (10, self.height-40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return combined
    
    def start(self):
        """Start the controller in a separate thread"""
        if not self.start_cameras():
            logger.error("Failed to start cameras")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Starting hand control loop")
        
        try:
            while self.running:
                # Process frames from both cameras
                self.process_frame(0)
                self.process_frame(1)
                
                # Predict joint positions if hand detected in first camera
                if self.hand_detected[0]:
                    x, y = self.tip_coords[0]
                    z = self.z_value
                    
                    # Predict and send to robot
                    self.predicted_joints = self.predict_joints(x, y, z)
                    if np.sum(self.predicted_joints) > 0:
                        self.robot.move_to_joint_positions(
                            self.predicted_joints.round().astype(int),
                            arm_type=self.arm_type
                        )
                
                # Optional: display the combined view
                cv2.imshow('Hand Controlled Robot', self.create_combined_display())
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Short sleep to reduce CPU usage
                time.sleep(0.001)
        
        except Exception as e:
            logger.error(f"Error in control loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop and release resources"""
        self.running = False
        
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # Close hand models
        for hand_model in self.hands_models:
            hand_model.close()
        
        # Release cameras
        for cap in self.caps:
            if cap and cap.isOpened():
                cap.release()
        
        cv2.destroyAllWindows()
        logger.info("Hand controller stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dual Camera Hand Controlled Robot')
    parser.add_argument('--arm', type=str, default='follower', choices=['follower', 'leader'],
                      help='Control arm type (follower or leader)')
    parser.add_argument('--model', type=str, default='lerobot/capdol/model_parameters2.npz',
                      help='Neural network model parameters file')
    parser.add_argument('--camera1', type=int, default=0,
                      help='Camera 1 ID (for X,Y coordinates)')
    parser.add_argument('--camera2', type=int, default=2,
                      help='Camera 2 ID (for Z value)')
    args = parser.parse_args()
    
    try:
        # Initialize robot
        logger.info("Initializing robot...")
        robot = ManipulatorRobot(KochRobotConfig())
        robot.connect()
        
        # Start controller
        logger.info("Starting hand controller...")
        controller = DualCameraHandRobotController(robot, args.model, args.arm)
        controller.camera1_id = args.camera1
        controller.camera2_id = args.camera2
        controller.start()
        
        # Simple status display
        print("\n=== Hand Controlled Robot ===")
        print(f"- Camera 1 (ID: {args.camera1}): X,Y coordinates")
        print(f"- Camera 2 (ID: {args.camera2}): Z value")
        print(f"- Controlling: {args.arm} arm")
        print("Press 'q' to quit\n")
        
        # Wait for user to quit
        try:
            while controller.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Clean up
        if 'controller' in locals():
            controller.stop()
        
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()