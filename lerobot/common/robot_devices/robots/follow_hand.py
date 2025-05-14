import sys
import logging
import time
import numpy as np
import cv2
import mediapipe as mp
import torch
import argparse
import os

from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from typing import Tuple, Optional, Dict, List


class JointController:
    """Manages joint movements for a robotic arm."""
    
    def __init__(self, robot, arm_type='follower'):
        """Initialize the joint controller.
        
        Args:
            robot (ManipulatorRobot): The robot to control
            arm_type (str): 'follower' or 'leader' arm type
        """
        self.robot = robot
        self.arm_type = arm_type
        self.arms = self.robot.follower_arms if arm_type == 'follower' else self.robot.leader_arms
        self.logger = logging.getLogger(__name__)

    def _move_joint(self, joint_index, delta_value):
        """Move a specific joint by a delta value.
        
        Args:
            joint_index (int): Index of the joint to move
            delta_value (float): Amount to move the joint
        """
        try:
            # Read current position
            current_pos = self.arms['main'].read('Present_Position')
            current_pos = torch.as_tensor(current_pos, dtype=torch.float32)
            
            # Create delta tensor
            delta = torch.zeros_like(current_pos)
            delta[joint_index] = delta_value
            
            # Calculate goal position
            goal_pos = current_pos + delta
            
            # Send action with arm type
            self.robot.send_action(goal_pos, arm_type=self.arm_type)
            
            self.logger.info(f"Moving {self.arm_type} joint {joint_index}. "
                             f"Current pos: {current_pos}, Goal pos: {goal_pos}")
        
        except Exception as e:
            self.logger.error(f"Error moving joint {joint_index}: {e}")
            raise
            
    def get_current_position(self):
        """Get the current position of all joints.
        
        Returns:
            torch.Tensor: Current position of all joints
        """
        try:
            current_pos = self.arms['main'].read('Present_Position')
            return torch.as_tensor(current_pos, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"Error reading current position: {e}")
            raise


class HandTrackingController:
    """Controls robot arm using hand tracking via camera."""
    
    def __init__(self, robot, config=None):
        """Initialize the hand tracking controller.
        
        Args:
            robot (ManipulatorRobot): The robot to control
            config (dict, optional): Configuration parameters
        """
        self.robot = robot
        self.follower_controller = JointController(robot, 'follower')
        self.leader_controller = JointController(robot, 'leader')
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'camera_id': 0,
            'camera_width': 640,
            'camera_height': 480,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'model_complexity': 0,
            'control_hand': 'right',  # Which hand controls the robot: 'right', 'left', or 'both'
            'control_arm': 'follower',  # Which arm to control: 'follower', 'leader', or 'both'
            'update_rate': 0.1,  # Time between updates in seconds
            'x_sensitivity': 0.5,  # Sensitivity for x-axis movement (shoulder pan)
            'y_sensitivity': 0.5,  # Sensitivity for y-axis movement (shoulder tilt)
            'z_sensitivity': 3.0,  # Sensitivity for z-axis movement (elbow)
            'max_delta': 15.0,     # Maximum delta value for any joint
        }
        
        # Update configuration with provided values
        if config is not None:
            self.config.update(config)
            
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Previous hand positions for delta calculation
        self.prev_positions = {}
        
        # Joint mapping
        # Maps hand movement dimensions to robot joints:
        # x -> joint 0 (shoulder pan)
        # y -> joint 1 (shoulder tilt)
        # z -> joint 2 (elbow)
        # We'll use a combination of y and z for joint 3 (wrist)
        self.joint_mapping = {
            'x': 0,  # X movement -> shoulder pan
            'y': 1,  # Y movement -> shoulder tilt
            'z': 2,  # Z movement -> elbow
        }
        
    def _list_available_cameras(self, max_cameras=10):
        """List all available camera devices.
        
        Args:
            max_cameras (int): Maximum number of cameras to check
            
        Returns:
            List[int]: List of available camera indices
        """
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
        
    def _init_camera(self):
        """Initialize the camera for video capture.
        
        Attempts to find and open an available camera.
        
        Returns:
            cv2.VideoCapture: Initialized camera object
            
        Raises:
            RuntimeError: If no camera could be opened
        """
        # First try the specified camera_id
        self.logger.info(f"Attempting to open camera with ID {self.config['camera_id']}")
        
        # Try different backend options
        backends = [
            cv2.CAP_V4L2,   # Video4Linux2 - most common on Linux
            cv2.CAP_V4L,    # Original Video4Linux
            cv2.CAP_ANY     # Let OpenCV choose the best backend
        ]
        
        cap = None
        for backend in backends:
            self.logger.info(f"Trying backend: {backend}")
            cap = cv2.VideoCapture(self.config['camera_id'], backend)
            if cap.isOpened():
                self.logger.info(f"Successfully opened camera with backend {backend}")
                break
            else:
                self.logger.warning(f"Failed to open camera with backend {backend}")
                cap.release()
                cap = None
        
        # If that fails, try to find any available camera
        if cap is None:
            self.logger.warning(f"Failed to open camera with ID {self.config['camera_id']}, searching for available cameras...")
            available_cameras = self._list_available_cameras()
            
            if not available_cameras:
                self.logger.error("No cameras available")
                raise RuntimeError("No cameras available")
                
            self.logger.info(f"Found available cameras at indices: {available_cameras}")
            
            # Try the first available camera with each backend
            for camera_id in available_cameras:
                for backend in backends:
                    self.logger.info(f"Trying camera {camera_id} with backend {backend}")
                    cap = cv2.VideoCapture(camera_id, backend)
                    if cap.isOpened():
                        self.logger.info(f"Successfully opened camera {camera_id} with backend {backend}")
                        # Update the config with the actual camera ID used
                        self.config['camera_id'] = camera_id
                        break
                if cap is not None and cap.isOpened():
                    break
        
        if cap is None or not cap.isOpened():
            self.logger.error("Failed to open any camera")
            raise RuntimeError("Failed to open any camera")
        
        # Try to set camera format to MJPG for better performance
        # First, save the original format
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        original_fourcc_str = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        self.logger.info(f"Original camera format: {original_fourcc_str}")
        
        # Try MJPG format first (faster)
        mjpg_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        ret = cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
        if ret:
            self.logger.info("Successfully set camera format to MJPG")
        else:
            self.logger.warning("Failed to set MJPG format, using default format")
        
        # Configure camera resolution
        self.logger.info(f"Setting camera resolution to {self.config['camera_width']}x{self.config['camera_height']}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
        
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify the camera is properly configured
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        self.logger.info(f"Camera configuration:")
        self.logger.info(f"  Resolution: {actual_width}x{actual_height}")
        self.logger.info(f"  FPS: {actual_fps}")
        self.logger.info(f"  Format: {actual_fourcc_str}")
        
        # Read one frame to verify camera works
        ret, frame = cap.read()
        if not ret:
            self.logger.error("Failed to read first frame from camera")
            cap.release()
            raise RuntimeError("Failed to read first frame from camera")
            
        self.logger.info(f"Successfully read first frame: {frame.shape}")
        
        return cap
        
    def _get_controller(self, hand_type):
        """Get the appropriate controller based on hand type and configuration.
        
        Args:
            hand_type (str): 'left' or 'right'
            
        Returns:
            JointController: The controller to use
        """
        if self.config['control_arm'] == 'both':
            # Use right hand for follower, left hand for leader
            return self.follower_controller if hand_type == 'right' else self.leader_controller
        else:
            # Use configured arm type
            return self.follower_controller if self.config['control_arm'] == 'follower' else self.leader_controller
    
    def _should_process_hand(self, hand_type):
        """Determine if the hand should be processed based on configuration.
        
        Args:
            hand_type (str): 'left' or 'right'
            
        Returns:
            bool: True if the hand should be processed
        """
        control_hand = self.config['control_hand']
        return (control_hand == 'both' or 
                (control_hand == 'right' and hand_type == 'right') or
                (control_hand == 'left' and hand_type == 'left'))
    
    def _calculate_joint_deltas(self, current_pos, prev_pos):
        """Calculate delta values for each joint based on hand movement.
        
        Args:
            current_pos (Dict[str, float]): Current x, y, z positions
            prev_pos (Dict[str, float]): Previous x, y, z positions
            
        Returns:
            Dict[int, float]: Joint index to delta value mapping
        """
        deltas = {}
        
        # Calculate raw deltas
        dx = (prev_pos['x'] - current_pos['x']) * self.config['x_sensitivity'] * 100
        dy = (prev_pos['y'] - current_pos['y']) * self.config['y_sensitivity'] * 100
        dz = (prev_pos['z'] - current_pos['z']) * self.config['z_sensitivity'] * 100
        
        # Map to joints
        deltas[self.joint_mapping['x']] = dx
        deltas[self.joint_mapping['y']] = dy
        deltas[self.joint_mapping['z']] = dz
        
        # Calculate wrist movement (joint 3) - using a combination of y and z
        deltas[3] = (dy - dz) * 0.5
        
        # Clip to max delta
        max_delta = self.config['max_delta']
        for joint in deltas:
            deltas[joint] = np.clip(deltas[joint], -max_delta, max_delta)
            
        return deltas
    
    def _process_hand_landmarks(self, results, image_shape):
        """Process hand landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe hand detection results
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dict: Processed hand data
        """
        hands_data = {}
        
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return hands_data
            
        width, height = image_shape[1], image_shape[0]
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand type (left or right)
            hand_type = handedness.classification[0].label.lower()
            
            if not self._should_process_hand(hand_type):
                continue
                
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Store normalized coordinates
            hands_data[hand_type] = {
                'x': index_finger_tip.x,
                'y': index_finger_tip.y,
                'z': index_finger_tip.z,
                'pixel_x': int(index_finger_tip.x * width),
                'pixel_y': int(index_finger_tip.y * height),
            }
            
        return hands_data
        
    def _move_robot_joints(self, hands_data):
        """Move robot joints based on hand tracking data.
        
        Args:
            hands_data (Dict): Hand tracking data
        """
        for hand_type, position in hands_data.items():
            # Get controller for this hand
            controller = self._get_controller(hand_type)
            
            # Check if we have previous position data for this hand
            if hand_type in self.prev_positions:
                # Calculate joint deltas
                deltas = self._calculate_joint_deltas(position, self.prev_positions[hand_type])
                
                # Move each joint
                for joint_index, delta_value in deltas.items():
                    # Only move if delta is significant
                    if abs(delta_value) > 1.0:
                        try:
                            controller._move_joint(joint_index, delta_value)
                        except Exception as e:
                            self.logger.error(f"Error moving joint {joint_index}: {e}")
            
            # Update previous position
            self.prev_positions[hand_type] = position
    
    def _print_info(self):
        """Print controller information and instructions."""
        print("Hand Tracking Robot Control")
        print("===========================")
        print(f"Control hand: {self.config['control_hand']}")
        print(f"Control arm: {self.config['control_arm']}")
        print("\nJoint mapping:")
        print("  Left-right movement -> Shoulder pan (joint 0)")
        print("  Up-down movement -> Shoulder tilt (joint 1)")
        print("  Forward-backward movement -> Elbow (joint 2)")
        print("  Combined movement -> Wrist (joint 3)")
        print("\nPress 'q' to quit")
    
    def _get_camera_info(self):
        """Get information about available cameras.
        
        Returns:
            str: Information about available cameras
        """
        available_cameras = self._list_available_cameras()
        camera_info = f"Available cameras: {available_cameras}\n"
        
        if available_cameras:
            camera_info += "Camera details:\n"
            for idx in available_cameras:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    camera_info += f"  Camera {idx}: {width}x{height} @ {fps}fps\n"
                    cap.release()
        
        return camera_info
        
    def _run_without_camera(self):
        """Run the controller without camera for testing purposes."""
        self.logger.info("Running in camera-less mode (for testing)")
        print("Camera-less mode active. Simulating hand movements.")
        print("Press Ctrl+C to exit")
        
        # Simulated hand positions
        simulated_positions = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0},  # Center
            {'x': 0.6, 'y': 0.5, 'z': 0.0},  # Right
            {'x': 0.4, 'y': 0.5, 'z': 0.0},  # Left
            {'x': 0.5, 'y': 0.4, 'z': 0.0},  # Up
            {'x': 0.5, 'y': 0.6, 'z': 0.0},  # Down
            {'x': 0.5, 'y': 0.5, 'z': -0.1}, # Forward
            {'x': 0.5, 'y': 0.5, 'z': 0.1},  # Backward
        ]
        
        # Define hand type based on configuration
        hand_type = self.config['control_hand']
        if hand_type == 'both':
            hand_type = 'right'  # Default to right in simulation mode
            
        try:
            # Initialize previous position
            self.prev_positions[hand_type] = simulated_positions[0]
            
            # Iterate through simulated positions
            for i, position in enumerate(simulated_positions):
                # Add pixel coordinates for consistency
                position['pixel_x'] = int(position['x'] * self.config['camera_width'])
                position['pixel_y'] = int(position['y'] * self.config['camera_height'])
                
                # Create hands data structure
                hands_data = {hand_type: position}
                
                # Print simulated position
                print(f"Simulated position {i+1}/{len(simulated_positions)}: "
                      f"({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})")
                
                # Move robot
                self._move_robot_joints(hands_data)
                
                # Wait between movements
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
            
    def run(self):
        """Run the hand tracking controller."""
        self._print_info()
        
        # Try to initialize camera
        cap = None
        try:
            cap = self._init_camera()
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.logger.error(self._get_camera_info())
            
            # Ask if user wants to run without camera
            print("\nNo camera available. Options:")
            print("1. Run without camera (simulation mode)")
            print("2. Exit")
            
            choice = input("Enter choice (1/2): ")
            if choice == '1':
                self._run_without_camera()
                return
            else:
                print("Exiting...")
                return
            
        # Initialize hand detection
        with self.mp_hands.Hands(
            model_complexity=self.config['model_complexity'],
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']) as hands:
            
            last_update_time = time.time()
            last_frame_time = time.time()
            fps = 0
            frame_count = 0
            
            while cap.isOpened():
                # Calculate FPS
                current_time = time.time()
                if current_time - last_frame_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_frame_time = current_time
                
                # Read frame
                success, image = cap.read()
                if not success:
                    self.logger.warning("Failed to read frame from camera")
                    # Try to reopen the camera if reading fails
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process image
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                
                # Draw hand landmarks (for visualization)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Process hand landmarks
                hands_data = self._process_hand_landmarks(results, image.shape)
                
                # Display information
                cv2.putText(
                    image,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                y_offset = 60
                for hand_type, data in hands_data.items():
                    text = f"{hand_type.capitalize()}: ({data['pixel_x']}, {data['pixel_y']}, {data['z']:.2f})"
                    cv2.putText(
                        image, 
                        text, 
                        (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0) if hand_type == 'left' else (0, 0, 255), 
                        2)
                    y_offset += 30
                
                # Update robot at specified rate
                if current_time - last_update_time >= self.config['update_rate'] and hands_data:
                    self._move_robot_joints(hands_data)
                    last_update_time = current_time
                
                # Show the image
                cv2.imshow('Robot Hand Control', image)
                
                # Check for quit
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset previous positions
                    self.prev_positions = {}
                    self.logger.info("Reset hand tracking reference positions")
                    
        # Clean up
        if cap:
            cap.release()
        cv2.destroyAllWindows()


class MockRobot:
    """Mock robot implementation for testing without hardware."""
    
    def __init__(self):
        """Initialize the mock robot."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using MockRobot for testing (no hardware required)")
        
        # Create mock arms
        self.follower_arms = {'main': MockArm('follower')}
        self.leader_arms = {'main': MockArm('leader')}
        
    def connect(self):
        """Mock connect method."""
        self.logger.info("MockRobot connected")
        
    def disconnect(self):
        """Mock disconnect method."""
        self.logger.info("MockRobot disconnected")
        
    def send_action(self, goal_pos, arm_type='follower'):
        """Mock send_action method.
        
        Args:
            goal_pos (torch.Tensor): Goal position
            arm_type (str): Arm type ('follower' or 'leader')
        """
        self.logger.info(f"Moving {arm_type} arm to position: {goal_pos}")
        
        
class MockArm:
    """Mock arm implementation for testing."""
    
    def __init__(self, arm_type):
        """Initialize the mock arm.
        
        Args:
            arm_type (str): Arm type ('follower' or 'leader')
        """
        self.arm_type = arm_type
        self.position = torch.zeros(4, dtype=torch.float32)
        self.logger = logging.getLogger(__name__)
        
    def read(self, parameter):
        """Mock read method.
        
        Args:
            parameter (str): Parameter to read
            
        Returns:
            torch.Tensor: Current position
        """
        if parameter == 'Present_Position':
            self.logger.debug(f"Reading {self.arm_type} arm position: {self.position}")
            return self.position
        
        return None


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Hand Tracking Robot Control')
    
    # Camera options
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera device index (default: 0)')
    parser.add_argument('--width', type=int, default=640, 
                        help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, 
                        help='Camera height (default: 480)')
    
    # Control options
    parser.add_argument('--control-hand', type=str, default='right', choices=['left', 'right', 'both'],
                        help='Hand to use for control (default: right)')
    parser.add_argument('--control-arm', type=str, default='follower', choices=['follower', 'leader', 'both'],
                        help='Arm to control (default: follower)')
    parser.add_argument('--update-rate', type=float, default=0.1,
                        help='Update rate in seconds (default: 0.1)')
                        
    # Robot options
    parser.add_argument('--mock', action='store_true',
                        help='Use mock robot (for testing without hardware)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main function to set up and run hand tracking robot control."""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Log system information
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"MediaPipe version: {mp.__version__}")
    logger.info(f"Python version: {sys.version}")
    
    robot = None
    try:
        # Configuration from arguments
        config = {
            'camera_id': args.camera,
            'camera_width': args.width,
            'camera_height': args.height,
            'control_hand': args.control_hand,
            'control_arm': args.control_arm,
            'update_rate': args.update_rate,
        }
        
        # Initialize the robot
        if args.mock:
            robot = MockRobot()
            robot.connect()
        else:
            logger.info("Initializing robot hardware...")
            robot_config = KochRobotConfig()
            robot = ManipulatorRobot(robot_config)
            robot.connect()

        # Create a hand tracking controller
        controller = HandTrackingController(robot, config)
        controller.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping robot control.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure robot is properly disconnected
        if robot:
            try:
                robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting robot: {e}")


if __name__ == "__main__":
    main()