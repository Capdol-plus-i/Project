# robot.py
import logging
import numpy as np
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

logger = logging.getLogger(__name__)

class ManipulatorRobot:
    def __init__(self, config):
        self.config = config
        self.leader_arms = make_motors_buses_from_configs(config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(config.follower_arms)
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            return
            
        # Connect arms
        for name in self.follower_arms:
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            self.leader_arms[name].connect()

        # Import dynamically to avoid dependency issues
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        
        # First disable torque for configuration
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        
        # Configure operating mode properly for each motor type
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                # Use integer 4 (Extended Position Control Mode)
                arms[name].write("Operating_Mode", 4)
                # Re-enable torque
                arms[name].write("Torque_Enable", 1)

        self.is_connected = True
        logger.info("Robot connected successfully")

    def send_action(self, action, arm_type='follower'):
        arms = self.follower_arms if arm_type == 'follower' else self.leader_arms
        
        from_idx, to_idx = 0, 0
        for name in arms:
            motor_count = len(arms[name].motor_names)
            to_idx += motor_count
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            
            # Read current position and ensure numpy array
            present_pos = np.array(arms[name].read("Present_Position"), dtype=np.float32)
            
            # Convert goal position to numpy array
            if hasattr(goal_pos, 'numpy'):  # For torch tensors
                goal_pos = goal_pos.numpy()
            goal_pos = np.array(goal_pos, dtype=np.float32)
            
            # Apply safety limits
            max_delta = 150.0
            diff = goal_pos - present_pos
            diff = np.clip(diff, -max_delta, max_delta)
            safe_goal_pos = present_pos + diff
            
            # IMPORTANT: Convert to integers before sending to dynamixel
            safe_goal_pos = np.round(safe_goal_pos).astype(np.int32)
            
            # Send command to motors
            arms[name].write("Goal_Position", safe_goal_pos)

    def disconnect(self):
        if not self.is_connected:
            return
            
        logger.info("Disconnecting robot")
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()