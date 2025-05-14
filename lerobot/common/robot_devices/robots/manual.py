import sys, termios, tty, select, logging
import numpy as np
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

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
            
        for arms in [self.follower_arms, self.leader_arms]:
            for name in arms:
                arms[name].disconnect()
                
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class JointController:
    def __init__(self, robot, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.arms = robot.follower_arms if arm_type == 'follower' else robot.leader_arms

    def move_joint(self, joint_index, delta_value):
        # Get current position as numpy array
        current_pos = np.array(self.arms['main'].read('Present_Position'), dtype=np.float32)
        
        # Create a new array for the goal position
        goal_pos = current_pos.copy()
        goal_pos[joint_index] += delta_value
        
        # Send the goal position
        self.robot.send_action(goal_pos, arm_type=self.arm_type)


class KeyboardController:
    def __init__(self, robot):
        self.robot = robot
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.follower_ctrl = JointController(robot, 'follower')
        self.leader_ctrl = JointController(robot, 'leader')
        
        # Key mapping: key -> (controller, joint_index, delta)
        self.key_mapping = {
            # Follower arm
            'w': (self.follower_ctrl, 0, 100.0),   # Shoulder pan up
            's': (self.follower_ctrl, 0, -100.0),  # Shoulder pan down
            'e': (self.follower_ctrl, 1, 100.0),   # Shoulder tilt up
            'd': (self.follower_ctrl, 1, -100.0),  # Shoulder tilt down
            'r': (self.follower_ctrl, 2, 100.0),   # Elbow up
            'f': (self.follower_ctrl, 2, -100.0),  # Elbow down
            't': (self.follower_ctrl, 3, 100.0),   # Wrist up
            'g': (self.follower_ctrl, 3, -100.0),  # Wrist down
            
            # Leader arm
            'y': (self.leader_ctrl, 0, 100.0),     # Shoulder pan up
            'h': (self.leader_ctrl, 0, -100.0),    # Shoulder pan down
            'u': (self.leader_ctrl, 1, 100.0),     # Shoulder tilt up
            'j': (self.leader_ctrl, 1, -100.0),    # Shoulder tilt down
            'i': (self.leader_ctrl, 2, 100.0),     # Elbow up
            'k': (self.leader_ctrl, 2, -100.0),    # Elbow down
            'o': (self.leader_ctrl, 3, 100.0),     # Wrist up
            'l': (self.leader_ctrl, 3, -100.0),    # Wrist down
        }

    def run(self):
        print("Koch Robot Control | Press 'q' to quit")
        print("Controls: w/s, e/d, r/f, t/g - Follower arm joints")
        print("          y/h, u/j, i/k, o/l - Leader arm joints")
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    c = sys.stdin.read(1)
                    if c == 'q':
                        break
                    if c in self.key_mapping:
                        controller, joint_index, delta = self.key_mapping[c]
                        controller.move_joint(joint_index, delta)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main():
    try:
        robot = ManipulatorRobot(KochRobotConfig())
        robot.connect()
        KeyboardController(robot).run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'robot' in locals() and getattr(robot, 'is_connected', False):
            robot.disconnect()


if __name__ == "__main__":
    main()