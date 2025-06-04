# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import time
import logging
import numpy as np
import torch
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError

def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    """Cap relative action target magnitude for safety."""
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos

class TeleopRobot:
    """Simplified robot class focused on teleoperation functionality."""
    
    def __init__(self, leader_arms, follower_arms, cameras=None, max_relative_target=None):
        self.leader_arms = leader_arms
        self.follower_arms = follower_arms
        self.cameras = cameras or {}
        self.max_relative_target = max_relative_target
        self.is_connected = False
        self.logs = {}
    
    def connect(self):
        """Connect to robot arms and cameras."""
        # Implementation would go here
        self.is_connected = True
        
    def teleop_step(self, record_data=False, print_positions=True):
        """Execute one step of teleoperation, reading from leader and commanding follower.
        
        Args:
            record_data: Whether to record and return observation/action data
            print_positions: Whether to print current positions of both arms
            
        Returns:
            None or (observation_dict, action_dict) if record_data=True
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Robot is not connected. You need to run `robot.connect()`."
            )

        # Read the position of the leader
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
            
            if print_positions:
                print(f"Leader arm '{name}' position: {leader_pos[name].tolist()}")

        # Send goal position to the follower
        follower_goal_pos = {}
        follower_present_pos = {}
        
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Read current follower position
            present_pos = self.follower_arms[name].read("Present_Position")
            present_pos = torch.from_numpy(present_pos)
            follower_present_pos[name] = present_pos
            
            if print_positions:
                print(f"Follower arm '{name}' current position: {present_pos.tolist()}")

            # Cap goal position when too far away from present position
            if self.max_relative_target is not None:
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.max_relative_target)

            # Store for recording
            follower_goal_pos[name] = goal_pos

            # Send to hardware
            goal_pos_np = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos_np)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t
            
            if print_positions:
                print(f"Follower arm '{name}' goal position: {goal_pos.tolist()}")

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            state.append(follower_present_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict
    
    def disconnect(self):
        """Disconnect from all hardware devices."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Robot is not connected. You need to run `robot.connect()` before disconnecting."
            )
            
        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False
        
    def __del__(self):
        """Ensure disconnection when object is deleted."""
        if getattr(self, "is_connected", False):
            self.disconnect()