#!/usr/bin/env python3
"""
로봇 팔 캘리브레이션 스크립트

이 스크립트는 지정된 로봇 팔(leader arm 또는 follower arm)의 캘리브레이션을 수행하여
필요한 캘리브레이션 데이터를 /lerobot/capdol/ 디렉토리에 JSON 파일로 저장합니다.
"""

import os
import json
import logging
import numpy as np
import argparse
import time
from pathlib import Path
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
from lerobot.common.robot_devices.motors.dynamixel import (
    CalibrationMode,
    TorqueMode,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 캘리브레이션 상수
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90

# 모터 스텝 관련 상수 (convert_degrees_to_steps 대체용)
# 대부분의 다이나믹셀 모터는 4096 스텝을 360도로 사용
STEPS_PER_FULL_ROTATION = 4096  # 모터 사양에 따라 조정 필요
STEPS_PER_DEGREE = STEPS_PER_FULL_ROTATION / 360.0

# 기본 출력 디렉토리 설정
DEFAULT_OUTPUT_DIR = "lerobot/capdol/calibration_data"

def assert_drive_mode(drive_mode):
    """drive_mode 값이 유효한지 확인 (0 또는 1만 가능)"""
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")

def apply_drive_mode(position, drive_mode):
    """drive_mode를 위치 값에 적용"""
    assert_drive_mode(drive_mode)
    # drive_mode를 [0, 1]에서 [-1, 1]로 변환
    # 0 -> 1 (정방향), 1 -> -1 (역방향)
    signed_drive_mode = -(drive_mode * 2 - 1)
    position *= signed_drive_mode
    return position

def compute_nearest_rounded_position(position, models):
    """가장 가까운 회전 단위로 위치 값 반올림"""
    # convert_degrees_to_steps 대신 직접 계산
    delta_turn = STEPS_PER_DEGREE * ROTATED_POSITION_DEGREE
    nearest_pos = np.round(position.astype(float) / delta_turn) * delta_turn
    return nearest_pos.astype(position.dtype)

def degrees_to_steps(degrees):
    """각도를 모터 스텝으로 변환 (convert_degrees_to_steps 대체 함수)"""
    return int(degrees * STEPS_PER_DEGREE)

class ArmCalibrator:
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR):
        """
        팔 캘리브레이션 초기화
        
        Args:
            output_dir: 캘리브레이션 데이터 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.leader_arms = {}
        self.follower_arms = {}
    
    def connect_robot(self, arm_type=None):
        """
        로봇 연결 및 초기화
        
        Args:
            arm_type: 연결할 팔 유형 ('leader', 'follower', 'both', None=모두)
        """
        logger.info("로봇 초기화 중...")
        self.config = KochRobotConfig()
        
        if arm_type in ['leader', 'both', None]:  # leader 또는 둘 다
            self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
            # leader arm 연결
            for name in self.leader_arms:
                logger.info(f"리더 암 '{name}' 연결 중...")
                self.leader_arms[name].connect()
        
        if arm_type in ['follower', 'both', None]:  # follower 또는 둘 다
            self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
            # follower arm 연결
            for name in self.follower_arms:
                logger.info(f"팔로워 암 '{name}' 연결 중...")
                self.follower_arms[name].connect()
        
        logger.info("로봇 연결 완료")
        
        # 토크 비활성화 (캘리브레이션을 위해)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
    
    def calibrate_arm(self, arm_type):
        """
        특정 유형의 로봇 팔 캘리브레이션 수행
        
        Args:
            arm_type: 캘리브레이션할 팔 유형 ('leader', 'follower', 'both')
        """
        if arm_type == 'both':
            # 양쪽 팔 모두 캘리브레이션
            self.calibrate_arm('leader')
            self.calibrate_arm('follower')
            return
        
        logger.info(f"{arm_type.upper()} ARM 캘리브레이션 시작")
        
        # 팔 유형에 따라 선택
        arms = self.leader_arms if arm_type == 'leader' else self.follower_arms
        if not arms:
            logger.error(f"{arm_type} arm이 연결되어 있지 않습니다.")
            return
        
        # 캘리브레이션 수행 및 저장
        calibration_data = {}
        for name in arms:
            logger.info(f"{arm_type} 암 '{name}' 캘리브레이션 중...")
            calibration_data[name] = self._calibrate_single_arm(
                arms[name], arm_type, name)
        
        # 파일 이름 생성 및 저장
        filename = f"{arm_type}_arm.json"
        self._save_calibration(calibration_data, filename)
        
        logger.info(f"{arm_type.upper()} ARM 캘리브레이션 완료!")
    
    def _calibrate_single_arm(self, arm, arm_type, arm_name):
        """
        단일 로봇 팔 캘리브레이션
        
        Args:
            arm: 캘리브레이션할 로봇 암 객체
            arm_type: 암 유형 ('leader' 또는 'follower')
            arm_name: 암 이름
        
        Returns:
            dict: 캘리브레이션 데이터
        """
        print(f"\n======= {arm_type.upper()} ARM '{arm_name}' 캘리브레이션 =======")
        print("ARM 캘리브레이션은 3단계로 이루어집니다:")
        print("1. 영점 위치로 이동 (모든 관절 0도 위치)")
        print("2. 회전 위치로 이동 (모든 관절 90도 회전)")
        print("3. 휴식 위치로 이동 (안전하게 놓아둘 위치)")
        
        # 단계 1: 영점 위치
        print("\n1단계: 로봇 팔을 영점 위치로 이동시키세요")
        print("- 모든 관절이 0도 위치에 있어야 합니다.")
        print("- 팔이 수평으로 뻗어 있고, 그리퍼가 위를 향하게 하세요.")
        input("완료되면 Enter 키를 누르세요...")
        
        # 영점 위치 읽기
        zero_pos = arm.read("Present_Position")
        zero_target_pos = degrees_to_steps(ZERO_POSITION_DEGREE)
        
        print(f"영점 위치: {zero_pos}")
        
        # 단계 2: 회전 위치
        print("\n2단계: 로봇 팔을 회전 위치로 이동시키세요")
        print("- 각 관절을 시계 방향으로 약 90도 회전시키세요.")
        input("완료되면 Enter 키를 누르세요...")
        
        # 회전 위치 읽기
        rotated_pos = arm.read("Present_Position")
        rotated_target_pos = degrees_to_steps(ROTATED_POSITION_DEGREE)
        
        print(f"회전 위치: {rotated_pos}")
        
        # drive_mode 계산 (모터 회전 방향 - 0: 정방향, 1: 역방향)
        drive_mode = (rotated_pos < zero_pos).astype(np.int32)
        
        # 단계 3: 휴식 위치
        print("\n3단계: 로봇 팔을 휴식 위치로 이동시키세요")
        print("- 안전하게 놓아둘 수 있는 위치로 이동시키세요.")
        input("완료되면 Enter 키를 누르세요...")
        
        # 회전 방향에 따라 homing_offset 계산
        rotated_drived_pos = apply_drive_mode(rotated_pos, drive_mode)
        rotated_nearest_pos = compute_nearest_rounded_position(rotated_drived_pos, arm.motor_models)
        homing_offset = rotated_target_pos - rotated_nearest_pos
        
        # 회전 관절 (DEGREE)만 있다고 가정
        calib_mode = [CalibrationMode.DEGREE.name] * len(arm.motor_names)
        
        # 그리퍼의 경우 LINEAR로 설정 가능 (필요한 경우 수정)
        if "gripper" in arm.motor_names:
            calib_idx = arm.motor_names.index("gripper")
            calib_mode[calib_idx] = CalibrationMode.LINEAR.name
        
        # 캘리브레이션 데이터 반환
        calib_data = {
            "homing_offset": homing_offset.tolist(),
            "drive_mode": drive_mode.tolist(),
            "start_pos": zero_pos.tolist(),
            "end_pos": rotated_pos.tolist(),
            "calib_mode": calib_mode,
            "motor_names": arm.motor_names
        }
        
        print(f"\n{arm_type} '{arm_name}' 캘리브레이션 데이터:")
        print(f"  호밍 오프셋: {homing_offset.tolist()}")
        print(f"  드라이브 모드: {drive_mode.tolist()}")
        print(f"  모터 이름: {arm.motor_names}")
        
        return calib_data
    
    def _save_calibration(self, calibration_data, filename):
        """
        캘리브레이션 데이터를 JSON 파일로 저장
        
        Args:
            calibration_data: 저장할 캘리브레이션 데이터
            filename: 저장할 파일 이름
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 첫 번째 arm의 데이터만 저장 (일반적으로는 'main')
        if calibration_data:
            arm_name = next(iter(calibration_data))
            with open(filepath, 'w') as f:
                json.dump(calibration_data[arm_name], f, indent=2)
            logger.info(f"캘리브레이션 데이터가 {filepath}에 저장되었습니다.")
        else:
            logger.warning(f"저장할 캘리브레이션 데이터가 없습니다: {filename}")
    
    def disconnect(self):
        """로봇 연결 해제"""
        logger.info("로봇 연결 해제 중...")
        
        # leader arms 연결 해제
        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        
        # follower arms 연결 해제
        for name in self.follower_arms:
            self.follower_arms[name].disconnect()
        
        logger.info("로봇 연결 해제 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Robot Arm Calibration')
    parser.add_argument('--arm', type=str, default=None, choices=['leader', 'follower', 'both'],
                        help='Arm to calibrate (leader, follower, or both)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save calibration data (default: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()
    
    calibrator = None
    
    try:
        print("\n===== 로봇 팔 캘리브레이션 =====")
        
        if args.arm is None:
            # 사용자에게 팔 선택 요청
            print("캘리브레이션할 로봇 팔을 선택하세요:")
            print("1. Leader Arm")
            print("2. Follower Arm")
            print("3. Both Arms")
            choice = input("선택 (1, 2, 또는 3): ").strip()
            
            if choice == '1':
                arm_type = 'leader'
            elif choice == '2':
                arm_type = 'follower'
            elif choice == '3':
                arm_type = 'both'
            else:
                print("잘못된 선택입니다. 종료합니다.")
                return
        else:
            arm_type = args.arm
        
        if arm_type == 'both':
            print("\n두 팔 모두(LEADER ARM과 FOLLOWER ARM)를 캘리브레이션합니다.")
        else:
            print(f"\n{arm_type.upper()} ARM을 캘리브레이션합니다.")
        
        print("캘리브레이션 과정에서 로봇 팔을 수동으로 움직여야 합니다.")
        print("지시에 따라 각 단계를 진행해주세요.")
        
        # 캘리브레이터 초기화 및 로봇 연결
        calibrator = ArmCalibrator(args.output)
        calibrator.connect_robot(arm_type)
        
        # 선택된 팔 캘리브레이션
        calibrator.calibrate_arm(arm_type)
        
        if arm_type == 'both':
            print("\nLEADER ARM과 FOLLOWER ARM 캘리브레이션이 성공적으로 완료되었습니다!")
            print(f"캘리브레이션 데이터는 다음 위치에 저장되었습니다:")
            print(f"  - '{args.output}/leader_arm.json'")
            print(f"  - '{args.output}/follower_arm.json'")
        else:
            print(f"\n{arm_type.upper()} ARM 캘리브레이션이 성공적으로 완료되었습니다!")
            print(f"캘리브레이션 데이터는 '{args.output}/{arm_type}_arm.json'에 저장되었습니다.")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    finally:
        # 로봇 연결 해제
        if calibrator:
            calibrator.disconnect()


if __name__ == "__main__":
    main()