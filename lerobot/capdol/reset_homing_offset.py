#!/usr/bin/env python3
"""
모든 모터의 homing offset을 2048로 설정하는 스크립트

이 스크립트는 로봇 팔의 모든 모터에 대해 homing offset을 2048로 설정하여
중간 위치(0도)가 정확히 2048 step이 되도록 합니다.
"""

import os
import json
import logging
import numpy as np
import argparse
import sys
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HomingOffset")

# 필요한 모듈 가져오기
try:
    from lerobot.common.robot_devices.robots.configs import KochRobotConfig
    from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs
    from lerobot.common.robot_devices.motors.dynamixel import (
        CalibrationMode, TorqueMode
    )
except ImportError as e:
    logger.critical(f"필수 라이브러리를 불러올 수 없습니다: {e}")
    logger.critical("LeRobot 패키지가 올바르게 설치되어 있는지 확인하세요.")
    sys.exit(1)

# 상수 설정
DEFAULT_HOMING_OFFSET = 2048  # 중간 위치
DEFAULT_OUTPUT_DIR = "lerobot/capdol/calibration_data"

def reset_homing_offset(arm_type=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    지정된 로봇 팔의 모든 모터에 대해 homing offset을 2048로 설정
    
    Args:
        arm_type: 'leader', 'follower', 또는 None(모두)
        output_dir: 캘리브레이션 데이터 저장 디렉토리
    """
    print(f"모든 모터의 homing offset을 {DEFAULT_HOMING_OFFSET}로 설정합니다...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 로봇 설정 로드
    try:
        config = KochRobotConfig()
    except Exception as e:
        logger.critical(f"로봇 설정을 로드할 수 없습니다: {e}")
        raise
    
    leader_arms = {}
    follower_arms = {}
    
    # 필요한 로봇 팔 연결
    if arm_type != 'follower':  # leader 또는 모두
        leader_arms = make_motors_buses_from_configs(config.leader_arms)
        for name in leader_arms:
            logger.info(f"리더 암 '{name}' 연결 중...")
            leader_arms[name].connect()
            print(f"리더 암 '{name}' 연결됨")
    
    if arm_type != 'leader':  # follower 또는 모두
        follower_arms = make_motors_buses_from_configs(config.follower_arms)
        for name in follower_arms:
            logger.info(f"팔로워 암 '{name}' 연결 중...")
            follower_arms[name].connect()
            print(f"팔로워 암 '{name}' 연결됨")
    
    try:
        # 리더 암 처리
        for name, arm in leader_arms.items():
            _reset_arm_offsets(arm, name, 'leader', output_dir)
        
        # 팔로워 암 처리
        for name, arm in follower_arms.items():
            _reset_arm_offsets(arm, name, 'follower', output_dir)
        
    finally:
        # 모든 연결 해제
        for name in leader_arms:
            leader_arms[name].disconnect()
            print(f"리더 암 '{name}' 연결 해제됨")
        
        for name in follower_arms:
            follower_arms[name].disconnect()
            print(f"팔로워 암 '{name}' 연결 해제됨")

def _reset_arm_offsets(arm, arm_name, arm_type, output_dir):
    """
    단일 로봇 팔의 모든 모터에 대해 homing offset 설정
    
    Args:
        arm: 로봇 암 객체
        arm_name: 암 이름
        arm_type: 암 유형 ('leader' 또는 'follower')
        output_dir: 출력 디렉토리
    """
    print(f"\n{arm_type.upper()} ARM '{arm_name}'의 calibration 리셋 중...")
    
    # 현재 위치 읽기
    current_positions = arm.read("Present_Position")
    print(f"현재 모터 위치: {current_positions}")
    
    # 기존 캘리브레이션 데이터 로드 시도
    filepath = os.path.join(output_dir, f"{arm_type}_arm.json")
    try:
        with open(filepath, 'r') as f:
            existing_calib = json.load(f)
            print(f"기존 캘리브레이션 데이터 로드됨")
    except (FileNotFoundError, json.JSONDecodeError):
        # 기존 파일이 없거나 유효하지 않은 경우 기본값 생성
        existing_calib = {
            "motor_names": arm.motor_names,
            "drive_mode": [0] * len(arm.motor_names),
            "calib_mode": [CalibrationMode.DEGREE.name] * len(arm.motor_names),
            "start_pos": current_positions.tolist(),
            "end_pos": current_positions.tolist()
        }
        print(f"기존 캘리브레이션 데이터가 없어 기본값 생성")
    
    # 모든 모터의 homing offset을 2048로 설정
    new_homing_offset = np.array([DEFAULT_HOMING_OFFSET] * len(arm.motor_names))
    
    # 새 캘리브레이션 데이터 구성
    new_calib = existing_calib.copy()
    new_calib["homing_offset"] = new_homing_offset.tolist()
    
    # 파일에 저장
    with open(filepath, 'w') as f:
        json.dump(new_calib, f, indent=2)
    print(f"새 캘리브레이션 데이터가 {filepath}에 저장됨")
    print(f"모든 homing offset을 {DEFAULT_HOMING_OFFSET}로 설정함")
    
    # 모터 정보 출력
    print("\n모터 정보:")
    for idx, motor_name in enumerate(arm.motor_names):
        print(f"  {motor_name}: homing_offset={new_homing_offset[idx]}, "
              f"drive_mode={new_calib['drive_mode'][idx]}, "
              f"calib_mode={new_calib['calib_mode'][idx]}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='모든 모터의 homing offset을 2048로 설정하는 스크립트'
    )
    parser.add_argument('--arm', type=str, default=None, 
                      choices=['leader', 'follower'],
                      help='설정할 로봇 팔 (leader, follower, 또는 생략 시 모두)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f'캘리브레이션 데이터 저장 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    try:
        # 사용자 선택 (인자로 지정하지 않은 경우)
        arm_type = args.arm
        if arm_type is None:
            print("homing offset을 리셋할 로봇 팔을 선택하세요:")
            print("1. Leader Arm")
            print("2. Follower Arm")
            print("3. 모든 로봇 팔")
            
            choice = input("선택 (1, 2, 또는 3): ").strip()
            if choice == '1':
                arm_type = 'leader'
            elif choice == '2':
                arm_type = 'follower'
            elif choice == '3':
                arm_type = None  # 모두
            else:
                print("잘못된 선택입니다. 종료합니다.")
                return
        
        # homing offset 리셋 실행
        reset_homing_offset(arm_type, args.output)
        print("\nhoming offset 리셋 완료!")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()