import numpy as np, serial, threading, time
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

class Robot:
    def __init__(self):
        self.arms = make_motors_buses_from_configs(KochRobotConfig().leader_arms)
        self.on = False
        
    def connect(self):
        if self.on: return
        for a in self.arms.values():
            a.connect()
            a.write("Torque_Enable", 0)
            a.write("Operating_Mode", 3)
            a.write("Torque_Enable", 1)
            a.write("Position_P_Gain", 100)
        self.on = True
        
    def set_position(self, joint, pos):
        p = np.array(self.arms['main'].read('Present_Position'), dtype=np.float32)
        p[joint] = pos
        i, arms = 0, list(self.arms.values())
        for a in arms:
            n = len(a.motor_names)
            g = np.array(p[i:i+n], dtype=np.float32)
            # 클리핑 없이 직접 목표 위치로 이동
            a.write("Goal_Position", np.round(g).astype(np.uint32))
            i += n
            
    def close(self): 
        if self.on:
            [a.disconnect() for a in self.arms.values()]
            self.on = False
    
    def __del__(self): 
        if self.on: self.close()

def serial_listener(robot, ser):
    POS_0, POS_1 = 110, 1110  # 모드별 절대 위치
    
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("CMD:ROBOT:"):
                try:
                    mode = int(line.split(":")[-1])
                    robot.set_position(0, POS_0 if mode == 0 else POS_1)
                    print(f"모드 {mode}: 위치 {POS_0 if mode == 0 else POS_1}으로 이동")
                except Exception as e:
                    print(f"명령 처리 오류: {e}")
        time.sleep(0.1)

def main():
    r, ser = None, None
    try:
        ser = serial.Serial('/dev/ttyACM2', 9600, timeout=1)
        time.sleep(2)
        r = Robot()
        r.connect()
        print("로봇 연결 완료, 아두이노에서 버튼 입력 대기 중...")
        
        serial_listener(r, ser)
            
    except KeyboardInterrupt:
        print("프로그램 종료")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if r and r.on: r.close()
        if ser and ser.is_open: ser.close()
        print("연결 종료")

if __name__ == "__main__": main()