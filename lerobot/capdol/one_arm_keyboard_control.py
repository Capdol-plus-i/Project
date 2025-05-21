import sys, termios, tty, select, numpy as np
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
        self.on = True
        
    def act(self, act):
        i, arms = 0, list(self.arms.values())
        for a in arms:
            n = len(a.motor_names)
            g = np.array(act[i:i+n], dtype=np.float32)
            p = np.array(a.read("Present_Position"), dtype=np.float32)
            a.write("Goal_Position", np.round(p+np.clip(g-p,-150,150)).astype(np.uint32))
            i += n
            
    def close(self): 
        [a.disconnect() for a in self.arms.values() if self.on]
        self.on = False
        
    def __del__(self): self.close() if self.on else None

def main():
    r = None
    try:
        r = Robot()
        r.connect()
        old = termios.tcgetattr(sys.stdin)
        keys = {'y':(0,100), 'h':(0,-100)}
        print("Robot Control | 'q':quit, 'y/h':shoulder")
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if sys.stdin in select.select([sys.stdin],[],[], 0)[0]:
                c = sys.stdin.read(1)
                if c == 'q': break
                if c in keys:
                    j, d = keys[c]
                    p = np.array(r.arms['main'].read('Present_Position'), dtype=np.float32)
                    p[j] += d
                    r.act(p)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        if r and r.on: r.close()

if __name__ == "__main__": main()