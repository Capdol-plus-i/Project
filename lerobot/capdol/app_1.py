# app.py
import sys, logging, threading, time, os
from flask import Flask, Response, render_template, jsonify, request
from lerobot.common.robot_devices.robots.configs import KochRobotConfig

# Import our modules
from robot import ManipulatorRobot
from controllers import KeyboardController, HandController
from camera import CameraProcessor, get_camera_info

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 로그 디렉터리 확인/생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 파일 핸들러 추가
file_handler = logging.FileHandler(
    os.path.join(log_dir, 'robot_control.log')
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Flask 웹 애플리케이션
app = Flask(__name__)

# 전역 변수
robot = None
keyboard_controller = None
hand_controller = None
control_mode = "none"  # "keyboard", "hand", "none"
camera_processor = None

def gen_frames():
    """비디오 프레임 제너레이터"""
    global hand_controller, control_mode, camera_processor
    
    if camera_processor is None:
        # 더 낮은 해상도로 설정하여 지연 시간 감소
        camera_processor = CameraProcessor(camera_id=8, width=640, height=480)
    
    # 카메라 정보 로깅
    cameras = get_camera_info()
    if cameras:
        logger.info(f"Found {len(cameras)} camera(s):")
        for camera in cameras:
            logger.info(f"  {camera['device']}: {camera['name']} (ID: {camera['id']})")
    else:
        logger.warning("No cameras detected!")
    
    # 카메라 프로세서에 핸드 컨트롤러 전달
    camera_processor.hand_controller = hand_controller
    camera_processor.control_mode = control_mode
    
    # 프레임 제너레이터 반환
    return camera_processor.get_frame_generator()

# API 엔드포인트
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_info')
def camera_info():
    return jsonify({'cameras': get_camera_info()})

@app.route('/robot_status')
def robot_status():
    global robot, control_mode, hand_controller
    
    # 핸드 컨트롤러 상태 확인
    calibration_status = "not_started"
    if hand_controller:
        if hand_controller.calibration_done:
            calibration_status = "completed"
        elif hand_controller.calibration_mode:
            calibration_status = "in_progress"
    
    return jsonify({
        'connected': robot.is_connected if robot else False,
        'control_mode': control_mode,
        'calibration_status': calibration_status,
        'calibration_samples': len(hand_controller.calibration_positions) if hand_controller else 0
    })

@app.route('/robot_connect', methods=['POST'])
def robot_connect():
    global robot, keyboard_controller, hand_controller
    
    if not robot:
        try:
            # 로봇 초기화
            robot = ManipulatorRobot(KochRobotConfig())
            robot.connect()
            
            # 컨트롤러 생성
            keyboard_controller = KeyboardController(robot)
            hand_controller = HandController(robot)
            
            # 카메라 프로세서에 핸드 컨트롤러 전달
            if camera_processor:
                camera_processor.hand_controller = hand_controller
            
            return jsonify({'status': 'success', 'message': 'Robot connected successfully'})
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to connect robot: {str(e)}'})
    else:
        if not robot.is_connected:
            try:
                robot.connect()
                return jsonify({'status': 'success', 'message': 'Robot reconnected successfully'})
            except Exception as e:
                logger.error(f"Failed to reconnect robot: {e}")
                return jsonify({'status': 'error', 'message': f'Failed to reconnect robot: {str(e)}'})
        else:
            return jsonify({'status': 'info', 'message': 'Robot already connected'})

@app.route('/robot_disconnect', methods=['POST'])
def robot_disconnect():
    global robot, control_mode
    
    # 먼저 모든 제어 중지
    if control_mode != "none":
        set_control_mode_internal("none")
    
    if robot and robot.is_connected:
        try:
            robot.disconnect()
            return jsonify({'status': 'success', 'message': 'Robot disconnected successfully'})
        except Exception as e:
            logger.error(f"Failed to disconnect robot: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to disconnect robot: {str(e)}'})
    else:
        return jsonify({'status': 'info', 'message': 'Robot already disconnected'})

def set_control_mode_internal(mode):
    """제어 모드 설정 내부 함수"""
    global control_mode, keyboard_controller, hand_controller, camera_processor
    
    # 현재 제어 중지
    if control_mode == "keyboard" and keyboard_controller:
        keyboard_controller.stop()
    elif control_mode == "hand" and hand_controller:
        hand_controller.set_active(False)
    
    # 새 제어 모드 설정
    control_mode = mode
    
    # 카메라 프로세서에 전달
    if camera_processor:
        camera_processor.control_mode = mode
    
    # 새 제어 시작
    if mode == "keyboard" and keyboard_controller:
        keyboard_controller.start()
    elif mode == "hand" and hand_controller:
        hand_controller.set_active(True)
    
    logger.info(f"Control mode changed to: {mode}")

@app.route('/set_control_mode', methods=['POST'])
def set_control_mode():
    global robot
    
    if not robot or not robot.is_connected:
        return jsonify({
            'status': 'error', 
            'message': 'Robot is not connected. Please connect the robot first.'
        })
    
    try:
        data = request.json
        mode = data.get('mode', 'none')
        
        if mode not in ["keyboard", "hand", "none"]:
            return jsonify({
                'status': 'error', 
                'message': f'Invalid control mode: {mode}'
            })
        
        set_control_mode_internal(mode)
        
        return jsonify({
            'status': 'success', 
            'message': f'Control mode set to {mode}'
        })
    except Exception as e:
        logger.error(f"Failed to set control mode: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Failed to set control mode: {str(e)}'
        })

@app.route('/calibrate_keyboard', methods=['POST'])
def calibrate_keyboard():
    """키보드 컨트롤러 캘리브레이션 시작"""
    global keyboard_controller, control_mode
    
    if not keyboard_controller:
        return jsonify({
            'status': 'error',
            'message': 'Keyboard controller not available'
        })
    
    # 제어 모드를 키보드 제어로 변경
    if control_mode != "keyboard":
        set_control_mode_internal("keyboard")
    
    # 캘리브레이션 시작
    keyboard_controller.start_calibration()
    
    return jsonify({
        'status': 'success',
        'message': 'Keyboard calibration started. Please check console for instructions.'
    })

@app.route('/calibrate_hand', methods=['POST'])
def calibrate_hand():
    """손 컨트롤러 캘리브레이션 시작"""
    global hand_controller, control_mode
    
    if not hand_controller:
        return jsonify({
            'status': 'error',
            'message': 'Hand controller not available'
        })
    
    # 제어 모드를 손 제어로 변경
    if control_mode != "hand":
        set_control_mode_internal("hand")
    
    # 캘리브레이션 시작
    hand_controller.start_calibration()
    
    return jsonify({
        'status': 'success',
        'message': 'Hand calibration started. Move your hand throughout the workspace to calibrate positions.'
    })

@app.route('/logs', methods=['GET'])
def view_logs():
    """로그 파일 보기"""
    log_file = os.path.join(log_dir, 'robot_control.log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()
        # 최근 100줄만 표시
        logs = logs[-100:]
        return render_template('logs.html', logs=logs)
    else:
        return "Log file not found", 404

def main():
    try:
        logger.info("Starting Robot Hand Control server")
        # 카메라 지연 시간 줄이기 위해 스레드 우선순위 높이기 (Linux/macOS만 해당)
        if hasattr(os, 'nice'):
            try:
                os.nice(-10)  # 우선순위 높이기
            except:
                logger.warning("Failed to set process priority")
        
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리 작업
        global robot, keyboard_controller, control_mode, camera_processor
        
        if control_mode == "keyboard" and keyboard_controller:
            keyboard_controller.stop()
        
        if robot and robot.is_connected:
            robot.disconnect()
        
        if camera_processor:
            camera_processor.close()
        
        logger.info("Server shutdown complete")

if __name__ == '__main__':
    main()