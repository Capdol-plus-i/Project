import cv2
import time

def stream_camera(device_num=8, width=640, height=480):
    # Open camera (using /dev/video8 as default)
    cap = cv2.VideoCapture(device_num)
    
    if not cap.isOpened():
        print(f"Cannot open camera device {device_num}!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print(f"Camera streaming started (Device: {device_num}, Resolution: {width}x{height})")
    print("Press 'q' key to exit")
    
    # Variables for FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Cannot read frame!")
                break
            
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add FPS text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Camera Stream', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nTerminated by user.")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera streaming ended.")

if __name__ == "__main__":
    # Get device number from user input
    try:
        device_input = input("Enter camera device number (default: 8): ")
        device_num = int(device_input) if device_input.strip() else 8
        
        width_input = input("Enter desired width resolution (default: 640): ")
        width = int(width_input) if width_input.strip() else 640
        
        height_input = input("Enter desired height resolution (default: 480): ")
        height = int(height_input) if height_input.strip() else 480
        
        stream_camera(device_num, width, height)
    
    except ValueError:
        print("Invalid input. Using default values.")
        stream_camera()