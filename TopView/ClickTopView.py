import cv2
import numpy as np

# Global variables for mouse callback
points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    """Callback function to capture mouse clicks"""
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        # Draw circle on clicked point
        cv2.circle(img_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img_display, str(len(points)), (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw lines between points as we go
        if len(points) > 1:
            cv2.line(img_display, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        
        cv2.imshow('Select 4 Corners', img_display)
        print(f"Point {len(points)}: ({x}, {y})")

def calibrate_perspective(frame):
    """Calibrate by selecting 4 corners and return transform matrix"""
    global points, img_display
    points = []  # Reset points
    img_display = frame.copy()
    
    # Setup window and mouse callback
    cv2.namedWindow('Select 4 Corners')
    cv2.setMouseCallback('Select 4 Corners', mouse_callback)
    cv2.imshow('Select 4 Corners', img_display)
    
    print("\n" + "="*50)
    print("=== CALIBRATION ===")
    print("="*50)
    print("\nIMPORTANT: Click the OUTER CORNERS of your table/cup area")
    print("The further apart these corners, the better the transform!")
    print("\nClick in this order:")
    print("1. Top-left corner (furthest from camera on left)")
    print("2. Top-right corner (furthest from camera on right)")
    print("3. Bottom-right corner (closest to camera on right)")
    print("4. Bottom-left corner (closest to camera on left)")
    print("\nPress any key after selecting all 4 points...")
    print("="*50)
    
    # Wait for 4 points to be selected
    while len(points) < 4:
        cv2.waitKey(1)
    
    # Close the quadrilateral for visualization
    cv2.line(img_display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
    cv2.imshow('Select 4 Corners', img_display)
    
    # Wait for user to confirm
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert points to numpy array
    pts_src = np.array(points, dtype=np.float32)
    
    # Calculate the real-world aspect ratio from your clicks
    # This helps maintain proper proportions
    top_width = np.linalg.norm(pts_src[1] - pts_src[0])
    bottom_width = np.linalg.norm(pts_src[2] - pts_src[3])
    left_height = np.linalg.norm(pts_src[3] - pts_src[0])
    right_height = np.linalg.norm(pts_src[2] - pts_src[1])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    # Set output dimensions based on detected aspect ratio
    # Scale to reasonable size
    scale_factor = 800 / avg_width
    width = int(avg_width * scale_factor)
    height = int(avg_height * scale_factor)
    
    # Define destination points (perfect rectangle)
    pts_dst = np.array([
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height]       # bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    print("\nCalibration complete!")
    print(f"Detected aspect ratio: {avg_width:.0f} x {avg_height:.0f}")
    print(f"Output dimensions: {width} x {height}")
    
    return matrix, (width, height)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Capture a frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        cap.release()
        return
    
    # Calibrate and get transform matrix
    matrix, (width, height) = calibrate_perspective(frame)
    
    print("\n=== LIVE VIEW ===")
    print("Showing live top-down view...")
    print("Press 'q' to quit")
    print("Press 'r' to recalibrate")
    
    # Main loop - apply transform to live video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Apply perspective transform
        top_view = cv2.warpPerspective(frame, matrix, (width, height))
        
        # Display both views
        cv2.imshow('Original View', frame)
        cv2.imshow('Top-Down View', top_view)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Recalibrate
            print("\nRecalibrating...")
            matrix, (width, height) = calibrate_perspective(frame)
            print("Recalibration complete! Resuming live view...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()