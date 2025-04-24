import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

def main():
    # Page configuration to make it wider
    st.set_page_config(layout="wide")
    
    st.title("Hand Tracking Drawing App")
    
    # Create main columns for side-by-side layout
    camera_col, drawing_col = st.columns(2)
    
    # Initialize session state for clear button
    if 'clear_pressed' not in st.session_state:
        st.session_state.clear_pressed = False
    
    # Function to handle clear button press
    def handle_clear():
        st.session_state.clear_pressed = True
    
    with camera_col:
        st.header("Camera View")
        # Create a placeholder for the webcam feed
        frame_placeholder = st.empty()
        
        # Add controls under camera
        col_run, col_clear = st.columns(2)
        with col_run:
            run = st.checkbox("Start/Stop Webcam", value=False)
        with col_clear:
            # Use on_click to ensure the clear action is properly registered
            clear_button = st.button("Clear Canvas", on_click=handle_clear)
            
        # Color selection in a horizontal layout
        st.write("Select Drawing Color:")
        color_options = ["Black", "Blue", "Red", "Green", "Yellow", "White"]
        default_color = "Black"
        
        # Create radio buttons for better UX
        selected_color = st.radio(
            "Select Color", 
            color_options, 
            horizontal=True,
            index=color_options.index(default_color),
            label_visibility="collapsed"
        )
        
        # Line thickness
        line_thickness = st.slider("Line Thickness", 1, 10, 2)
    
    with drawing_col:
        st.header("Drawing Canvas")
        # Create a placeholder for the canvas
        canvas_placeholder = st.empty()
        
        # Notepad directly under canvas
        st.subheader("Notes")
        whiteboard_text = st.text_area("", height=150, label_visibility="collapsed")
        
        # Instructions
        with st.expander("Instructions", expanded=False):
            st.markdown("""
            ### How to Use:
            1. Start the webcam using the checkbox
            2. Select a color from the radio buttons
            3. Show your hand to the camera
            4. Extend your index finger to draw - drawing stops when you fold your finger
            5. Use the Clear button to erase everything
            6. Move your index finger to the color circles at the top of the camera view to change colors
            7. Use the notes area for additional information
            """)
    
    # Initialize MediaPipe with caching to improve performance
    @st.cache_resource
    def initialize_hand_tracking():
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        mpDraw = mp.solutions.drawing_utils
        return mpHands, hands, mpDraw
    
    mpHands, hands, mpDraw = initialize_hand_tracking()
    
    # Mapping colors
    color_map = {
        "Black": (0, 0, 0),
        "Blue": (255, 0, 0),
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Yellow": (0, 255, 255),
        "White": (255, 255, 255)
    }
    
    # Initialize canvas with session state to persist between reruns
    if 'paintWindow' not in st.session_state:
        st.session_state.paintWindow = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    if 'points' not in st.session_state:
        # Giving different arrays to handle colour points of different colour
        st.session_state.points = {color: [deque(maxlen=1024)] for color in color_options}
    
    if 'color_index' not in st.session_state:
        # Color indexes
        st.session_state.color_index = {color: 0 for color in color_options}
    
    # Get references to session state variables
    paintWindow = st.session_state.paintWindow
    points = st.session_state.points
    color_index = st.session_state.color_index
    
    # Handle clear button press - do this BEFORE the webcam loop
    if st.session_state.clear_pressed:
        for color_name in color_options:
            points[color_name] = [deque(maxlen=1024)]
            color_index[color_name] = 0
        paintWindow.fill(255)
        st.session_state.clear_pressed = False  # Reset flag
    
    # Flag to track if we're currently drawing
    is_drawing = False
    
    # Set current color to selected color
    current_color = selected_color
    
    # Initialize webcam
    cap = None
    
    # Optimization variable for debouncing
    last_draw_time = time.time()
    smoothed_position = None
    
    # Function to check if index finger is extended
    def is_index_finger_extended(landmarks):
        """
        Check if the index finger is extended based on hand landmarks.
        Returns true only if index finger is extended and other fingers are folded.
        """
        if len(landmarks) < 21:
            return False
        
        # Get coordinates for finger joints
        wrist = landmarks[0]
        index_mcp = landmarks[5]  # Index finger MCP joint
        index_pip = landmarks[6]  # Index finger PIP joint
        index_dip = landmarks[7]  # Index finger DIP joint
        index_tip = landmarks[8]  # Index finger tip
        
        # Get middle finger coordinates to check if it's folded
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        
        # Get ring finger coordinates
        ring_tip = landmarks[16]
        
        # Get pinky coordinates
        pinky_tip = landmarks[20]
        
        # Get thumb coordinates
        thumb_tip = landmarks[4]
        
        # Calculate distances
        # Check if index finger is extended (tip is farther from wrist than MCP)
        index_extended = distance(index_tip, wrist) > distance(index_mcp, wrist)
        
        # Check if other fingers are relatively folded (closer to palm)
        middle_folded = distance(middle_tip, middle_mcp) < distance(index_tip, index_mcp) * 0.8
        ring_folded = distance(ring_tip, wrist) < distance(index_tip, wrist) * 0.8
        pinky_folded = distance(pinky_tip, wrist) < distance(index_tip, wrist) * 0.8
        
        # Extended index should have relatively aligned segments
        index_straight = (distance(index_tip, index_dip) + distance(index_dip, index_pip)) > distance(index_tip, index_pip) * 0.85
        
        # Only want drawing with index finger, so check thumb is not too close to index
        thumb_away = distance(thumb_tip, index_tip) > distance(index_mcp, wrist) * 0.5
        
        # Return True if index is extended and conditions met
        return index_extended and index_straight and thumb_away
    
    # Helper function to calculate distance between two points
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    if run:
        cap = cv2.VideoCapture(0)
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Process frames only when running
        while run:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture image from webcam. Please check your connection.")
                break
            
            # Flip the frame horizontally for a more natural feel
            frame = cv2.flip(frame, 1)
            
            # Optimize by working with a smaller frame for processing
            frame_small = cv2.resize(frame, (320, 240))
            framergb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            result = hands.process(framergb_small)
            
            # Convert back to original resolution for drawing
            scale_factor_x = frame.shape[1] / frame_small.shape[1]
            scale_factor_y = frame.shape[0] / frame_small.shape[0]
            
            # Draw color indicators at the top
            y_pos = 40
            spacing = 80
            radius = 25
            
            # Draw color indicators
            for i, color_name in enumerate(color_options):
                color_rgb = color_map[color_name]
                x_pos = 50 + i * spacing
                
                # Make background for white color visible
                if color_name == "White":
                    cv2.circle(frame, (x_pos, y_pos), radius, (200, 200, 200), -1)
                
                # Highlight selected color
                if color_name == current_color:
                    # Selected color gets larger circle
                    cv2.circle(frame, (x_pos, y_pos), radius + 5, (0, 0, 0), 2)
                
                cv2.circle(frame, (x_pos, y_pos), radius, color_rgb, -1)
                cv2.circle(frame, (x_pos, y_pos), radius, (0, 0, 0), 2)
                
                # Add text labels for colors
                cv2.putText(frame, color_name[0], (x_pos - 5, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if color_name != "Black" else (255, 255, 255), 2)
            
            # Clear button indicator
            clear_x = 50 + len(color_options) * spacing
            cv2.circle(frame, (clear_x, y_pos), radius, (200, 200, 200), -1)
            cv2.circle(frame, (clear_x, y_pos), radius, (0, 0, 0), 2)
            cv2.putText(frame, "C", (clear_x - 5, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Variable to track if index finger is extended
            index_extended = False
            
            # Post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    # Get all landmarks as normalized coordinates first
                    for lm in handslms.landmark:
                        # Convert to pixel coordinates and scale back to original frame size
                        lmx = int(lm.x * frame_small.shape[1] * scale_factor_x)
                        lmy = int(lm.y * frame_small.shape[0] * scale_factor_y)
                        landmarks.append([lmx, lmy])
                    
                    # Draw landmarks on frames with thinner lines for better performance
                    mpDraw.draw_landmarks(
                        frame, 
                        handslms, 
                        mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
                    )
                
                # Check if we have a complete hand
                if len(landmarks) >= 21:
                    # Check if index finger is extended
                    index_extended = is_index_finger_extended(landmarks)
                    
                    # Get index finger position
                    fore_finger = (landmarks[8][0], landmarks[8][1])
                    center = fore_finger
                    
                    # Apply smoothing for better drawing - moving average
                    if smoothed_position is None:
                        smoothed_position = center
                    else:
                        # Interpolate between previous position and current for smoothness
                        smoothed_position = (
                            int(0.7 * smoothed_position[0] + 0.3 * center[0]),
                            int(0.7 * smoothed_position[1] + 0.3 * center[1])
                        )
                        center = smoothed_position
                    
                    # Draw circle at index finger tip - green if extended, red otherwise
                    circle_color = (0, 255, 0) if index_extended else (0, 0, 255)
                    cv2.circle(frame, center, 5, circle_color, -1)
                    
                    # Add status text
                    status_text = "Drawing Mode" if index_extended else "Not Drawing"
                    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, circle_color, 2)
                    
                    # Check if index finger is in color selection area
                    if center[1] <= 75:
                        # Stop drawing when in the toolbar area
                        is_drawing = False
                        
                        for i, color_name in enumerate(color_options):
                            x_pos = 50 + i * spacing
                            if x_pos - radius <= center[0] <= x_pos + radius:
                                if current_color != color_name:
                                    current_color = color_name
                                    selected_color = color_name  # Update the UI control
                        
                        # Check if user touched clear button
                        if clear_x - radius <= center[0] <= clear_x + radius:
                            for color_name in color_options:
                                points[color_name] = [deque(maxlen=1024)]
                                color_index[color_name] = 0
                            paintWindow.fill(255)
                    
                    # Draw on canvas when index finger is not in selection area
                    elif index_extended:  # Only draw if index finger is extended
                        current_time = time.time()
                        # Add debouncing to reduce points and improve performance
                        if current_time - last_draw_time > 0.01:  # 10ms debounce
                            if is_drawing:
                                # Continue drawing
                                points[current_color][color_index[current_color]].appendleft(center)
                                last_draw_time = current_time
                            else:
                                # Start a new line (create a new deque)
                                is_drawing = True
                                points[current_color].append(deque(maxlen=1024))
                                color_index[current_color] += 1
                                points[current_color][color_index[current_color]].appendleft(center)
                                last_draw_time = current_time
                    else:
                        # Index not extended, stop drawing
                        is_drawing = False
                else:
                    # Not enough landmarks - likely partial hand detection
                    is_drawing = False
            else:
                # Hand not detected - stop drawing
                is_drawing = False
                smoothed_position = None
            
            # Draw lines of all the colors on the canvas and frame
            for color_name in color_options:
                for i in range(len(points[color_name])):
                    for j in range(1, len(points[color_name][i])):
                        if points[color_name][i][j - 1] is None or points[color_name][i][j] is None:
                            continue
                        cv2.line(frame, points[color_name][i][j - 1], points[color_name][i][j], 
                                color_map[color_name], line_thickness)
                        cv2.line(paintWindow, points[color_name][i][j - 1], points[color_name][i][j], 
                                color_map[color_name], line_thickness)
            
            # Update the frames - use container_width instead of column_width
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            canvas_placeholder.image(paintWindow, channels="RGB", use_container_width=True)
            
            # Update the selected color from radio buttons
            if selected_color != current_color:
                current_color = selected_color
            
            # Cap the frame rate for performance balance
            time.sleep(0.03)  # ~33 FPS
            
            # Update the session state each frame
            st.session_state.paintWindow = paintWindow
            st.session_state.points = points
            st.session_state.color_index = color_index
    else:
        # Display a static image when webcam is not running
        frame_placeholder.image(np.ones((480, 640, 3), dtype=np.uint8) * 200, 
                            channels="RGB", use_container_width=True)
        canvas_placeholder.image(paintWindow, channels="RGB", use_container_width=True)
        st.warning("Webcam is turned off. Click Start to begin.")
    
    # Release webcam when stopped
    if cap is not None and not run:
        cap.release()

if __name__ == "__main__":
    main()