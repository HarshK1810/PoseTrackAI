# Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import collections  # For smoothing

# Mediapipe Utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to Calculate Angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize Video Capture and Pose Detection
cap = cv2.VideoCapture(0)
leg_angle_buffer = collections.deque(maxlen=5)  # Buffer for smoothing leg angle
breathing_buffer = collections.deque(maxlen=20)  # Buffer for breathing analysis

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Recolor Image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Pose Detection
        results = pose.process(image)

        # Recolor Image Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, c = image.shape  # Image dimensions for scaling

            # Get Coordinates
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate Angles
            leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            spine_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

            # Smooth the Leg Angle
            leg_angle_buffer.append(leg_angle)
            smoothed_leg_angle = sum(leg_angle_buffer) / len(leg_angle_buffer)

            # Feedback Logic
            feedback = []
            feedback_colors = []

            # Check Standing Leg Angle
            if 165 <= smoothed_leg_angle <= 180:
                feedback.append("Standing leg straight!")
                feedback_colors.append((0, 255, 0))  # Green
            else:
                feedback.append("Keep your standing leg straight.")
                feedback_colors.append((0, 0, 255))  # Red

            # Spine Alignment Feedback Logic
            if 165 <= spine_angle <= 180:  # Adjusted range for better flexibility
                feedback.append("Spine is aligned!")
                feedback_colors.append((0, 255, 0))  # Green
            else:
                feedback.append("Straighten your spine.")
                feedback_colors.append((0, 0, 255))  # Red

            # Check Left Leg Position
            if 90 < leg_angle < 140:
                feedback.append("Lower your left leg slightly.")
                feedback_colors.append((0, 0, 255))  # Red
            elif leg_angle > 140:
                feedback.append("Raise your left leg slightly.")
                feedback_colors.append((0, 0, 255))  # Red
            else:
                feedback.append("Left leg position is correct!")
                feedback_colors.append((0, 255, 0))  # Green

            # Breathing Analysis
            chest_center_y = (left_shoulder[1] + right_shoulder[1]) / 2  # Average y-coordinate of shoulders
            breathing_buffer.append(chest_center_y)

            if len(breathing_buffer) == breathing_buffer.maxlen:
                breathing_range = max(breathing_buffer) - min(breathing_buffer)
                if breathing_range > 0.02:  # Threshold for noticeable movement
                    feedback.append("Breathing is steady.")
                    feedback_colors.append((0, 255, 0))  # Green
                else:
                    feedback.append("Monitor your breathing pattern.")
                    feedback_colors.append((0, 0, 255))  # Red

            # Display Feedback on Frame
            for i, fb in enumerate(feedback):
                cv2.putText(image, fb, (10, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_colors[i], 2, cv2.LINE_AA)

            # Display Leg Angle
            cv2.putText(image, f"Leg Angle: {int(leg_angle)} degrees",
                        (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing landmarks: {e}")
            pass

        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the Frame
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
            break

# Release Resources
cap.release()
cv2.destroyAllWindows()
