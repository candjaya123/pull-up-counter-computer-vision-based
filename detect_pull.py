import cv2
import mediapipe as mp
import numpy as np
# import math

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Initialize variables
count_left = False
count_right = False
count = 0
up_position = [False, False]

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
    
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get landmarks for both shoulders, elbows, and wrists
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle of both arms
            def calculate_angle(a,b,c):
                a = np.array(a) # First
                b = np.array(b) # Mid
                c = np.array(c) # End

                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)


                if angle >180.0:
                    angle = 360-angle

                return angle 

            if right_shoulder and right_elbow and right_wrist:
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                if right_angle > 120:
                    up_position[0] = True
                elif up_position[0] and right_angle < 50:
                    count_right = True
                    up_position[0] = False

                # Display the angle for the right arm
                cv2.putText(frame, f"Right Arm Angle: {right_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if left_shoulder and left_elbow and left_wrist:
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                if left_angle > 120:
                    up_position[1] = True
                elif up_position[1] and left_angle < 50:
                    count_left = True
                    up_position[1] = False

                # Display the angle for the left arm
                cv2.putText(frame, f"Left Arm Angle: {left_angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if count_left and count_right:
                count += 1
                count_left = count_right = False

            # Display the count on the frame
            cv2.putText(frame, f"Count: {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            

        cv2.imshow('Pull-up Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()