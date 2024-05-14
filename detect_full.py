import cv2
import dlib
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Pose Detector
pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Face Detector
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define function to calculate angle between points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_leg_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = math.degrees(radians)
    if angle < 0:
        angle += 360
    return angle

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # Use the default camera

# Define the codec for video writing
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Initialize variables for counting
count_left = False
count_right = False
chin_up = False
l_leg = False
r_leg =False
count = 0
up_position = [False, False]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Convert frame to RGB for Pose Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(7, 10):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 4)
        chin_y = face_landmarks.part(8).y

    
    # Process frame for Pose Detection
    results = pose_detector.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get landmarks for both shoulders, elbows, and wrists
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
        right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]

        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        left_leg_angle = calculate_leg_angle(left_ankle, left_knee, left_hip)
        right_leg_angle = calculate_leg_angle(right_ankle, right_knee, right_hip)

        # # Display angles
        # # cv2.putText(frame, f"Left Angle: {int(left_angle)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # # cv2.putText(frame, f"Right Angle: {int(right_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # # Check if legs are straight
        if left_leg_angle > 160 and left_leg_angle < 190:
            l_leg = True
            # cv2.putText(frame, "Left Leg Straight", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # cv2.putText(frame, "Left Leg Not Straight", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            l_leg = False

        if right_leg_angle > 160 and right_leg_angle < 190:
            r_leg = True
            # cv2.putText(frame, "Right Leg Straight", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     # cv2.putText(frame, "Right Leg Not Straight", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #     r_leg = False
        
        cv2.line(frame, (int(left_index.x * frame.shape[1]), int(left_index.y * frame.shape[0])),
                     (int(right_index.x * frame.shape[1]), int(right_index.y * frame.shape[0])),
                     (255, 0, 0), 3)

        # Calculate angle of both arms
        if right_shoulder and right_elbow and right_wrist:
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            if right_angle > 120:
                up_position[0] = True
            elif up_position[0] and right_angle < 50:
                count_right = True
                up_position[0] = False

        if left_shoulder and left_elbow and left_wrist:
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if left_angle > 120:
                up_position[1] = True
            elif up_position[1] and left_angle < 50:
                count_left = True
                up_position[1] = False
            

        tangan_y = int(right_index.y * frame.shape[0])
        if  tangan_y > chin_y : 
            chin_up = True
            # print("lewat")

        if count_left and count_right and chin_up and r_leg and l_leg:
            count += 1
            count_left = count_right = False
        # if chin_up :
        
        # print(tangan_y)
        # print(chin_y)

        cv2.putText(frame, f"Right Arm Angle: {right_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Left Arm Angle: {left_angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Count: {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if r_leg:
            cv2.putText(frame, f"Right Leg Straight", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else :
            cv2.putText(frame, "Right Leg Not Straight", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if l_leg:
            cv2.putText(frame, f"Left Leg Straight", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else :
            cv2.putText(frame, "Left Leg Not Straight", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw Pose Landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display Frame
    cv2.imshow('Integrated Detection', frame)

    # Write frame to video output
    out.write(frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture, VideoWriter, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
