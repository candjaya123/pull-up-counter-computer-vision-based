import cv2
import dlib
import mediapipe as mp
import numpy as np
import math

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

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    count_left = False
    count_right = False
    chin_up = False
    l_leg = False
    r_leg = False
    right_angle = 0
    left_angle = 0
    count = 0
    up_position = [False, False]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            for n in range(7, 10):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 255), 4)
            chin_y = face_landmarks.part(8).y

        results = pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

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

            if left_leg_angle > 160 and left_leg_angle < 190:
                l_leg = True
                l_leg_pos = "STRAIGHT"
            else:
                l_leg = False
                l_leg_pos = " NOT STRAIGHT"

            if right_leg_angle > 160 and right_leg_angle < 190:
                r_leg = True
                r_leg_pos = " STRAIGHT"

            else:
                r_leg = True
                r_leg_pos = " NOT STRAIGHT"

            cv2.line(frame, (int(left_index.x * frame.shape[1]), int(left_index.y * frame.shape[0])),
                         (int(right_index.x * frame.shape[1]), int(right_index.y * frame.shape[0])),
                         (255, 0, 0), 3)

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

            if count_left and count_right and chin_up and r_leg and l_leg:
                count += 1
                count_left = count_right = False

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(frame, f"Right Arm Angle: {right_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Left Arm Angle: {left_angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Count: {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Leg {l_leg_pos}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Leg {r_leg_pos}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Integrated Detection', frame)
        # cv2.imshow('bla', frame_rgb)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
