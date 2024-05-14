import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image)

    if results.pose_landmarks:
        # Get landmarks
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]

        # Draw line between left wrist and right wrist
        cv2.line(image, (int(left_wrist.x * image.shape[1]), int(left_wrist.y * image.shape[0])),
                 (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])),
                 (255, 0, 0), 3)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
