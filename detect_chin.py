import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        # for n in range(8, 9):
        #     x = face_landmarks.part(n).x
        #     y = face_landmarks.part(n).y
        #     cv2.circle(frame, (x, y), 1, (0, 255, 255), 4)
        #     cv2.putText(frame, f"y: {y}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        x = face_landmarks.part(9).x
        y = face_landmarks.part(9).y
        cv2.circle(frame, (x, y), 1, (0, 255, 255), 4)
        cv2.putText(frame, f"y: {y}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
