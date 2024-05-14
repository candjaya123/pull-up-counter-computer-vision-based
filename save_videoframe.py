import cv2

# Define the codec using VideoWriter_fourcc(), set the resolution and frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
res = (640, 480)
fps = 20.0

# Define the video writer
out = cv2.VideoWriter('output.avi', fourcc, fps, res)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if camera is opened
if not cap.isOpened():
    print("Unable to read camera feed")

while True:
    # Read each frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(1)
    if k%256 == 27:
        break

# Release the camera and output file
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
