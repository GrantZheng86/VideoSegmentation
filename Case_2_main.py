import cv2
import numpy
import time
import Case_2_Processing

FILE_NAME = "New Videos/1-2.mp4"

if __name__ == "__main__":
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, :, :]
            Case_2_Processing.get_bottom_two_parts(frame)
            cv2.imshow("Frame", frame)

        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()