import cv2
import numpy
import time
import Case_2_Processing


FILE_NAME = "New Videos/1-2.mp4"

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output.avi", fourcc, fps=30, frameSize=(580, 825))
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, 0:580, :]
            frame = Case_2_Processing.get_bottom_two_parts(frame, counter)
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            videoWriter.write(frame)

        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()