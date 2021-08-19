import cv2
import PCAsegmentation

FILE_NAME = "NEW VIDEOS/Vid_1.mp4"

if __name__ == "__main__":
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            cropped_frame = frame[140:965, 0:580, :].copy()
            cv2.putText(frame, "Frame #{}".format(counter), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(cropped_frame, "Frame #{}".format(counter), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped", cropped_frame)
            cv2.imshow("Original Frame", frame)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()