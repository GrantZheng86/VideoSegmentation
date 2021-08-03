import cv2
import Case_3_processing

FILE_NAME = "New Videos/1-4.mp4"

if __name__ == "__main__":
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret and counter == 1:
            frame = frame[140:965, :, :]
            template = Case_3_processing.extract_template(frame)