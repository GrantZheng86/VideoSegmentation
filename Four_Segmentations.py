import cv2
import Case_4_Processing

from time import sleep
import numpy as np

FILE_NAME = "New Videos/2-3.mp4"
CASE_Num = 4

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output.avi", fourcc, fps=30, frameSize=(616, 1080))
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0
    template = None

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret and counter == 1:
            frame = frame[140:965, :, :]
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            template = Case_4_Processing.findLandMarkFeature(frame)

        elif ret:
            top_annotation = frame[0:140, :, :]
            bottom_annotation = frame[965:, :, :]
            frame = frame[140:965, :, :]

            top_binary = Case_4_Processing.top_half_sesgmentation(frame)
            top_binary_bottom_contour = Case_4_Processing.findBottomContour(top_binary, True)
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            h = int(frame.shape[0] / 3)
            bottom_half = frame[h:, :, :]
            top_left, bottom_right, center, max_val = Case_4_Processing.match_template(bottom_half, template)
            if max_val < 0.6:
                cv2.putText(frame, "Unreliable Tracking", (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)
            top_left = (top_left[0], top_left[1] + h)
            bottom_right = (bottom_right[0], bottom_right[1] + h)
            center = (center[0], center[1] + h)
            distance = Case_4_Processing.findDistance(center, top_binary_bottom_contour)
            intersect = (int(center[0]), int(center[1] - distance))
            cv2.line(frame, center, intersect, (0, 0, 255), 3)
            frame = Case_4_Processing.annotate_frame(frame, (top_left, bottom_right))
            frame = np.vstack((top_annotation, frame, bottom_annotation))

            cv2.imshow("Template Matching", frame)
            videoWriter.write(frame)
            sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
