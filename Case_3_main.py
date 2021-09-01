import cv2
import Case_3_processing
import numpy as np
import time
from PCAsegmentation import main_wrapper
from Case_4_Processing import findDistance

FILE_NAME = "New Videos/1-4.mp4"
TEMPLATE_TRACKING_FRAME_RATIO = 1 / 3

if __name__ == "__main__":
    state_list = main_wrapper(FILE_NAME)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Case4.avi", fourcc, fps=30, frameSize=(616, 825))

    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0
    template = None

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, :, :]
            cv2.putText(frame, str(state_list[counter-1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if counter == 1:
                template = Case_3_processing.extract_template(frame)

            else:
                bottom_contour = Case_3_processing.get_bottom_contour(frame, reduction=False)
                non_tracking_frame_height = int(frame.shape[0] * TEMPLATE_TRACKING_FRAME_RATIO)
                tracking_frame = frame[non_tracking_frame_height:, :, :]
                non_tracking_frame = frame[0:non_tracking_frame_height, :, :]
                top_left, bottom_right, center, max_val = Case_3_processing.match_template(tracking_frame, template)
                print(max_val)
                tracking_frame = Case_3_processing.annotate_frame(tracking_frame, (top_left, bottom_right))

                bottom_contour = Case_3_processing.correct_contour_path(bottom_contour, non_tracking_frame_height)
                if max_val < 0.81:
                    cv2.putText(frame, "Unreliable Tracking", (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)

                top_contour = Case_3_processing.find_top_bottom_contour(non_tracking_frame, reduction=False)
                center = (center[0], center[1]+non_tracking_frame_height)
                distance = findDistance(center, top_contour)
                intersect = (int(center[0]), int(center[1] - distance))
                cv2.line(frame, center, intersect, (0, 0, 255), 3)
                cv2.putText(frame, str(distance), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                encirclement = np.concatenate((np.flip(top_contour, 0), bottom_contour))
                frame = np.vstack((non_tracking_frame, tracking_frame))
                cv2.drawContours(frame, [encirclement], -1, (0, 255, 255), 1)
                cv2.imshow("Template Matching", frame)
                print(frame.shape)
                videoWriter.write(frame)

        time.sleep(1 / 40)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
