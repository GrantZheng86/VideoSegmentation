import cv2
import numpy as np
import time
import Case_2_Processing
import Case_2_top_processing
import Case_2_middle_processing

FILE_NAME = "New Videos/2-2.mp4"

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Case2.avi", fourcc, fps=30, frameSize=(580, 825))

    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, 0:580, :]
            original_frame = frame.copy()
            # gray_frame = frame.copy()
            # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
            # gray_frame = gray_frame.astype(np.int64)
            # gray_frame = np.multiply(gray_frame, gray_frame)
            # gray_frame = gray_frame / np.amax(gray_frame) * 255
            # gray_frame = gray_frame.astype(np.uint8)
            # frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('Testing', frame)

            frame, height, valid_frame = Case_2_Processing.get_bottom_two_parts(frame, counter)

            print("Frame Number : {}".format(counter))
            if counter == 37:
                print()
            if valid_frame:
                # Partitioning the bottom frame
                bt_frame_start = frame.shape[0] - height
                bt_frame = original_frame[bt_frame_start:, :, :]
                fumer, pelvis = Case_2_Processing.partition_bottom_frame(bt_frame)

                if np.abs(pelvis[0, 0] - pelvis[-1, 0]) < 175:
                    print("Fumer Length {}, Pelvis Length {}".format(np.abs(pelvis[0, 0] - pelvis[-1, 0]),
                                                                     np.abs(fumer[0, 0] - fumer[-1, 0])))
                    cv2.putText(frame, "Unreliable segmentation", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                pelvis_contour = Case_2_Processing.fill_pelvis_contour(pelvis, bt_frame)
                fumer_contour = Case_2_Processing.fill_fumer_contour(fumer, bt_frame)
                pelvis_contour_adjusted = Case_2_Processing.height_adjustment(pelvis_contour, bt_frame_start)
                fumer_contour_adjusted = Case_2_Processing.height_adjustment(fumer_contour, bt_frame_start)
                cv2.polylines(frame, [pelvis_contour_adjusted], False, (0, 255, 255), 2)

                # partitioning the top frame
                # TODO: Check if the bottom contour needs to be moved up
                top_contour, segment_height = Case_2_top_processing.get_top_element_bottom_contour(original_frame)
                if top_contour is not None:
                    cv2.polylines(frame, [top_contour], False, (0, 255, 255), 2)

                    # partitioning the middle part
                    segment_height += 50
                    middle_frame = original_frame[segment_height:bt_frame_start, :, :]
                    middle_top_contour, middle_bottom_contour = Case_2_middle_processing.partition_middle_part(middle_frame)

                    middle_top_contour[:, 1] += segment_height
                    middle_bottom_contour[:, 1] += segment_height
                    cv2.polylines(frame, [middle_top_contour], False, (155, 0, 255), 1)
                    cv2.polylines(frame, [middle_bottom_contour], False, (155, 0, 255), 1)
                    # cv2.imshow("Middle frame", middle_frame)
                else:
                    cv2.putText(frame, "Top Segmentation Failed", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)





            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            videoWriter.write(frame)

        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
