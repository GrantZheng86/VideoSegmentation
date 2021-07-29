import cv2
import numpy as np

video_name = 'Vid_1.mp4'
user_click_locations = []
templates = []
initial_frame = None;

template_size = 150;
frameSize = None;


def userClick():
    """
    This function handles the user click input on the image.
    """
    cap = cv2.VideoCapture(video_name)
    global initial_frame
    global frameSize;
    if cap.isOpened():
        _, initial_frame = cap.read();
        frameSize = initial_frame.shape;

        cv2.imshow("Select a region to track", initial_frame)
        cv2.setMouseCallback("Select a region to track", get_xy)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()


def annotate_and_extract_frame(frame, center_locations):
    """
    Draws rectangles on the image that user clicked, and
    extract template from the image. image templates
    are stored in global variable "templates"

    returns a frame that has been annotated with squares and
    numbers
    """
    global templates
    color = (0, 255, 0)
    thickness = 3;
    counter = 1;
    font = cv2.FONT_HERSHEY_SIMPLEX
    to_annotate_frame = np.copy(frame)

    for each_center in center_locations:
        x = each_center[0]
        y = each_center[1]

        pt1 = (int(x - template_size ), int(y - template_size / 2))
        pt2 = (int(x + template_size ), int(y + template_size / 2))
        curr_template = frame[pt1[1]:pt2[1], pt1[0]: pt2[0], :]
        templates.append(curr_template)

        cv2.rectangle(to_annotate_frame, pt1, pt2, color, thickness)
        to_annotate_frame = cv2.circle(to_annotate_frame, pt1, 8, color, -1)
        cv2.putText(to_annotate_frame, str(counter), (x - 10, y + 10), font, 1, color, 2)
        counter += 1

    return to_annotate_frame


def get_xy(event, x, y, flags, param):
    """
    This is the mouse click listener that is attach to the frame.
    this function stores the user click information to a global
    array called "user_click_locations
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y);
        user_click_locations.append((x, y))


def match_template(image, template):
    """
    This functions does the template matching for a given "image" with the
    the "template" given. And will return the opposite corners of a rectangle
    for annotation
    """
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)

    top_left = max_loc
    bottom_right = (top_left[0] + template_size*2, top_left[1] + template_size)

    return (top_left, bottom_right, max_val);


def annotate_frame(frame, corners, counter, jerk):
    """
    Draws a rectangle from a given pair of opposite corners,
    and the counter as a index that will be shown in the box
    """
    color = (0, 255, 0)
    thickness = 3;
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = corners[0][0]
    y = corners[1][1]

    cv2.rectangle(frame, corners[0], corners[1], color, thickness)
    cv2.putText(frame, str(counter), (x + int(template_size / 4), y - int(template_size / 4)), font, 1, color, 2)

    if jerk:
        cv2.putText(frame, "Drift Detected", (x + int(template_size / 4), y - int(template_size / 4)), font, 1, color, 2)

    return frame


def detect_sudden_motion(pt1, pt2, frame):
    (x, y, _) = frame.shape

    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    x_lim = x / 3
    y_lim = y / 3

    movement = pt1 - pt2
    movement = np.linalg.norm(movement)


    if (movement > x_lim) or (movement > y_lim):
        return True
    else:
        return False

if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output.avi", fourcc, fps=30, frameSize=(616, 1080))

    # prompt user for input
    userClick()
    user_click = annotate_and_extract_frame(initial_frame, user_click_locations)

    # shows the area clicked
    cv2.imshow("You clicked", user_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # shows if the template is the corners that has been clicked
    for temp in templates:
        cv2.imshow("Template", temp);
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Template matching and write to video file
    cap = cv2.VideoCapture(video_name)
    frame_counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_counter += 1

        if ret == True:
            counter = 1;

            # Template matching
            for each_template in templates:
                corners = match_template(frame, each_template)

                if frame_counter > 1:
                    jerk = detect_sudden_motion(prev_ul, corners[0], frame)
                else:
                    jerk = False
                prev_ul = corners[0]
                frame = annotate_frame(frame, corners, counter, jerk)
                frame = cv2.circle(frame, corners[0], 8, (0, 255, 0), -1)
                counter += 1;

            # Write to file
            cv2.imshow("Annotated", frame)
            videoWriter.write(frame)

        # type "q" to exit out of the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

