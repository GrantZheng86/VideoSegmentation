from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import numpy as np
import pandas as pd
import os


def scale_image(raw_img):
    # raw_img = cv2.imread(image_name)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    raw_img = raw_img[143:, 0:575]
    scaler = StandardScaler()
    scaler.fit(raw_img)
    scaled_img = scaler.transform(raw_img)
    # plt.imshow(raw_img, cmap='gray')
    # plt.show()
    # plt.imshow(scaled_img, cmap='gray')
    # plt.show()
    return raw_img


def perform_pca(scaled_img):
    pca_image = PCA(n_components=1)
    # pca_image = PCA(0.85)
    pca_image.fit(scaled_img)
    to_return = pca_image.transform(scaled_img)
    approx = pca_image.inverse_transform(to_return)
    combined = np.hstack((approx, scaled_img))
    # plt.imshow(combined, cmap='gray')
    # plt.show()
    return combined, to_return


def write2Video(video_name):
    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output.avi", fourcc, fps=30, frameSize=(1150, 937))
    pca_list = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = scale_image(frame)
            pac_img, pca_vec = perform_pca(frame)
            pca_list.append(pca_vec)
            pac_img = pac_img.astype(np.uint8)
            pac_img = cv2.cvtColor(pac_img, cv2.COLOR_GRAY2RGB)
            cv2.putText(pac_img, str(counter), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Processed", pac_img)
            videoWriter.write(pac_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return pca_list


def segment_pc(csv_file):
    changes = []
    # Low pass filter
    weights = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    # weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    window_size = len(weights)
    pcs = pd.read_csv(csv_file, index_col=0)
    l = len(pcs)

    for i in range(window_size, l, 1):
        curr_PC = pcs.iloc[i]
        curr_change = 0

        for j in range(1, window_size, 1):
            prev_PC = pcs.iloc[i - j]
            dist = np.linalg.norm(curr_PC - prev_PC)
            curr_change = weights[j] * dist

        changes.append(curr_change)

    threshed_change, rising_sig = calculate_z_score(changes)

    return threshed_change, rising_sig


def calculate_z_score(data_list):
    window_size = 20
    l = len(data_list)
    z_list = []
    threshed_sig = []
    thresh = 2
    counter = 0
    rising_list = []

    for i in range(window_size, l, 1):
        mu = np.average(data_list[i - window_size:i])
        sigma = np.std(data_list[i - window_size:i])
        x = data_list[i]

        curr_z = (x - mu) / sigma
        if curr_z >= thresh:
            threshed_sig.append(1)
        else:
            threshed_sig.append(-1)

        if counter >= 1:
            rising = (threshed_sig[counter] - threshed_sig[counter - 1]) > 0
            rising_list.append(rising)

        z_list.append(np.abs(curr_z))
        counter += 1

    plt.plot(z_list)
    plt.plot(threshed_sig)
    plt.show()

    return threshed_sig, rising_list


def write_state_to_video(video_name, state_change_list):
    cap = cv2.VideoCapture(video_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output_with_state.avi", fourcc, fps=30, frameSize=(1150, 937))
    counter = 0
    state_len = len(state_change_list)
    lag = total_frames - state_len
    state = True
    state_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:

            if counter >= lag and state_change_list[counter - lag - 1]:
                state = not state

            state_list.append(state)
            cv2.putText(frame, str(state), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Processed", frame)
            videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return state_list

def main_wrapper(video_name):
    pca_vec = write2Video(video_name)
    pca_vec = np.array(pca_vec)
    pca_vec = np.squeeze(pca_vec)
    pac_df = pd.DataFrame(pca_vec)

    pwd = os.getcwd()
    csv_name = "First_PC.csv"
    pac_df.to_csv(os.path.join(pwd, csv_name))

    _, state_change = segment_pc("First_PC.csv")
    state_list = write_state_to_video("output.avi", state_change)
    return state_list

if __name__ == "__main__":
    # scaled_img = scale_image('Vid_1.mp4 extracted.jpg')
    # perform_pca(scaled_img)
    # pca_vec = write2Video("New Videos/2-3.mp4")
    # pca_vec = np.array(pca_vec)
    # pca_vec = np.squeeze(pca_vec)
    # pac_df = pd.DataFrame(pca_vec)
    #
    # pwd = os.getcwd()
    # csv_name = "First_PC.csv"
    # pac_df.to_csv(os.path.join(pwd, csv_name))
    #
    # _, state_change = segment_pc("First_PC.csv")
    # write_state_to_video("output.avi", state_change)
    main_wrapper("New Videos/2-4.mp4")
