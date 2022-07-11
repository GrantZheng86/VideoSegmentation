import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILE_1_NAME = 'Ground_truth/DUS_Study_DataM1-10.xlsx'
FILE_2_NAME = 'Ground_truth/DUS_Study_DataM11-20.xlsx'
FILE_3_NAME = 'Ground_truth/DUS_Study_DataM21-30.xlsx'
LABEL_NAME = 'Patient_labels.xlsx'
GROUND_TRUTH = 'Summary_Ground_Truth.csv'
computer_measurement_csv_path = 'Regrouped_images/information.csv'
sheet_range = 10


def ground_truth_summary():
    files = [FILE_1_NAME, FILE_2_NAME, FILE_3_NAME]
    patient_num = 1
    to_save_dict = {}

    for file_name in files:
        for sheet in range(sheet_range):
            curr_data = pd.read_excel(file_name, sheet_name=sheet)
            resting = list(curr_data[9:12]["Unnamed: 1"])
            contracted = list(curr_data[9:12]["Unnamed: 2"])
            rest_key = "{}_R".format(patient_num)
            contract_key = "{}_C".format(patient_num)
            to_save_dict[rest_key] = resting
            to_save_dict[contract_key] = contracted
            patient_num += 1

    to_save_df = pd.DataFrame(to_save_dict)
    to_save_df.to_csv('Summary_Ground_Truth.csv')


if __name__ == '__main__':
    ground_truth_df = pd.read_csv(GROUND_TRUTH)
    label_df = pd.read_excel(LABEL_NAME)
    computer_measurements = pd.read_csv(computer_measurement_csv_path)

    measured_counts = len(list(computer_measurements['Img']))
    ground_truth_list = []
    measured_distance_list = []
    contracted_ground_truth_list = []
    relaxed_ground_truth_list = []
    contracted_measured_list = []
    relaxed_measured_list = []

    total_samples = 0
    for i in range(measured_counts):
        img_name = computer_measurements['Img'][i]
        measured_distance = computer_measurements['Distance'][i]
        img_name = img_name.split('.')[0]
        img_name = img_name.split(' ')

        labeled_patient = list(label_df['MS'])
        if len(img_name) == 4 and measured_distance > 0:
            patient_num = int(img_name[0][1:])
            patient_type = img_name[1]

            c_or_r = img_name[3][0]
            trial_number = int(img_name[3][1])

            if patient_num in labeled_patient:
                patient_index = labeled_patient.index(patient_num)
                if patient_type == label_df['Type'][patient_index]:
                    DUS_num = label_df['DUS'][patient_index]
                    column_name_ground_truth = '{}_{}'.format(DUS_num, c_or_r)
                    curr_ground_truth = ground_truth_df[column_name_ground_truth][trial_number-1]
                    ground_truth_list.append(curr_ground_truth)
                    measured_distance_list.append(measured_distance)
                    total_samples += 1

                    if c_or_r == 'C':
                        contracted_measured_list.append(measured_distance)
                        contracted_ground_truth_list.append(curr_ground_truth)
                    else:
                        relaxed_measured_list.append(measured_distance)
                        relaxed_ground_truth_list.append(curr_ground_truth)

    contracted_correlation_matrix = np.corrcoef(contracted_measured_list, contracted_ground_truth_list)
    contracted_correlation_xy = contracted_correlation_matrix[0, 1]
    contracted_r_sq = contracted_correlation_xy ** 2

    relaxed_correlation_matrix = np.corrcoef(relaxed_measured_list, relaxed_ground_truth_list)
    relaxed_correlation_xy = relaxed_correlation_matrix[0, 1]
    relaxed_r_sq = relaxed_correlation_xy ** 2

    plt.plot(contracted_measured_list, contracted_ground_truth_list, 'ko', label="Contracted")
    plt.plot(relaxed_measured_list, relaxed_ground_truth_list, 'ro', label="Relaxed")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')
    plt.title('Computer Measured vs Human Annotation')
    plt.xlabel('Computer Measured')
    plt.ylabel('Human Label')
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(contracted_r_sq))
    plt.text(3.5, 4.5, "Correlation = {:.5f}".format(relaxed_r_sq), color="red")
    plt.grid()

    plt.figure(2)
    plt.plot(relaxed_measured_list, relaxed_ground_truth_list, 'ko', label="Relax")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')
    plt.title("Computer Analysis vs Human Annotation, Relaxed Case")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel("Computer Analysis (CM)")
    plt.ylabel("Human Labeled (CM)")
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(relaxed_r_sq))
    plt.text(3.5, 4.6, "Number of samples {}".format(len(relaxed_measured_list)))

    plt.figure(3)
    plt.plot(contracted_measured_list, contracted_ground_truth_list, 'ko', label="Contraction")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')
    plt.title("Computer Analysis vs Human Annotation, Contraction Case")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel("Computer Analysis (CM)")
    plt.ylabel("Human Labeled (CM)")
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(contracted_r_sq))
    plt.text(3.5, 4.6, "Number of samples {}".format(len(contracted_measured_list)))
    plt.show()
