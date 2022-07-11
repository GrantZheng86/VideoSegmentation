import math

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FOLDER_PATH = r'C:\Users\Grant\OneDrive - Colorado School of Mines\VideoSegmentation\ES & LM images & data Oct2021\ES & LM images & data Oct2021'
ORIGINAL_FILE = "ES & GS data Oct2021.xlsx"
GENERATED_FILE = r"C:\Users\Grant\OneDrive - Colorado School of Mines\VideoSegmentation\ES & LM images & data Oct2021\ES & LM images & data Oct2021\Images\cropped_imgs\information.csv"


def find_correlation(generated, reformatted_original):
    generated_list_rest = []
    original_list_rest = []
    generates_list_con = []
    original_list_con = []

    for i in range(len(generated)):
        curr_generated = generated.loc[i]
        name = curr_generated[0]
        data = curr_generated[1]

        if data > 0 and "22" not in name:
            name = name.split('.')[0] + '.png'
            original_data = reformatted_original.loc[name][0]

            if not math.isnan(original_data):

                if 'R' in name:
                    generated_list_rest.append(data)
                    original_list_rest.append(original_data)
                else:
                    generates_list_con.append(data)
                    original_list_con.append(original_data)

    return generated_list_rest, original_list_rest, generates_list_con, original_list_con


def match_generated(original):
    """
    This function converts the data format in the original excel to the one that's generated based on images
    :param original: A pandas dataframe directly from the excel data
    :return: A converted dataframe that matches the generated
    """
    ect1 = original['ECT1']
    ect2 = original['ECT2']
    ect3 = original['ECT3']

    ert1 = original['ERT1']
    ert2 = original['ERT2']
    ert3 = original['ERT3']

    id = original['ID']

    original_dict = {}

    for index, each_id in enumerate(id):
        curr_ect1 = ect1[index]
        curr_ect2 = ect2[index]
        curr_ect3 = ect3[index]

        curr_ert1 = ert1[index]
        curr_ert2 = ert2[index]
        curr_ert3 = ert3[index]

        ert1_name = "{:02d}ERT1.png".format(int(each_id))
        ert2_name = "{:02d}ERT2.png".format(int(each_id))
        ert3_name = "{:02d}ERT3.png".format(int(each_id))

        ect1_name = "{:02d}ECT1.png".format(int(each_id))
        ect2_name = "{:02d}ECT2.png".format(int(each_id))
        ect3_name = "{:02d}ECT3.png".format(int(each_id))

        original_dict[ert1_name] = curr_ert1
        original_dict[ert2_name] = curr_ert2
        original_dict[ert3_name] = curr_ert3

        original_dict[ect1_name] = curr_ect1
        original_dict[ect2_name] = curr_ect2
        original_dict[ect3_name] = curr_ect3

    return pd.DataFrame.from_dict(original_dict, orient='index')


def calculate_r_sq(computer, human):
    z = np.polyfit(computer, human, 1)
    f = np.poly1d(z)
    predicted_human = f(computer)
    ss_residual = sum(np.power(predicted_human - human, 2))
    ss_total = sum(np.power(human - np.average(human), 2))
    return 1 - ss_residual / ss_total


if __name__ == "__main__":
    original = os.path.join(FOLDER_PATH, ORIGINAL_FILE)
    # generated = os.path.join(FOLDER_PATH, GENERATED_FILE)
    generated = GENERATED_FILE

    original_copy = pd.read_excel(original)
    generated = pd.read_csv(generated)
    original = match_generated(original_copy)

    generated_list, original_list, generated_list_con, original_list_con = find_correlation(generated, original)
    r2 = r2_score(generated_list, original_list)

    correlation_matrix = np.corrcoef(generated_list, original_list)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    correlation_matrix_con = np.corrcoef(generated_list_con, original_list_con)
    correlation_xy_con = correlation_matrix_con[0, 1]
    r_squared_con = correlation_xy_con ** 2

    plt.plot(generated_list, original_list, 'ko', label="Relax")
    plt.plot(generated_list_con, original_list_con, 'ro', label="Contract")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')

    plt.title("Computer Analysis vs Human Annotation")
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(r_squared))
    plt.text(3.5, 4.5, "Correlation = {:.5f}".format(r_squared_con), color="red")

    # plt.title("Computer detected vs Human annotated")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel("Computer Analysis (CM)")
    plt.ylabel("Human Labeled (CM)")

    plt.figure(2)
    plt.plot(generated_list, original_list, 'ko', label="Relax")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')
    plt.title("Computer Analysis vs Human Annotation, Relaxed Case")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel("Computer Analysis (CM)")
    plt.ylabel("Human Labeled (CM)")
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(r_squared))
    plt.text(3.5, 4.6, "Number of samples {}".format(len(generated_list)))

    plt.figure(3)
    plt.plot(generated_list_con, original_list_con, 'ko', label="Contraction")
    plt.legend()
    plt.plot([1, 5], [1, 5], 'b--')
    plt.title("Computer Analysis vs Human Annotation, Contraction Case")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel("Computer Analysis (CM)")
    plt.ylabel("Human Labeled (CM)")
    plt.text(3.5, 4.8, "Correlation = {:.5f}".format(r_squared_con))
    plt.text(3.5, 4.6, "Number of samples {}".format(len(generated_list_con)))

    plt.show()

    print()
