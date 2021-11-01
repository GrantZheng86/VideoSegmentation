import math

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FOLDER_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
              "ES & LM images & data Oct2021"
ORIGINAL_FILE = "ES & GS data Oct2021.xlsx"
GENERATED_FILE = "information.csv"

def find_correlation(generated, reformatted_original):

    generated_list = []
    original_list = []


    for i in range(len(generated)):
        curr_generated = generated.loc[i]
        name = curr_generated[0]
        data = curr_generated[1]

        if data != -1 and "22" not in name:
            original_data = reformatted_original.loc[name][0]

            if not math.isnan(original_data):
                generated_list.append(data)
                original_list.append(original_data)

    return generated_list, original_list



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


if __name__ == "__main__":
    original = os.path.join(FOLDER_PATH, ORIGINAL_FILE)
    generated = os.path.join(FOLDER_PATH, GENERATED_FILE)

    original_copy = pd.read_excel(original)
    generated = pd.read_csv(generated)
    original = match_generated(original_copy)

    generated_list, original_list = find_correlation(generated, original)
    r2 = r2_score(generated_list, original_list)

    correlation_matrix = np.corrcoef(generated_list, original_list)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2


    plt.plot(generated_list, original_list, 'ko')
    plt.title("Correltation = {}".format(r_squared))
    plt.show()

    print()
