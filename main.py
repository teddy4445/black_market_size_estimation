import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from model import MLmodel


class Main:
    """
    Manage the process of the repo
    """

    DATA_FOLDER = "data"
    ANSWER_FILE_PATH = os.path.join(os.path.dirname(__file__),DATA_FOLDER, "answer.csv")

    SAMPLES_STEP_SIZE = 3
    MAX_SAMPLES_TO_TRAIN = 2019 - 1995 - SAMPLES_STEP_SIZE
    MIN_SAMPLES_TO_TRAIN = MAX_SAMPLES_TO_TRAIN - 4 * SAMPLES_STEP_SIZE

    def __init__(self):
        pass

    @staticmethod
    def run():
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), Main.DATA_FOLDER))
        except Exception as error:
            pass
        
        dfs = Main.prepare_data()
        predict_col_name = "RCW"

        if not os.path.exists(Main.ANSWER_FILE_PATH):
            with open(Main.ANSWER_FILE_PATH, "w") as answer_file:
                answer_file.write("model_type,category,train_size,MSE_avg,MAE_avg,R2_avg,feature_importance,params\n")

        with open(Main.ANSWER_FILE_PATH, "a") as answer_file:
            for model_name in [MLmodel.DT, MLmodel.RF, MLmodel.LG]:
                print("Start testing model: {}".format(model_name))
                scores_mae = []
                scores_mse = []
                scores_r2 = []
                predictor = None
                x = None
                for train_size in range(Main.MIN_SAMPLES_TO_TRAIN, Main.MAX_SAMPLES_TO_TRAIN + 1, Main.SAMPLES_STEP_SIZE):
                    print("Working on {} train points and {} test points".format(train_size, Main.SAMPLES_STEP_SIZE))
                    for key, df in dfs.items():
                        print("Working on df = {}".format(key))
                        x = df.drop([predict_col_name], axis=1)
                        y = df[predict_col_name]

                        predictor = MLmodel(x_train=x.head(train_size),
                                            y_train=y.head(train_size),
                                            x_test=x[train_size:train_size + Main.SAMPLES_STEP_SIZE],
                                            y_test=y[train_size:train_size + Main.SAMPLES_STEP_SIZE],
                                            model_name=model_name)
                        predictor.fit()

                        mse = predictor.test_mse()
                        mae = predictor.test_mae()
                        r2 = predictor.test_r2()
                        print(
                            "For model '{}' with {} train points and {} test points gets: mse = {:.7f}, mse = {:.7f}, r2 = {:.7f}".format(
                                model_name, train_size, Main.SAMPLES_STEP_SIZE, mse, mae, r2
                            ))
                        if key == "all":
                            scores_mae.append(mae)
                            scores_mse.append(mse)
                            scores_r2.append(r2)

                        answer_file.write("{},{},{:.6f},{:.6f},{:.6f},{},{}\n"
                                          .format(model_name,
                                                  key,
                                                  train_size,
                                                  mse,
                                                  mae,
                                                  r2,
                                                  predictor.get_feature_importance(x_columns=list(x.columns)),
                                                  predictor.get_params()))

                answer_file.write("{},all_model_summary,{},{:.6f},{:.6f},{:.6f},{},{}\n"
                                  .format(model_name,
                                          Main.MAX_SAMPLES_TO_TRAIN,
                                          np.mean(scores_mse),
                                          np.mean(scores_mae),
                                          np.mean(scores_r2),
                                          predictor.get_feature_importance(x_columns=list(x.columns)),
                                          predictor.get_params()))

    @staticmethod
    def prepare_data() -> dict:
        """
        Prepare the data from the file as 4 data sets to work upon
        """
        df = pd.read_csv("data.csv")
        df = df.drop(['YEAR', 'CW', 'ISEF1', 'ISEM1', 'ISE1', 'RPCYDCORRECTED', 'RPCYDN', 'VOP'], axis=1)
        df_no_tax = df.drop(['TAX', 'TD', 'TI', 'TR1', 'TR2', 'TOG'], axis=1)
        df_no_crime = df.drop(['RIFM'], axis=1)
        df_no_self_employ = df.drop(['ISEF', 'ISEM', 'ISE'], axis=1)
        # split into four data frames
        dfs = {"no_tax": df_no_tax,
               "no_crime": df_no_crime,
               "no_self_employ": df_no_self_employ,
               "all": df}
        return dfs

    @staticmethod
    def get_black_market_size(rcw_full_model: float,
                              rcw_without_tax_model: float,
                              rcw_without_crime_model: float,
                              rcw_without_self_employ_model: float,
                              vop: float,
                              gdp: float):
        delta1 = (rcw_full_model - rcw_without_tax_model) * vop
        delta2 = (rcw_full_model - rcw_without_crime_model) * 1000 * vop / gdp
        delta3 = (rcw_full_model - rcw_without_self_employ_model) * vop
        return delta1 + delta2 + delta3


if __name__ == "__main__":
    Main.run()
