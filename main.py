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

from plot_manager import PlotManager

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from model import MLmodel


class Main:
    """
    Manage the process of the repo
    """

    DATA_FOLDER = "data"
    PLOT_FOLDER = "plot"
    ANSWER_FILE_PATH = os.path.join(os.path.dirname(__file__),DATA_FOLDER, "answer.csv")
    DIRECTION_ANSWER_FILE_PATH = os.path.join(os.path.dirname(__file__),DATA_FOLDER, "answer_direction.csv")

    START_YEAR = 1995
    END_YEAR = 2019
    YEAR_COUNT = END_YEAR - START_YEAR + 1

    SAMPLES_STEP_SIZE = 3
    MAX_SAMPLES_TO_TRAIN = END_YEAR - START_YEAR - SAMPLES_STEP_SIZE
    MIN_SAMPLES_TO_TRAIN = MAX_SAMPLES_TO_TRAIN - 4 * SAMPLES_STEP_SIZE

    def __init__(self):
        pass

    @staticmethod
    def run(is_direction: bool = False):
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), Main.DATA_FOLDER))
        except Exception as error:
            pass
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), Main.PLOT_FOLDER))
        except Exception as error:
            pass

        raw_df = pd.read_csv("data.csv")

        dfs = Main.prepare_data()
        predict_col_name = "RCW"

        if is_direction:
            dfs = Main.fix_to_direction_data(dfs)

        if False:
            for key, df in dfs.items():
                print("Print analysis for {} data set".format(key))
                new_df = df
                new_df["pred"] = new_df["YEAR"].apply(lambda x: 0 if x < Main.END_YEAR - Main.SAMPLES_STEP_SIZE + 1 else 1)
                new_df.drop(["YEAR"], axis=1, inplace=True)
                PlotManager.analyze_data(df=df,
                                         model_type=key)

        if not os.path.exists(Main.ANSWER_FILE_PATH if not is_direction else Main.DIRECTION_ANSWER_FILE_PATH):
            with open(Main.ANSWER_FILE_PATH if not is_direction else Main.DIRECTION_ANSWER_FILE_PATH, "w") as answer_file:
                if is_direction:
                    answer_file.write("model_type,category,train_size,accuracy_avg,f1_avg,precision_avg,feature_importance,params\n")
                else:
                    answer_file.write("model_type,category,train_size,MSE_avg,MAE_avg,R2_avg,feature_importance,params\n")

        with open(Main.ANSWER_FILE_PATH if not is_direction else Main.DIRECTION_ANSWER_FILE_PATH, "a") as answer_file:
            summery_predict_df = dfs["all"][["YEAR", "RCW"]]
            for model_name in MLmodel.REGRESS_MODELS if not is_direction else MLmodel.CLASSIFY_MODELS:
                if True :
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
                            x = df.drop([predict_col_name, "YEAR"], axis=1)
                            y = df[predict_col_name]

                            predictor = MLmodel(x_train=x.head(train_size),
                                                y_train=y.head(train_size),
                                                x_test=x[train_size:train_size + Main.SAMPLES_STEP_SIZE],
                                                y_test=y[train_size:train_size + Main.SAMPLES_STEP_SIZE],
                                                model_name=model_name)
                            predictor.fit()

                            if is_direction:
                                mse = predictor.test_accuracy()
                                mae = predictor.test_f1()
                                r2 = predictor.test_auc()
                                print(
                                    "For model '{}' with {} train points and {} test points gets: accuracy = {:.7f}, f1 = {:.7f}, auc = {:.7f}".format(
                                        model_name, train_size, Main.SAMPLES_STEP_SIZE, mse, mae, r2
                                    ))
                            else:
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

                market_size_models = {}
                x_test = {}
                for key, df in dfs.items():
                    print("Working on df = {}".format(key))
                    x = df.drop([predict_col_name, "YEAR"], axis=1)
                    y = df[predict_col_name]

                    predictor = MLmodel(x_train=x,
                                        y_train=y,
                                        x_test=x,
                                        y_test=y,
                                        model_name=model_name)
                    predictor.fit()
                    market_size_models[key] = predictor
                    x_test[key] = x

                PlotManager.black_market_size(models=market_size_models,
                                              x=x_test,
                                              vop=list(raw_df["VOP"]),
                                              gdp=list(raw_df["GDP"]),
                                              model_name=model_name,
                                              start_year=Main.START_YEAR)

                if True:
                    feature_impt_dict = predictor.get_feature_importance(x_columns=list(x.columns))
                    # save a bar print
                    PlotManager.feature_importance(feature_name_val=feature_impt_dict,
                                                   model_name=model_name)

                    summery_predict_df["RCW_{}".format(model_name)] = predictor.predict(x=x)

                    answer_file.write("{},all_model_summary,{},{:.6f},{:.6f},{:.6f},{},{}\n"
                                      .format(model_name,
                                              Main.MAX_SAMPLES_TO_TRAIN,
                                              np.mean(scores_mse),
                                              np.mean(scores_mae),
                                              np.mean(scores_r2),
                                              feature_impt_dict,
                                              predictor.get_params()))

            y_list = []
            names = []
            for name in list(summery_predict_df):
                if "RCW" in name:
                    y_list.append(list(summery_predict_df[name]))
                    names.append(name.replace("labib", "Shami et al. 2021").replace("rfr", "proposed model"))

            PlotManager.plot_compare(x=[Main.START_YEAR + i for i in range(Main.END_YEAR - Main.START_YEAR + 1)],
                                     y_list=y_list,
                                     models_names=names,
                                     train_size=Main.MAX_SAMPLES_TO_TRAIN)

    @staticmethod
    def fix_to_direction_data(dfs) -> dict:
        y_col = "RCW"
        for key, df in dfs.items():
            for row_index in range(Main.END_YEAR - Main.START_YEAR + 1):
                if row_index < Main.END_YEAR - Main.START_YEAR:
                    dfs[key].ix[row_index, y_col] = 1 if dfs[key].iloc[row_index + 1][y_col] - dfs[key].iloc[row_index][y_col] > 0 else 0
                else:
                    dfs[key].ix[row_index, y_col] = 1
        return dfs

    @staticmethod
    def prepare_data() -> dict:
        """
        Prepare the data from the file as 4 data sets to work upon
        """
        df = pd.read_csv("data.csv")
        df = df.drop(['CW', 'ISEF1', 'ISEM1', 'ISE1', 'RPCYDCORRECTED', 'RPCYDN', 'VOP'], axis=1)
        df_no_tax = df.drop(['TAX', 'TD', 'TI', 'TR1', 'TR2', 'TOG'], axis=1)
        df_no_crime = df.drop(['RIFM'], axis=1)
        df_no_self_employ = df.drop(['ISEF', 'ISEM', 'ISE'], axis=1)
        # split into four data frames
        dfs = {"no_tax": df_no_tax,
               "no_crime": df_no_crime,
               "no_self_employ": df_no_self_employ,
               "all": df}
        return dfs


if __name__ == "__main__":
    Main.run(is_direction=False)
