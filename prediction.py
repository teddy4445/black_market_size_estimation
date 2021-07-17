# library imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# prediction signal
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# project imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from plot_manager import PlotManager

from main import Main


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


rmse_score = make_scorer(rmse,
                         greater_is_better=False)


class Prediction:
    """
    Make predictions on the data
    """

    START_YEAR = 1995
    END_YEAR = 2019
    YEAR_COUNT = END_YEAR - START_YEAR + 1

    SAMPLES_STEP_SIZE = 3
    MAX_SAMPLES_TO_TRAIN = END_YEAR - START_YEAR - SAMPLES_STEP_SIZE
    MIN_SAMPLES_TO_TRAIN = MAX_SAMPLES_TO_TRAIN - 4 * SAMPLES_STEP_SIZE

    @staticmethod
    def run_signal():
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), "plot", "prediction"))
        except:
            pass

        df = Main.prepare_data()["all"]
        y = df["RCW"]
        y_test = y[Prediction.MAX_SAMPLES_TO_TRAIN:]
        y_train = y.head(Prediction.MAX_SAMPLES_TO_TRAIN)

        y_list = [list(y)]

        # fit model
        model = ARIMA(y_train)
        model = model.fit()
        # make prediction
        y_pred = model.predict(len(y_train) + 1, len(y_train) + len(y_test))

        mse_score_three_years = mean_squared_error(y_test, y_pred)
        mae_score_three_years = mean_absolute_error(y_test, y_pred)
        r2_three_years = r2_score(y_test, y_pred)
        print("ARIMA")
        print("MSE = {:.8f}, MSE = {:.8f}, R2 = {:.8f}".format(mse_score_three_years, mae_score_three_years,
                                                               r2_three_years))
        """
        y_pred = [list(model.forecast(i + 1))[0] for i in range(len(y_test))]
        mse_score_three_years = mean_squared_error(list(y_test), y_pred)
        r2_three_years = r2_score(y_test, y_pred)
        print("MSE: E = {:.8f} | SD = {:.8f}, R2: E = {:.8f} | SD = {:.8f}".format(np.mean(mse_score_three_years),
                                                                                   np.std(mse_score_three_years),
                                                                                   np.mean(r2_three_years),
                                                                                   np.std(r2_three_years)))
        """

        # fit model
        print("AutoReg")
        model = AutoReg(y_train, lags=1)
        model_fit = model.fit()


        y_list.append(model_fit.predict(1, len(y_train) + len(y_test)))

        # make prediction
        y_pred = model_fit.predict(len(y_train) + 1, len(y_train) + len(y_test))
        mse_score_three_years = mean_squared_error(y_test, y_pred)
        mae_score_three_years = mean_absolute_error(y_test, y_pred)
        r2_three_years = r2_score(y_test, y_pred)
        print("MSE = {:.8f}, MSE = {:.8f}, R2 = {:.8f}".format(mse_score_three_years, mae_score_three_years,
                                                               r2_three_years))
        y_pred = [list(model_fit.predict(len(y_train) + i + 1, len(y_train) + i + 1))[0] for i in range(len(y_test))]
        mse_score_three_years = mean_squared_error(list(y_test), y_pred)
        r2_three_years = r2_score(y_test, y_pred)
        print("MSE: E = {:.8f} | SD = {:.8f}, R2: E = {:.8f} | SD = {:.8f}".format(np.mean(mse_score_three_years),
                                                                                   np.std(mse_score_three_years),
                                                                                   np.mean(r2_three_years),
                                                                                   np.std(r2_three_years)))
        PlotManager.plot_compare(x=[Main.START_YEAR + i for i in range(Main.END_YEAR - Main.START_YEAR + 1)],
                                 y_list=y_list,
                                 models_names=["rcw_history", "rcw_Proposed model"],
                                 train_size=Main.MAX_SAMPLES_TO_TRAIN)
    @staticmethod
    def run():
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), "plot", "prediction"))
        except:
            pass

        dfs = Main.prepare_data()
        for key, df in dfs.items():
            print("Working on df = {}".format(key))
            y = df["RCW"]
            x = df.drop(["RCW", "YEAR"], axis=1)
            x_train = x.head(Prediction.MAX_SAMPLES_TO_TRAIN)
            y_test = y[Prediction.MAX_SAMPLES_TO_TRAIN:]
            y_train = y.head(Prediction.MAX_SAMPLES_TO_TRAIN)
            x_test = x[Prediction.MAX_SAMPLES_TO_TRAIN:]

            models = []
            models.append(('Shami et al.', LinearRegression()))
            models.append(('Proposed model', RandomForestRegressor(max_depth=5,
                                                                   min_samples_split=3,
                                                                   ccp_alpha=0.01,
                                                                   n_estimators=100)))  # Ensemble method - collection of many decision trees
            # Evaluate each model in turn
            results = []
            names = []
            for name, model in models:
                # TimeSeries Cross validation
                tscv = TimeSeriesSplit(n_splits=3)

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                mse_score = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                scale = 1000000 if "Shami" in name else 731754
                results.append((mse_score, scale * mse_score * (1 - r2)))
                names.append(name)
                # print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                try:
                    imp = model.feature_importances_
                    features = x_train.columns
                    indices = np.argsort(imp)
                    plt.title('Feature Importances')
                    plt.barh(range(len(indices)), imp[indices], color='b', align='center')
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Relative Importance')
                    plt.savefig(os.path.join(os.path.dirname(__file__), "plot", "prediction",
                                             "feature_importance_{}_for_df_{}.png".format(name, key)))
                    plt.close()
                except Exception as error:
                    pass

            # Compare Algorithms
            plt.boxplot(results, labels=names)
            plt.ylabel("$10^{-5}$ Mean square error (3 years)")
            plt.xlabel("Model")
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 2 decimal places
            plt.savefig(os.path.join(os.path.dirname(__file__), "plot", "prediction",
                                     "algorithm_comparision_for_df_{}.png".format(key)))
            plt.close()

            model = RandomForestRegressor()
            param_search = {
                'n_estimators': [3, 5, 7, 10, 15, 20],
                'max_features': ['auto', 'log2'],
                'max_depth': [i for i in range(3, 10)]
            }
            tscv = TimeSeriesSplit(n_splits=3)
            gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
            gsearch.fit(x_train, y_train)
            best_score = gsearch.best_score_
            best_model = gsearch.best_estimator_

            # check best model on test data
            y_true = y_test.values
            y_pred = best_model.predict(x_test)
            Prediction.regression_results(y_true, y_pred)

            imp = best_model.feature_importances_
            features = x_train.columns
            indices = np.argsort(imp)
            plt.barh(range(len(indices)), imp[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig(os.path.join(os.path.dirname(__file__), "plot", "prediction",
                                     "best_model_importance_for_df_{}.png".format(key)))
            plt.close()

    @staticmethod
    def regression_results(y_true, y_pred):
        # Regression metrics
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        print('explained_variance: ', round(explained_variance, 4))
        print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
        print('r2: ', round(r2, 4))
        print('MAE: ', round(mean_absolute_error, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(np.sqrt(mse), 4))


if __name__ == '__main__':
    # Prediction.run()
    Prediction.run_signal()
