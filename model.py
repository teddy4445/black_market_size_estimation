import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


class MLmodel:
    """
    The ml model used to predict the data
    """

    # model names #
    DT = "dt"
    RF = "rf"
    LG = "lg"
    # end - model names #

    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 model_name: str = "dt"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model_name = model_name
        self.model = None
        self.parms = None
        self.best_model = None

    def fit(self,
            verbose=0):

        # ---> PICK THE MODEL TO TRAIN <--- #

        if self.model_name == MLmodel.DT:
            self.model = DecisionTreeRegressor()
            self.parms = {'max_depth': [5, 7, 9, 11],
                          'min_samples_split': [5, 10, 15],
                          'ccp_alpha': [0, 0.01, 0.05]}
        elif self.model_name == MLmodel.RF:
            self.model = RandomForestRegressor()
            self.parms = {'max_depth': [3, 5, 7, 9, 11, 13],
                          'min_samples_split': [5, 10, 15],
                          'ccp_alpha': [0, 0.01, 0.05, 0.1, 0.2]}
        elif self.model_name == MLmodel.LG:
            self.model = LinearRegression(positive=True)

        # ---> RUN THE MODEL <--- #

        if self.model_name in [MLmodel.DT, MLmodel.RF]:
            self.best_model = HalvingGridSearchCV(self.model,
                                                  self.parms,
                                                  verbose=verbose,
                                                  cv=5,
                                                  factor=2,
                                                  scoring=mean_absolute_error).fit(self.x_train, self.y_train)
        else:
            self.best_model = self.model

    def test_mae(self) -> float:
        return mean_absolute_error(self.best_model.predict(self.x_test), self.y_test)

    def test_mse(self) -> float:
        return mean_squared_error(self.best_model.predict(self.x_test), self.y_test)

    def predict(self, x: list = None) -> list:
        return self.best_model.predict(x if x is not None else self.x_train)

    def get_params(self) -> dict:
        if self.model_name == MLmodel.DT:
            return  self.best_model.get_params()
        elif self.model_name == MLmodel.RF:
            return  self.best_model.get_params()
        elif self.model_name == MLmodel.LG:
            # TODO: fix here
            return {}

    def get_feature_importance(self,
                               x_columns) -> dict:
        feature_importances = {}
        feature_importances_values = self.best_model.best_estimator_.feature_importances_
        feature_importances_names = list(x_columns)
        for i in range(len(feature_importances_values)):
            feature_importances[feature_importances_names[i]] = feature_importances_values[i]

        if self.model_name == MLmodel.DT:
            return  self.best_model.get_params()
        elif self.model_name == MLmodel.RF:
            return  self.best_model.get_params()
        elif self.model_name == MLmodel.LG:
            # TODO: fix here
            return {}


    def feature_importance(self) -> dict:
        if self.model_name == MLmodel.DT:
            return  self.best_model.feature_importance_
        elif self.model_name == MLmodel.RF:
            self.model = RandomForestRegressor()
            self.parms = {'max_depth': [3, 5, 7, 9, 11, 13],
                          'min_samples_split': [5, 10, 15],
                          'ccp_alpha': [0, 0.01, 0.05, 0.1, 0.2]}
        elif self.model_name == MLmodel.LG:
            self.model = LinearRegression(positive=True)

    def compare_predict(self) -> list:
        y_pred = list(self.best_model.predict(self.x_train))
        y_true = list(self.y_train)
        return [[y_pred[i], y_true[i]] for i in range(len(y_pred))]
