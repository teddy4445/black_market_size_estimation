import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from labib_model import LabibModel


class MLmodel:
    """
    The ml model used to predict the data
    """

    # model names #
    DT = "dt"
    RF = "rf"
    LG = "lg"
    KNN = "knn"
    NN = "NN"
    SVR = "SVR"
    POLY2 = "poly2"
    POLY3 = "poly3"
    LABIB = "labib"

    MODELS = [LG, KNN, DT, RF]
    # end - model names #

    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 model_name: str = "dt"):
        self.x_train = [[float(val.replace(",", "") if isinstance(val, str) else val) for val in arr] for arr in list(x_train.values)]
        self.y_train = [float(val) for val in list(y_train.values)]
        self.x_test = [[float(val.replace(",", "") if isinstance(val, str) else val) for val in arr] for arr in list(x_test.values)]
        self.y_test = [float(val) for val in list(y_test.values)]

        self.model_name = model_name
        self.model = None
        self.parms = None
        self.best_model = None

    def fit(self):

        # ---> PICK THE MODEL TO TRAIN <--- #

        if self.model_name == MLmodel.DT:
            self.model = DecisionTreeRegressor()
            self.parms = {'max_depth': [5, 7, 9, 11],
                          'min_samples_split': [5, 10, 15],
                          'ccp_alpha': [0, 0.01, 0.05]}
        elif self.model_name == MLmodel.RF:
            self.model = RandomForestRegressor(n_estimators=20)
            self.parms = {'max_depth': [3, 5, 7, 9, 11, 13],
                          'min_samples_split': [5, 10, 15],
                          'ccp_alpha': [0, 0.01, 0.05, 0.1, 0.2]}
        elif self.model_name == MLmodel.LG:
            self.model = LinearRegression(positive=True)
        elif self.model_name == MLmodel.LABIB:
            self.model = LabibModel()
        elif self.model_name == MLmodel.KNN:
            self.model = KNeighborsRegressor()
            self.parms = {'n_neighbors': [3, 4, 5, 6, 7]}
        elif self.model_name == MLmodel.SVR:
            self.model = SVR(gamma='auto')
        elif self.model_name == MLmodel.NN:
            self.model = MLPRegressor(random_state=1, max_iter=500)

        # ---> RUN THE MODEL <--- #

        if self.model_name in [MLmodel.DT, MLmodel.RF, MLmodel.KNN]:
            self.best_model = GridSearchCV(self.model,
                                           self.parms,
                                           cv=5)
        else:
            self.best_model = self.model
        self.best_model.fit(self.x_train, self.y_train)

    def test_mae(self) -> float:
        return mean_absolute_error(y_true=self.y_test, y_pred=self.best_model.predict(self.x_test))

    def test_mse(self) -> float:
        return mean_squared_error(y_true=self.y_test, y_pred=self.best_model.predict(self.x_test))

    def test_r2(self) -> float:
        return r2_score(y_true=self.y_test, y_pred=self.best_model.predict(self.x_test))

    def predict(self, x: list = None) -> list:
        x = [[float(val.replace(",", "") if isinstance(val, str) else val) for val in arr] for arr in list(x.values)]
        return self.best_model.predict(x if x is not None else self.x_train)

    def get_feature_importance(self,
                               x_columns) -> dict:
        feature_importances = {}
        feature_importances_values = None
        feature_importances_names = list(x_columns)
        try:
            if self.model_name in [MLmodel.DT,  MLmodel.RF]:
                try:
                    feature_importances_values = self.best_model.feature_importances_
                except:
                    feature_importances_values = self.best_model.best_estimator_.feature_importances_
            elif self.model_name in [MLmodel.LG, MLmodel.LABIB]:
                feature_importances_values = [abs(val) for val in self.best_model.coef_]
                normalizer = sum(feature_importances_values)
                feature_importances_values = [val / normalizer for val in feature_importances_values]
            elif self.model_name in [MLmodel.KNN]:
                feature_importances_values = [1 / len(x_columns) for i in range(len(x_columns))]
            for i in range(len(feature_importances_values)):
                feature_importances[feature_importances_names[i]] = feature_importances_values[i]
        except:
            for i in range(len(feature_importances_names)):
                feature_importances[feature_importances_names[i]] = 0
        return feature_importances

    def get_params(self) -> dict:
        if self.model_name == MLmodel.DT:
            return self.best_model.get_params()
        elif self.model_name == MLmodel.RF:
            return self.best_model.get_params()
        elif self.model_name == MLmodel.LG:
            return self.best_model.get_params()

    def compare_predict(self) -> list:
        y_pred = list(self.best_model.predict(self.x_train))
        y_true = list(self.y_train)
        return [[y_pred[i], y_true[i]] for i in range(len(y_pred))]
