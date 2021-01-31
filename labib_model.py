class LabibModel:
    """
    The model from Labib's paper
    """

    def __init__(self):
        self.coef_ = [0.040558,
                      -0.046875,
                      0.184550,
                      -0.205878,
                      0.019279,
                      0.134269,
                      -0.129523,
                      0.262576,
                      -0.309748,
                      0.996577,
                      -0.415382]

    def fit(self):
        pass

    def predict(self, x: list) -> list:
        return [self.coef_[0]
                + self.coef_[1] * row["RPCYD"]
                + self.coef_[2] * row["IOTD"]
                + self.coef_[3] * row["PRIME"]
                + self.coef_[4] * row["RIFM"]
                + self.coef_[5] * row["TI"]
                + self.coef_[6] * row["TD"]
                + self.coef_[7] * row["TR1"]
                + self.coef_[8] * row["TR2"]
                + self.coef_[9] * (row["ISE"] * row["TAX"])
                + self.coef_[10] * (row["ISE"] * row["ISEM"])
            for row in x]
