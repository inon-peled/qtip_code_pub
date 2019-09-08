from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import StandardScaler


class TrainedModel(object):
    def __init__(self, scaler_x, scaler_y, trained_model):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.trained_model = trained_model

    def predict(self, X_test):
        return self.scaler_y.inverse_transform(self.trained_model.predict(self.scaler_x.transform(X_test)).reshape(-1, 1))


class GaussianProcessRegressor(object):
    def __init__(self, **kwargs):
        self.gpr_init_kwargs = kwargs

    def fit(self, X, y):
        model = GPR(**self.gpr_init_kwargs)
        scaler_x = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        model.fit(scaler_x.transform(X), scaler_y.transform(y.reshape(-1, 1)))
        return TrainedModel(scaler_x, scaler_y, model)
