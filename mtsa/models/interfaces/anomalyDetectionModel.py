from sklearn.base import BaseEstimator, OutlierMixin
from mtsa.models.interfaces.subject import Subject

class AnomalyDetectionModel(Subject):
    def __init__(self):
        super().__init__()
    
    def fit(self, **kwargs):
        pass
    
    def predict(self, X):
        pass

    def score(self, X):
        pass

    def score_samples(self, X):
        pass
