from xgboost import XGBClassifier
from sklearn.utils import class_weight

class WeightedXGBClassifier:
    def __init__(self, class_weight=False, **xgb_params):
        self.class_weight = class_weight
        self.xgb_params = xgb_params
        self.model = XGBClassifier(**xgb_params)

    def fit(self, X, y):
        if self.class_weight:
            sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
        else:
            sample_weights = None

        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        # Include class_weight in params for grid search
        return {'class_weight': self.class_weight, **self.model.get_params(deep)}
    
    def set_params(self, **params):
        # Extract and store class_weight separately
        if 'class_weight' in params:
            self.class_weight = params.pop('class_weight')
        
        self.model.set_params(**params)
        return self