# ==========================================================
#ALL REQUIRED LIBRARIES:
# ==========================================================
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from xgboost import XGBRegressor
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42),
            "ExtraTrees": ExtraTreesRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42,n_estimators=100)
        }

    def evaluate_models(self, X, y):
        results = []

        for name, model in self.models.items():

            scores = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="r2"
            )

            results.append({
                "Model": name,
                "Mean_R2": scores.mean()
            })

        return pd.DataFrame(results).sort_values(
            by="Mean_R2",
            ascending=False
        )

    def run(self, X, y):
        report = self.evaluate_models(X, y)
        return report