# ==========================================================
#ALL REQUIRED LIBRARIES
# ==========================================================
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

class HyperparameterTuner:

    def __init__(self):
        self.kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    # ==========================================================
    # DEFINE MODEL PARAMETER GRIDS
    # ==========================================================
    def get_model_params(self):

        return {
            "XGBoost": {
                "model": XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror"
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0]
                }
            },

            "GradientBoosting": {
                "model": GradientBoostingRegressor(
                    random_state=42
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            },

            "RandomForest": {
                "model": RandomForestRegressor(
                    random_state=42
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "max_features": ["sqrt", "log2"]
                }
            }
        }

    # ==========================================================
    # TUNE ALL MODELS
    # ==========================================================
    def run(self, X, y):

        results = []

        model_params = self.get_model_params()

        for model_name, config in model_params.items():

            print(f"\nTuning started for: {model_name}")

            search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="r2",
                cv=self.kfold,
                n_jobs=-1,
                verbose=1
            )

            search.fit(X, y)

            results.append({
                "model": model_name,
                "best_score": search.best_score_,
                "best_params": search.best_params_,
                "best_estimator": search.best_estimator_
            })

            print(f"Best Score: {search.best_score_}")
            print(f"Best Params: {search.best_params_}")

        return pd.DataFrame(results)