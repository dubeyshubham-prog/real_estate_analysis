# ==========================================================
# REQUIRED LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import RFE


class FeatureSelector:
    # ==========================================================
    # CREATE LUXURY CATEGORY
    # ==========================================================
    def _categorize_luxury(self, score):

        if 0 <= score < 50:
            return "Low"

        elif 50 <= score < 150:
            return "Medium"

        elif 150 <= score <= 175:
            return "High"

        return None

    # ==========================================================
    # CREATE FLOOR CATEGORY
    # ==========================================================
    def _categorize_floor(self, floor):

        if 0 <= floor <= 2:
            return "Low Floor"

        elif 3 <= floor <= 10:
            return "Mid Floor"

        elif 11 <= floor <= 51:
            return "High Floor"

        return None

    # ==========================================================
    # PREPARE TRAINING DATA
    # ==========================================================
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        train_df = df.drop(
            columns=["society", "price_per_sqft"],
            errors="ignore"
        )

        train_df["luxury_category"] = (
            train_df["luxury_score"]
            .apply(self._categorize_luxury)
        )

        train_df["floor_category"] = (
            train_df["floorNum"]
            .apply(self._categorize_floor)
        )

        train_df.drop(
            columns=["floorNum", "luxury_score"],
            inplace=True,
            errors="ignore"
        )

        # HANDLE REMAINING MISSING VALUES
        numeric_cols = train_df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        categorical_cols = train_df.select_dtypes(
            include=["object"]
        ).columns

        for col in numeric_cols:
            train_df[col] = train_df[col].fillna(
                train_df[col].median()
            )

        for col in categorical_cols:
            train_df[col] = train_df[col].fillna(
                train_df[col].mode()[0]
            )

        return train_df

    # ==========================================================
    # LABEL ENCODE CATEGORICAL COLUMNS
    # ==========================================================
    def label_encode_data(self, train_df: pd.DataFrame) -> pd.DataFrame:

        data_label_encoded = train_df.copy()

        categorical_cols = (
            data_label_encoded
            .select_dtypes(include=["object"])
            .columns
        )

        for col in categorical_cols:

            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

            data_label_encoded[col] = encoder.fit_transform(
                data_label_encoded[[col]]
            )

        return data_label_encoded

    # ==========================================================
    # CREATE FEATURE IMPORTANCE REPORT
    # ==========================================================
    def create_feature_importance_report(self, data_label_encoded):

        x_label = data_label_encoded.drop("price", axis=1)
        y_label = data_label_encoded["price"]

        fi_df1 = (
            data_label_encoded
            .corr()["price"]
            .to_frame()
            .reset_index()
            .rename(
                columns={
                    "index": "feature",
                    "price": "corr_coeff"
                }
            )
        )

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        rf.fit(x_label, y_label)

        fi_df2 = pd.DataFrame({
            "feature": x_label.columns,
            "rf_importance": rf.feature_importances_
        })

        gb = GradientBoostingRegressor(
            random_state=42
        )
        gb.fit(x_label, y_label)

        fi_df3 = pd.DataFrame({
            "feature": x_label.columns,
            "gb_importance": gb.feature_importances_
        })

        x_train, x_test, y_train, y_test = train_test_split(
            x_label,
            y_label,
            test_size=0.2,
            random_state=42
        )

        rf_perm = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        rf_perm.fit(x_train, y_train)

        perm_importance = permutation_importance(
            rf_perm,
            x_test,
            y_test,
            n_repeats=10,
            random_state=42
        )

        fi_df4 = pd.DataFrame({
            "feature": x_label.columns,
            "permutation_importance": perm_importance.importances_mean
        })

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_label)

        lasso = Lasso(
            alpha=0.01,
            random_state=42
        )
        lasso.fit(x_scaled, y_label)

        fi_df5 = pd.DataFrame({
            "feature": x_label.columns,
            "lasso_coeff": lasso.coef_
        })

        estimator = RandomForestRegressor(
            random_state=42
        )

        selector = RFE(
            estimator,
            n_features_to_select=x_label.shape[1],
            step=1
        )

        selector.fit(x_label, y_label)

        fi_df6 = pd.DataFrame({
            "feature": x_label.columns[selector.support_],
            "rfe_score": selector.estimator_.feature_importances_
        })

        lin_reg = LinearRegression()
        lin_reg.fit(x_scaled, y_label)

        fi_df7 = pd.DataFrame({
            "feature": x_label.columns,
            "reg_coeffs": lin_reg.coef_
        })

        final_fi_df = (
            fi_df1
            .merge(fi_df2, on="feature")
            .merge(fi_df3, on="feature")
            .merge(fi_df4, on="feature")
            .merge(fi_df5, on="feature")
            .merge(fi_df6, on="feature")
            .merge(fi_df7, on="feature")
            .set_index("feature")
        )

        return final_fi_df

    # ==========================================================
    # FINAL FEATURE SELECTION
    # ==========================================================
    def select_final_features(self, data_label_encoded):

        x_label = data_label_encoded.drop("price", axis=1)
        y_label = data_label_encoded["price"]

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        scores_all = cross_val_score(
            rf,
            x_label,
            y_label,
            cv=5,
            scoring="r2"
        )

        drop_cols = [
            "pooja room",
            "study room",
            "others"
        ]

        export_df = x_label.drop(
            columns=drop_cols,
            errors="ignore"
        )

        scores_selected = cross_val_score(
            rf,
            export_df,
            y_label,
            cv=5,
            scoring="r2"
        )

        export_df["price"] = y_label

        return export_df, scores_all.mean(), scores_selected.mean()

    # ==========================================================
    # COMPLETE FEATURE SELECTION PIPELINE
    # ==========================================================
    def run(self, df: pd.DataFrame):

        train_df = self.prepare_training_data(df)

        data_label_encoded = self.label_encode_data(train_df)

        feature_importance_report = (
            self.create_feature_importance_report(data_label_encoded)
        )

        export_df, score_all, score_selected = (
            self.select_final_features(data_label_encoded)
        )

        return export_df, feature_importance_report, score_all, score_selected