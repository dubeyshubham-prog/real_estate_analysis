# ==========================================================
#ALL REQUIRED LIBRARIES ARE HERE
# ==========================================================
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class FacilityRecommender:
    # ==========================================================
    # EXTRACT FACILITY LIST FROM STRING
    # ==========================================================
    def _extract_list(self, value):

        if pd.isna(value):
            return []

        return re.findall(r"'(.*?)'", str(value))

    # ==========================================================
    # PREPARE FACILITY SIMILARITY MATRIX
    # ==========================================================
    def fit(self, df: pd.DataFrame):

        self.df = df.copy()

        self.df["TopFacilities"] = (
            self.df["TopFacilities"]
            .apply(self._extract_list)
        )

        self.df["FacilitiesStr"] = (
            self.df["TopFacilities"]
            .apply(" ".join)
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["FacilitiesStr"]
        )

        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix,
            self.tfidf_matrix
        )

        return self

    # ==========================================================
    # RECOMMEND SIMILAR PROPERTIES
    # ==========================================================
    def recommend(self, property_name: str, top_n: int = 5):

        if property_name not in self.df["PropertyName"].values:
            raise ValueError("Property name not found")

        idx = self.df.index[
            self.df["PropertyName"] == property_name
        ].tolist()[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = sim_scores[1: top_n + 1]

        property_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        return pd.DataFrame({
            "PropertyName": self.df["PropertyName"].iloc[property_indices].values,
            "SimilarityScore": scores
        })

class PriceConfigRecommender:

    # ==========================================================
    # PARSE PRICE DETAILS COLUMN
    # ==========================================================
    def _parse_price_details(self, detail_str):

        try:
            details = json.loads(
                str(detail_str).replace("'", "\"")
            )
        except Exception:
            return {}

        extracted = {}

        for bhk, detail in details.items():

            extracted[f"building type_{bhk}"] = detail.get(
                "building_type"
            )

            area = detail.get("area", "")
            area_parts = area.split("-")

            if len(area_parts) == 1:
                try:
                    value = float(
                        area_parts[0]
                        .replace(",", "")
                        .replace(" sq.ft.", "")
                        .strip()
                    )
                    extracted[f"area low {bhk}"] = value
                    extracted[f"area high {bhk}"] = value
                except Exception:
                    extracted[f"area low {bhk}"] = None
                    extracted[f"area high {bhk}"] = None

            elif len(area_parts) == 2:
                try:
                    extracted[f"area low {bhk}"] = float(
                        area_parts[0]
                        .replace(",", "")
                        .replace(" sq.ft.", "")
                        .strip()
                    )

                    extracted[f"area high {bhk}"] = float(
                        area_parts[1]
                        .replace(",", "")
                        .replace(" sq.ft.", "")
                        .strip()
                    )
                except Exception:
                    extracted[f"area low {bhk}"] = None
                    extracted[f"area high {bhk}"] = None

            price_range = detail.get("price-range", "")
            price_parts = price_range.split("-")

            if len(price_parts) == 2:
                try:
                    low_price = (
                        price_parts[0]
                        .replace("₹", "")
                        .replace(" Cr", "")
                        .replace(" L", "")
                        .strip()
                    )

                    high_price = (
                        price_parts[1]
                        .replace("₹", "")
                        .replace(" Cr", "")
                        .replace(" L", "")
                        .strip()
                    )

                    extracted[f"price low {bhk}"] = float(low_price)
                    extracted[f"price high {bhk}"] = float(high_price)

                    if "L" in price_parts[0]:
                        extracted[f"price low {bhk}"] /= 100

                    if "L" in price_parts[1]:
                        extracted[f"price high {bhk}"] /= 100

                except Exception:
                    extracted[f"price low {bhk}"] = None
                    extracted[f"price high {bhk}"] = None

        return extracted

    # ==========================================================
    # CREATE STRUCTURED PRICE CONFIG DATA
    # ==========================================================
    def _create_price_config_df(self, df):

        configs = [
            "1 BHK",
            "2 BHK",
            "3 BHK",
            "4 BHK",
            "5 BHK",
            "6 BHK",
            "1 RK",
            "Land"
        ]

        data = []

        for _, row in df.iterrows():

            features = self._parse_price_details(
                row["PriceDetails"]
            )

            new_row = {
                "PropertyName": row["PropertyName"]
            }

            for config in configs:

                new_row[f"building type_{config}"] = features.get(
                    f"building type_{config}"
                )

                new_row[f"area low {config}"] = features.get(
                    f"area low {config}"
                )

                new_row[f"area high {config}"] = features.get(
                    f"area high {config}"
                )

                new_row[f"price low {config}"] = features.get(
                    f"price low {config}"
                )

                new_row[f"price high {config}"] = features.get(
                    f"price high {config}"
                )

            data.append(new_row)

        final_df = pd.DataFrame(data).set_index("PropertyName")

        if "building type_Land" in final_df.columns:
            final_df["building type_Land"] = (
                final_df["building type_Land"]
                .replace({"": "Land"})
            )

        return final_df

    # ==========================================================
    # FIT PRICE CONFIG RECOMMENDER
    # ==========================================================
    def fit(self, df: pd.DataFrame):

        self.df = df.copy()

        price_config_df = self._create_price_config_df(self.df)

        categorical_columns = (
            price_config_df
            .select_dtypes(include=["object"])
            .columns
            .tolist()
        )

        ohe_df = pd.get_dummies(
            price_config_df,
            columns=categorical_columns,
            drop_first=True
        )

        ohe_df = ohe_df.fillna(0)

        scaler = StandardScaler()

        self.normalized_df = pd.DataFrame(
            scaler.fit_transform(ohe_df),
            columns=ohe_df.columns,
            index=ohe_df.index
        )

        self.cosine_sim = cosine_similarity(
            self.normalized_df
        )

        return self

    # ==========================================================
    # RECOMMEND SIMILAR PROPERTIES
    # ==========================================================
    def recommend(self, property_name: str, top_n: int = 5):

        if property_name not in self.normalized_df.index:
            raise ValueError("Property name not found")

        idx = self.normalized_df.index.get_loc(
            property_name
        )

        sim_scores = list(
            enumerate(self.cosine_sim[idx])
        )

        sorted_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        top_indices = [
            item[0]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_scores = [
            item[1]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_properties = (
            self.normalized_df
            .index[top_indices]
            .tolist()
        )

        return pd.DataFrame({
            "PropertyName": top_properties,
            "SimilarityScore": top_scores
        })

class LocationRecommender:
    # ==========================================================
    # CONVERT DISTANCE TEXT TO METERS
    # ==========================================================
    def _distance_to_meters(self, distance_str):

        try:
            distance_str = str(distance_str)

            if "Km" in distance_str or "KM" in distance_str:
                return float(distance_str.split()[0]) * 1000

            elif "Meter" in distance_str or "meter" in distance_str:
                return float(distance_str.split()[0])

            return None

        except Exception:
            return None

    # ==========================================================
    # CREATE LOCATION DISTANCE MATRIX
    # ==========================================================
    def _create_location_df(self, df):

        location_matrix = {}

        for index, row in df.iterrows():

            distances = {}

            try:
                location_advantages = ast.literal_eval(
                    row["LocationAdvantages"]
                )
            except Exception:
                location_advantages = {}

            for location, distance in location_advantages.items():
                distances[location] = self._distance_to_meters(
                    distance
                )

            location_matrix[index] = distances

        location_df = pd.DataFrame.from_dict(
            location_matrix,
            orient="index"
        )

        location_df.index = df["PropertyName"]

        location_df.fillna(
            54000,
            inplace=True
        )

        return location_df

    # ==========================================================
    # FIT LOCATION RECOMMENDER
    # ==========================================================
    def fit(self, df: pd.DataFrame, drop_bad_row=True):

        self.df = df.copy()

        if drop_bad_row and 22 in self.df.index:
            self.df = self.df.drop(index=22)

        self.df = self.df.reset_index(drop=True)

        self.location_df = self._create_location_df(self.df)

        scaler = StandardScaler()

        self.location_df_normalized = pd.DataFrame(
            scaler.fit_transform(self.location_df),
            columns=self.location_df.columns,
            index=self.location_df.index
        )

        self.cosine_sim = cosine_similarity(
            self.location_df_normalized
        )

        return self

    # ==========================================================
    # RECOMMEND SIMILAR PROPERTIES
    # ==========================================================
    def recommend(self, property_name: str, top_n: int = 5):

        if property_name not in self.location_df_normalized.index:
            raise ValueError("Property name not found")

        idx = self.location_df_normalized.index.get_loc(
            property_name
        )

        sim_scores = list(
            enumerate(self.cosine_sim[idx])
        )

        sorted_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        top_indices = [
            item[0]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_scores = [
            item[1]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_properties = (
            self.location_df_normalized
            .index[top_indices]
            .tolist()
        )

        return pd.DataFrame({
            "PropertyName": top_properties,
            "SimilarityScore": top_scores
        })

class HybridRecommender:
    # ==========================================================
    # INITIALIZE ALL RECOMMENDERS
    # ==========================================================
    def __init__(
            self,
            facility_weight=30,
            price_weight=20,
            location_weight=8
    ):
        self.facility_weight = facility_weight
        self.price_weight = price_weight
        self.location_weight = location_weight

        self.facility_recommender = FacilityRecommender()
        self.price_recommender = PriceConfigRecommender()
        self.location_recommender = LocationRecommender()

    # ==========================================================
    # FIT ALL RECOMMENDERS
    # ==========================================================
    def fit(self, df: pd.DataFrame):

        self.df = df.copy()

        if 22 in self.df.index:
            self.df = self.df.drop(index=22)

        self.df = self.df.reset_index(drop=True)

        self.facility_recommender.fit(self.df)
        self.price_recommender.fit(self.df)

        self.location_recommender.fit(
            self.df,
            drop_bad_row=False
        )

        return self

    # ==========================================================
    # CREATE HYBRID SIMILARITY MATRIX
    # ==========================================================
    def _create_hybrid_similarity_matrix(self):

        facility_sim = self.facility_recommender.cosine_sim
        price_sim = self.price_recommender.cosine_sim
        location_sim = self.location_recommender.cosine_sim

        hybrid_sim = (
            self.facility_weight * facility_sim
            + self.price_weight * price_sim
            + self.location_weight * location_sim
        )

        return hybrid_sim

    # ==========================================================
    # FINAL HYBRID RECOMMENDATION
    # ==========================================================
    def recommend(self, property_name: str, top_n: int = 5):

        property_index = self.location_recommender.location_df_normalized.index

        if property_name not in property_index:
            raise ValueError("Property name not found")

        hybrid_sim = self._create_hybrid_similarity_matrix()

        idx = property_index.get_loc(property_name)

        sim_scores = list(
            enumerate(hybrid_sim[idx])
        )

        sorted_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        top_indices = [
            item[0]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_scores = [
            item[1]
            for item in sorted_scores[1: top_n + 1]
        ]

        top_properties = (
            property_index[top_indices]
            .tolist()
        )

        return pd.DataFrame({
            "PropertyName": top_properties,
            "SimilarityScore": top_scores
        })
