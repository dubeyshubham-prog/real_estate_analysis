import ast
import re
from typing import ClassVar

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features.validation import require_columns


class FurnishingExtractor:
    """Convert furnishing descriptions into deterministic count features."""

    NEGATION_PREFIX: ClassVar[str] = "No "

    @staticmethod
    def _parse_details(details: object) -> list[str]:
        if not isinstance(details, str) or not details.strip():
            return []
        try:
            parsed = ast.literal_eval(details)
        except (SyntaxError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    @staticmethod
    def _furnishing_name(item: str) -> str:
        return re.sub(r"^(?:No\s+|\d+\s+)", "", item).strip()

    def _get_furnishing_count(
        self,
        details: object,
        furnishing: str,
    ) -> int:
        items = self._parse_details(details)
        negative_value = f"{self.NEGATION_PREFIX}{furnishing}".casefold()

        for item in items:
            normalized = item.casefold()
            if normalized == negative_value:
                return 0

            match = re.fullmatch(
                rf"(\d+)\s+{re.escape(furnishing)}",
                item,
                flags=re.IGNORECASE,
            )
            if match:
                return int(match.group(1))
            if normalized == furnishing.casefold():
                return 1
        return 0

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"furnishDetails"},
            "Furnishing extraction",
        )
        engineered = dataframe.copy()

        furnishing_names = sorted(
            {
                self._furnishing_name(item)
                for details in engineered["furnishDetails"].dropna()
                for item in self._parse_details(details)
                if self._furnishing_name(item)
            }
        )

        for furnishing in furnishing_names:
            engineered[furnishing] = engineered["furnishDetails"].map(
                lambda details, name=furnishing: self._get_furnishing_count(
                    details,
                    name,
                )
            )
        return engineered


class FurnishingClusterEngineer:
    """Create stable unfurnished, semi-furnished, and furnished labels."""

    NON_FURNISHING_COLUMNS: ClassVar[set[str]] = {
        "property_type",
        "society",
        "sector",
        "price",
        "price_per_sqft",
        "area",
        "areaWithType",
        "bedRoom",
        "bathroom",
        "balcony",
        "additionalRoom",
        "address",
        "floorNum",
        "facing",
        "agePossession",
        "nearbyLocations",
        "furnishDetails",
        "features",
        "floor_num",
        "total_floors",
        "super_built_up_area",
        "built_up_area",
        "carpet_area",
        "study room",
        "servant room",
        "store room",
        "pooja room",
        "others",
    }

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        engineered = dataframe.copy()
        furnishing_columns = sorted(
            set(engineered.columns).difference(self.NON_FURNISHING_COLUMNS)
        )
        if not furnishing_columns:
            raise ValueError("No furnishing features were available for clustering")

        furnishing_data = engineered[furnishing_columns].fillna(0)
        scaled_data = StandardScaler().fit_transform(furnishing_data)
        raw_labels = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit_predict(scaled_data)

        intensity = furnishing_data.sum(axis=1)
        cluster_intensity = (
            pd.DataFrame({"cluster": raw_labels, "intensity": intensity})
            .groupby("cluster")["intensity"]
            .mean()
            .sort_values()
        )
        semantic_label_mapping = {
            raw_label: semantic_label
            for semantic_label, raw_label in enumerate(cluster_intensity.index)
        }
        engineered["furnishing_type"] = pd.Series(
            raw_labels,
            index=engineered.index,
        ).map(semantic_label_mapping)
        return engineered.drop(columns=furnishing_columns)
