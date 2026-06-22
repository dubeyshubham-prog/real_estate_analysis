import ast
from typing import Any

import numpy as np
import pandas as pd

from src.common.exceptions import DataValidationError


def parse_literal(value: object, expected_type: type) -> Any:
    """Safely parse a Python-literal string into the expected container type."""
    if isinstance(value, expected_type):
        return value
    if not isinstance(value, str) or not value.strip():
        return expected_type()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return expected_type()
    return parsed if isinstance(parsed, expected_type) else expected_type()


def validate_recommendation_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid header rows and enforce unique project names."""
    required = {
        "PropertyName",
        "TopFacilities",
        "PriceDetails",
        "LocationAdvantages",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise DataValidationError(
            "Recommendation data is missing columns: "
            + ", ".join(sorted(missing))
        )

    valid = dataframe.copy()
    valid["PropertyName"] = valid["PropertyName"].astype("string").str.strip()
    valid = valid.loc[
        valid["PropertyName"].notna()
        & valid["PropertyName"].ne("")
        & valid["PropertyName"].ne("PropertyName")
    ].copy()
    if valid["PropertyName"].duplicated().any():
        duplicates = valid.loc[
            valid["PropertyName"].duplicated(),
            "PropertyName",
        ].tolist()
        raise DataValidationError(
            "Duplicate recommendation property names: "
            + ", ".join(duplicates)
        )
    return valid.reset_index(drop=True)


def top_similar(
    property_names: pd.Index,
    similarity_matrix: np.ndarray,
    property_name: str,
    top_n: int,
) -> pd.DataFrame:
    """Return top non-self similarities from an aligned matrix."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if property_name not in property_names:
        raise ValueError(f"Property not found: {property_name}")

    index = property_names.get_loc(property_name)
    scores = similarity_matrix[index]
    candidate_indexes = np.argsort(scores)[::-1]
    candidate_indexes = candidate_indexes[candidate_indexes != index][:top_n]
    return pd.DataFrame(
        {
            "PropertyName": property_names[candidate_indexes],
            "SimilarityScore": scores[candidate_indexes],
        }
    ).reset_index(drop=True)
