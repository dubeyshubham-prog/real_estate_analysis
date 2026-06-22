import pandas as pd

from src.common.exceptions import DataValidationError


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    stage_name: str,
) -> None:
    """Validate a feature-engineering stage's DataFrame requirements."""
    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            f"{stage_name} input must be a pandas DataFrame"
        )
    if dataframe.empty:
        raise DataValidationError(
            f"{stage_name} input DataFrame cannot be empty"
        )

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataValidationError(
            f"{stage_name} requires missing columns: {missing}"
        )
