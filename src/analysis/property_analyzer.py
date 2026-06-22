from typing import ClassVar

import pandas as pd
import plotly.express as px

from src.common.exceptions import DataValidationError


class PropertyAnalyzer:
    """Create validated market summaries and interactive property charts."""

    DEFAULT_TEMPLATE: ClassVar[str] = "plotly_dark"
    CHART_CONFIG: ClassVar[dict[str, bool]] = {
        "displayModeBar": False,
        "responsive": True,
    }

    def __init__(self, template: str = DEFAULT_TEMPLATE) -> None:
        self.template = template

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        analysis_name: str,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise DataValidationError(
                f"{analysis_name} input must be a pandas DataFrame"
            )
        if dataframe.empty:
            raise DataValidationError(
                f"{analysis_name} input DataFrame cannot be empty"
            )

        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataValidationError(
                f"{analysis_name} requires missing columns: {missing}"
            )

    def _to_html(self, figure) -> str:
        """Render one embeddable chart without loading Plotly repeatedly."""
        return figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config=self.CHART_CONFIG,
        )

    @staticmethod
    def _sector_average_price(
        dataframe: pd.DataFrame,
        ascending: bool,
        top_n: int,
    ) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        return (
            dataframe.groupby("sector", as_index=False)["price"]
            .mean()
            .sort_values("price", ascending=ascending)
            .head(top_n)
        )

    def prepare_geo_data(
        self,
        dataframe: pd.DataFrame,
        coordinates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Parse sector coordinates and merge them with property data."""
        self._require_columns(
            dataframe,
            {"sector"},
            "Geographic analysis",
        )
        self._require_columns(
            coordinates,
            {"sector", "coordinates"},
            "Coordinate preparation",
        )

        coordinate_data = coordinates.copy()
        extracted = coordinate_data["coordinates"].astype("string").str.extract(
            r"(?P<latitude>-?\d+(?:\.\d+)?)\D+"
            r"(?P<longitude>-?\d+(?:\.\d+)?)"
        )

        if extracted.isna().any(axis=None):
            invalid_rows = int(extracted.isna().any(axis=1).sum())
            raise DataValidationError(
                f"Coordinate parsing failed for {invalid_rows} sectors"
            )

        coordinate_data[["latitude", "longitude"]] = extracted.astype(float)
        return dataframe.merge(
            coordinate_data[["sector", "latitude", "longitude"]],
            on="sector",
            how="inner",
            validate="many_to_one",
        )

    def property_type_distribution(self, dataframe: pd.DataFrame) -> str:
        self._require_columns(
            dataframe,
            {"property_type"},
            "Property-type distribution",
        )
        plot_data = (
            dataframe["property_type"]
            .value_counts()
            .rename_axis("property_type")
            .reset_index(name="count")
        )
        figure = px.pie(
            plot_data,
            names="property_type",
            values="count",
            title="Property Type Distribution",
            template=self.template,
            hole=0.45,
        )
        return self._to_html(figure)

    def top_expensive_sectors(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 10,
    ) -> str:
        self._require_columns(
            dataframe,
            {"sector", "price"},
            "Expensive-sector analysis",
        )
        plot_data = self._sector_average_price(
            dataframe,
            ascending=False,
            top_n=top_n,
        )
        figure = px.bar(
            plot_data,
            x="price",
            y="sector",
            orientation="h",
            title="Top Expensive Sectors by Average Price",
            template=self.template,
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        return self._to_html(figure)

    def top_affordable_sectors(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 10,
    ) -> str:
        self._require_columns(
            dataframe,
            {"sector", "price"},
            "Affordable-sector analysis",
        )
        plot_data = self._sector_average_price(
            dataframe,
            ascending=True,
            top_n=top_n,
        )
        figure = px.bar(
            plot_data,
            x="price",
            y="sector",
            orientation="h",
            title="Top Affordable Sectors by Average Price",
            template=self.template,
        )
        figure.update_layout(yaxis={"categoryorder": "total descending"})
        return self._to_html(figure)

    def area_vs_price(self, dataframe: pd.DataFrame) -> str:
        self._require_columns(
            dataframe,
            {"built_up_area", "price", "property_type"},
            "Area-versus-price analysis",
        )
        figure = px.scatter(
            dataframe,
            x="built_up_area",
            y="price",
            color="property_type",
            title="Built-up Area vs Price",
            template=self.template,
            opacity=0.65,
        )
        return self._to_html(figure)

    def bhk_price_analysis(
        self,
        dataframe: pd.DataFrame,
        maximum_bhk: int = 6,
    ) -> str:
        self._require_columns(
            dataframe,
            {"bedRoom", "price", "property_type"},
            "BHK price analysis",
        )
        if maximum_bhk < 1:
            raise ValueError("maximum_bhk must be at least 1")

        plot_data = dataframe.loc[dataframe["bedRoom"] <= maximum_bhk].copy()
        figure = px.box(
            plot_data,
            x="bedRoom",
            y="price",
            color="property_type",
            title="BHK-wise Price Range",
            template=self.template,
        )
        return self._to_html(figure)

    def luxury_price_impact(self, dataframe: pd.DataFrame) -> str:
        return self._category_price_chart(
            dataframe=dataframe,
            category="luxury_category",
            title="Luxury Category vs Average Price",
            analysis_name="Luxury-price analysis",
        )

    def furnishing_price_impact(self, dataframe: pd.DataFrame) -> str:
        return self._category_price_chart(
            dataframe=dataframe,
            category="furnishing_type",
            title="Furnishing Type vs Average Price",
            analysis_name="Furnishing-price analysis",
        )

    def _category_price_chart(
        self,
        dataframe: pd.DataFrame,
        category: str,
        title: str,
        analysis_name: str,
    ) -> str:
        self._require_columns(
            dataframe,
            {category, "price"},
            analysis_name,
        )
        plot_data = (
            dataframe.groupby(category, as_index=False, dropna=False)["price"]
            .mean()
        )
        figure = px.bar(
            plot_data,
            x=category,
            y="price",
            title=title,
            template=self.template,
        )
        return self._to_html(figure)

    def sector_map(
        self,
        dataframe: pd.DataFrame,
        coordinates: pd.DataFrame,
    ) -> str:
        self._require_columns(
            dataframe,
            {"sector", "price", "built_up_area"},
            "Sector-map analysis",
        )
        geo_data = self.prepare_geo_data(dataframe, coordinates)
        plot_data = (
            geo_data.groupby("sector", as_index=False)[
                ["price", "built_up_area", "latitude", "longitude"]
            ]
            .mean()
        )
        figure = px.scatter_map(
            plot_data,
            lat="latitude",
            lon="longitude",
            color="price",
            size="built_up_area",
            hover_name="sector",
            zoom=9,
            title="Sector-wise Price Heatmap",
            map_style="open-street-map",
            template=self.template,
        )
        return self._to_html(figure)

    def market_summary(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 5,
    ) -> dict[str, object]:
        self._require_columns(
            dataframe,
            {"price", "property_type", "sector"},
            "Market summary",
        )
        expensive = self._sector_average_price(
            dataframe,
            ascending=False,
            top_n=top_n,
        )
        affordable = self._sector_average_price(
            dataframe,
            ascending=True,
            top_n=top_n,
        )

        return {
            "total_properties": len(dataframe),
            "average_price_cr": round(float(dataframe["price"].mean()), 2),
            "property_type_distribution": (
                dataframe["property_type"].value_counts().to_dict()
            ),
            "top_expensive_sectors": dict(
                zip(
                    expensive["sector"],
                    expensive["price"].round(2),
                    strict=True,
                )
            ),
            "top_affordable_sectors": dict(
                zip(
                    affordable["sector"],
                    affordable["price"].round(2),
                    strict=True,
                )
            ),
        }
