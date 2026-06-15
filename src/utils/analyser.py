import pandas as pd
import plotly.express as px

class PropertyAnalyzer:
    def __init__(self):
        self.template = "plotly_dark"

    # ==========================================================
    # PREPARE GEO DATA FOR MAP
    # ==========================================================
    def prepare_geo_data(self, df, latlong):

        latlong = latlong.copy()

        latlong["latitude"] = (
            latlong["coordinates"]
            .str.split(",")
            .str.get(0)
            .str.split("°")
            .str.get(0)
            .astype(float)
        )

        latlong["longitude"] = (
            latlong["coordinates"]
            .str.split(",")
            .str.get(1)
            .str.split("°")
            .str.get(0)
            .astype(float)
        )

        new_df = df.merge(
            latlong,
            on="sector",
            how="inner"
        )

        return new_df

    # ==========================================================
    # PROPERTY TYPE DISTRIBUTION
    # ==========================================================
    def property_type_distribution(self, df):

        plot_df = (
            df["property_type"]
            .value_counts()
            .reset_index()
        )

        plot_df.columns = [
            "property_type",
            "count"
        ]

        fig = px.pie(
            plot_df,
            names="property_type",
            values="count",
            title="Property Type Distribution",
            template=self.template,
            hole=0.45
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # TOP EXPENSIVE SECTORS
    # ==========================================================
    def top_expensive_sectors(self, df, top_n=10):

        plot_df = (
            df.groupby("sector")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            plot_df,
            x="price",
            y="sector",
            orientation="h",
            title="Top Expensive Sectors by Average Price",
            template=self.template
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # TOP AFFORDABLE SECTORS
    # ==========================================================
    def top_affordable_sectors(self, df, top_n=10):

        plot_df = (
            df.groupby("sector")["price"]
            .mean()
            .sort_values(ascending=True)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            plot_df,
            x="price",
            y="sector",
            orientation="h",
            title="Top Affordable Sectors by Average Price",
            template=self.template
        )

        fig.update_layout(
            yaxis={"categoryorder": "total descending"}
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # AREA VS PRICE SCATTER
    # ==========================================================
    def area_vs_price(self, df):

        fig = px.scatter(
            df,
            x="built_up_area",
            y="price",
            color="property_type",
            title="Built-up Area vs Price",
            template=self.template,
            opacity=0.65
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # BHK PRICE ANALYSIS
    # ==========================================================
    def bhk_price_analysis(self, df):

        temp_df = df[df["bedRoom"] <= 6]

        fig = px.box(
            temp_df,
            x="bedRoom",
            y="price",
            color="property_type",
            title="BHK-wise Price Range",
            template=self.template
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # LUXURY CATEGORY PRICE IMPACT
    # ==========================================================
    def luxury_price_impact(self, df):

        plot_df = (
            df.groupby("luxury_category")["price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            plot_df,
            x="luxury_category",
            y="price",
            title="Luxury Category vs Average Price",
            template=self.template
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # FURNISHING TYPE PRICE IMPACT
    # ==========================================================
    def furnishing_price_impact(self, df):

        plot_df = (
            df.groupby("furnishing_type")["price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            plot_df,
            x="furnishing_type",
            y="price",
            title="Furnishing Type vs Average Price",
            template=self.template
        )

        return fig.to_html(full_html=False)

    # ==========================================================
    # SECTOR MAP ANALYSIS
    # ==========================================================
    def sector_map(self, df, latlong):

        geo_df = self.prepare_geo_data(
            df,
            latlong
        )

        group_df = (
            geo_df
            .groupby("sector")[
                [
                    "price",
                    "built_up_area",
                    "latitude",
                    "longitude"
                ]
            ]
            .mean()
            .reset_index()
        )

        fig = px.scatter_mapbox(
            group_df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="built_up_area",
            hover_name="sector",
            zoom=9,
            title="Sector-wise Price Heatmap",
            mapbox_style="open-street-map",
            template=self.template
        )

        return fig.to_html(full_html=False)