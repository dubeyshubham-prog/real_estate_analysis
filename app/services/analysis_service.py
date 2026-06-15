from src.utils.data_loader import DataLoader
from src.utils.analyser import PropertyAnalyzer


class AnalysisService:

    def __init__(self):

        self.loader = DataLoader()
        self.analyzer = PropertyAnalyzer()

    # ==========================================================
    # LOAD DATASETS
    # ==========================================================
    def load_data(self):

        df = self.loader.load_gurgaon_property_analysis_data()
        latlong = self.loader.load_lat_long_data()

        return df, latlong

    # ==========================================================
    # CREATE ALL ANALYSIS CHARTS
    # ==========================================================
    def get_dashboard_charts(self):

        df, latlong = self.load_data()

        charts = {
            "property_type_distribution": (
                self.analyzer.property_type_distribution(df)
            ),

            "top_expensive_sectors": (
                self.analyzer.top_expensive_sectors(df)
            ),

            "top_affordable_sectors": (
                self.analyzer.top_affordable_sectors(df)
            ),

            "area_vs_price": (
                self.analyzer.area_vs_price(df)
            ),

            "bhk_price_analysis": (
                self.analyzer.bhk_price_analysis(df)
            ),

            "luxury_price_impact": (
                self.analyzer.luxury_price_impact(df)
            ),

            "furnishing_price_impact": (
                self.analyzer.furnishing_price_impact(df)
            ),

            # "sector_map": (
            #     self.analyzer.sector_map(df, latlong)
            # )
        }

        return charts