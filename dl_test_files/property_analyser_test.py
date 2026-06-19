from src.utils.analyser import PropertyAnalyzer
from src.utils.data_loader import DataLoader


def test_market_summary():

    data_loader = DataLoader()

    df = data_loader.load_gurgaon_property_analysis_data()

    analyzer = PropertyAnalyzer()

    summary = analyzer.market_summary(df)

    print(summary)

    assert summary["total_properties"] > 0
    assert summary["average_price_cr"] > 0
    assert len(summary["top_expensive_sectors"]) > 0
    assert len(summary["top_affordable_sectors"]) > 0