import pandas as pd
import sweetviz as sv
from config.settings import Config  # We import Config here
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EDAAnalyzer:

    def __init__(self) -> None:
        # Fix: Pull BASE_DIR safely from the Config class
        self.reports_dir = Config.BASE_DIR / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, df: pd.DataFrame, name: str) -> None:
        logger.info(f"Starting EDA for {name}...")
        try:
            report = sv.analyze(df)
            output_path = self.reports_dir / f"eda_{name}.html"
            report.show_html(str(output_path), open_browser=False)
            logger.info(f"EDA report saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate EDA for {name}: {str(e)}")