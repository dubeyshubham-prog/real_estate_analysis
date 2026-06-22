from pathlib import Path

from config.settings import Config


REQUIRED_DEPLOYMENT_FILES = (
    Path("Dockerfile"),
    Path(".dockerignore"),
    Path("requirements-core.txt"),
    Path("alembic.ini"),
    Path("migrations"),
    Config.PRICE_MODEL_FILE.relative_to(Config.BASE_DIR),
    Config.PRICE_REFERENCE_DATA_FILE.relative_to(Config.BASE_DIR),
    Config.PRICE_MODEL_METADATA_FILE.relative_to(Config.BASE_DIR),
    Config.RECOMMENDATION_MODEL_FILE.relative_to(Config.BASE_DIR),
    Config.RECOMMENDATION_METADATA_FILE.relative_to(Config.BASE_DIR),
    Config.APARTMENTS_CSV.relative_to(Config.BASE_DIR),
    Config.GURGAON_ANALYSIS_PROPERTY.relative_to(Config.BASE_DIR),
    Config.LAT_LONG_DATA.relative_to(Config.BASE_DIR),
)


def find_missing_deployment_files(
    project_root: Path = Config.BASE_DIR,
) -> list[Path]:
    """Return required Hugging Face deployment files that are missing."""
    return [
        relative_path
        for relative_path in REQUIRED_DEPLOYMENT_FILES
        if not (project_root / relative_path).exists()
    ]


def main() -> int:
    missing_files = find_missing_deployment_files()
    if missing_files:
        print("Hugging Face deployment check failed.")
        for path in missing_files:
            print(f"- Missing: {path.as_posix()}")
        return 1

    print("Hugging Face deployment files are ready.")
    print("Docker port: 7860")
    print("Vision enabled by default:", Config.ENABLE_VISION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
