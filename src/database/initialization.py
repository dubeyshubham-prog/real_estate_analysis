from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig

from config.settings import Config


def initialize_database() -> None:
    """Apply all pending database migrations during application startup."""
    migration_config = AlembicConfig(
        str(Path(Config.BASE_DIR) / "alembic.ini")
    )
    migration_config.set_main_option(
        "script_location",
        str(Path(Config.BASE_DIR) / "migrations"),
    )
    migration_config.attributes["database_url_override"] = (
        Config.DATABASE_URL
    )
    command.upgrade(migration_config, "head")
