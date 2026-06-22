from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine
from sqlalchemy import inspect


def test_initial_migration_creates_and_removes_users_table(
    tmp_path,
) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    config = AlembicConfig("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["database_url_override"] = database_url

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    assert "users" in inspect(engine).get_table_names()
    assert "alembic_version" in inspect(engine).get_table_names()

    command.downgrade(config, "base")

    assert "users" not in inspect(engine).get_table_names()
