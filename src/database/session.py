from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from config.settings import Config


def create_database_engine(database_url: str = Config.DATABASE_URL) -> Engine:
    """Create a database engine with SQLite-safe connection settings."""
    connection_options = (
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {}
    )
    return create_engine(
        database_url,
        connect_args=connection_options,
        pool_pre_ping=True,
    )


engine = create_database_engine()
SessionFactory = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_database_session() -> Generator[Session, None, None]:
    """Provide one database session and always close it after use."""
    with SessionFactory() as session:
        yield session
