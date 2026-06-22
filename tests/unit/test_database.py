from sqlalchemy import inspect
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.database.models.user import User
from src.database.session import create_database_engine


def build_test_session(tmp_path) -> Session:
    database_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=Session,
    )
    return factory()


def test_users_table_is_created(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'schema.db').as_posix()}"
    engine = create_database_engine(database_url)

    Base.metadata.create_all(bind=engine)

    assert "users" in inspect(engine).get_table_names()


def test_user_can_be_saved_and_read(tmp_path) -> None:
    with build_test_session(tmp_path) as session:
        user = User(
            email="learner@example.com",
            full_name="Database Learner",
            hashed_password="not-a-real-password-hash",
        )
        session.add(user)
        session.commit()

        saved_user = session.scalar(
            select(User).where(User.email == "learner@example.com")
        )

        assert saved_user is not None
        assert saved_user.id is not None
        assert saved_user.full_name == "Database Learner"
        assert saved_user.is_active is True
