# PropLens Database

PropLens uses SQLAlchemy for database access and Alembic for schema
migrations. Local development uses SQLite through `data/proplens.db`.

## Current tables

### users

- `id`: primary key
- `email`: unique indexed email address
- `full_name`: user display name
- `hashed_password`: password hash; plain passwords are never stored
- `is_active`: account status
- `created_at`: creation timestamp
- `updated_at`: last-update timestamp

## Migration commands

Apply pending migrations:

```powershell
python -m alembic upgrade head
```

Show the current revision:

```powershell
python -m alembic current
```

Create a migration after changing a model:

```powershell
python -m alembic revision --autogenerate -m "describe the change"
```

Roll back one migration:

```powershell
python -m alembic downgrade -1
```

The application applies pending migrations automatically during startup.
