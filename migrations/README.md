# Database migrations

Apply all pending migrations:

```powershell
python -m alembic upgrade head
```

Create a migration after changing a model:

```powershell
python -m alembic revision --autogenerate -m "describe the change"
```

Roll back the latest migration:

```powershell
python -m alembic downgrade -1
```
