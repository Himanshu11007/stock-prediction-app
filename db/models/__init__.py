"""Import every model module here so SQLModel.metadata is fully populated
for both Alembic autogenerate and create_all() in tests/scripts."""
from db.models.stock import Company, StockUniverseMember  # noqa: F401
