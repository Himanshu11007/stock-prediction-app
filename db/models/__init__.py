"""Import every model module here so SQLModel.metadata is fully populated
for both Alembic autogenerate and create_all() in tests/scripts."""
from db.models.stock import Company, StockUniverseMember  # noqa: F401
from db.models.user import RefreshToken, Role, User, UserRoleLink  # noqa: F401
from db.models.admin import AdminAuditLog  # noqa: F401
from db.models.tracker import (  # noqa: F401
    Recommendation,
    RecommendationValidation,
    Signal,
    WatchlistItem,
)
from db.models.market import (  # noqa: F401
    AppConfig,
    EngineRun,
    FundamentalSnapshot,
    Industry,
    MarketRegimeSnapshot,
    MarketSnapshot,
    RankingOutcome,
    RankingSnapshot,
    Sector,
    StockAnalysisResult,
)
from db.models.notifications import (  # noqa: F401
    Notification,
    NotificationDelivery,
    NotificationPreference,
    NotificationRun,
    PushDevice,
    UserFeedback,
    WatchlistAlertSetting,
)
