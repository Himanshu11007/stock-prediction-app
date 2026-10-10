"""Import every model module here so SQLModel.metadata is fully populated
for both Alembic autogenerate and create_all() in tests/scripts."""
from db.models.stock import Company, StockUniverseMember  # noqa: F401
from db.models.user import (  # noqa: F401
    AuthThrottleEvent,
    ExternalIdentity,
    OtpChallenge,
    PasswordResetToken,
    RefreshToken,
    Role,
    TrustedDevice,
    User,
    UserRoleLink,
)
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
    PriceQuote,
    RankingOutcome,
    RankingSnapshot,
    ScheduledJobRun,
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
from db.models.prediction import (  # noqa: F401
    EventClassification,
    ExitState,
    ExitTransition,
    FactorObservation,
    FeatureSnapshot,
    MarketEvent,
    Prediction,
    PredictionOutcome,
    PredictionRun,
    V2UniverseMember,
)
from db.models.news import EventEntity, NewsArticle, NewsIngestionRun  # noqa: F401,E402
