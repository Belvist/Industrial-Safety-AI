from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Industrial Safety AI"
    app_env: str = "dev"

    inference_provider: Literal["local_onnx", "linq"] = "local_onnx"
    model_path: str = "models/ppe.onnx"
    person_model_path: str = "models/person.onnx"
    # Dual-model (PPE + dedicated person) vs single-model (PPE only) inference.
    # Benchmark with backend/scripts/benchmark_models.py before changing.
    use_separate_person_model: bool = True
    # Adaptive PPE crops of 1-2 confirmed people; 'off' restores baseline.
    ppe_roi_mode: Literal["off", "auto"] = "auto"

    # LinQ accelerator access (online access is provided by hackathon organizers).
    # Empty endpoint => LinqProvider runs the same ONNX graph locally (dev fallback)
    # keeping the Detection contract unchanged.
    # Deprecated alias kept for old .env files; use linq_amqp_host.
    linq_endpoint: str = ""
    linq_api_key: str = ""
    linq_timeout_seconds: float = Field(default=8.0, ge=1.0, le=60.0)
    linq_model: str = "ppe-linq"
    # Keep 1 until an authorized hardware concurrency benchmark selects a limit.
    linq_max_in_flight: int = Field(default=1, ge=1, le=12)

    # LinQ TPU service: RabbitMQ broker + compiled program on the shared cache.
    # Empty host keeps LinqProvider on the local ONNX fallback. For the
    # hackathon network set LINQ_AMQP_HOST=10.0.114.30 (see README_HACKATHON).
    linq_amqp_host: str = ""
    linq_amqp_port: int = Field(default=5672, ge=1, le=65535)
    linq_amqp_user: str = "guest"
    linq_amqp_password: str = "guest"
    linq_amqp_vhost: str = "/"
    linq_amqp_exchange: str = "tpu.direct"
    linq_amqp_input_queue: str = "input"
    linq_tpu_program: str = "ppe_nosig_p999.tpu"
    linq_tpu_cache_dir: str = "/auto/cache/hackathon93154"
    # Our compiled program drops the final Sigmoid; class rows are logits.
    linq_tpu_output_is_logits: bool = True
    # Comma-separated ONNX execution providers in priority order, e.g.
    # "LinQExecutionProvider". The vendor EP name may differ — override here
    # without touching code. CPU fallback is always appended automatically.
    linq_providers: str = "LinQExecutionProvider"

    # ONNX execution providers for local inference, comma-separated by priority
    # (e.g. "CoreMLExecutionProvider" on Mac). Empty = CPU everywhere.
    # CPU fallback is always appended; the fact is reported via system/info.
    onnx_providers: str = ""

    @property
    def onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def linq_provider_list(self) -> list[str]:
        return [p.strip() for p in self.linq_providers.split(",") if p.strip()]

    # Motion and remote PPE have independent clocks and bounded work queues.
    independent_motion_enabled: bool = True
    ppe_max_in_flight_per_camera: int = Field(default=1, ge=1, le=12)
    ppe_interval_seconds: float = Field(default=0.2, ge=0.1, le=30)
    ppe_max_age_seconds: float = Field(default=2.5, ge=0.5, le=10)
    file_realtime_motion: bool = True

    # Per-violation defaults for new cameras.
    # MVP scope per project description §11: helmet + vest + danger zone.
    # Mask (respirator) and smoking are post-MVP extensions, off by default.
    rule_helmet: bool = True
    rule_vest: bool = True
    rule_mask: bool = False
    rule_smoking: bool = False
    rule_zone_enabled: bool = False

    incident_confirm_seconds: float = 2.0
    incident_clear_seconds: float = 2.0
    temporal_window: int = Field(default=4, ge=1, le=20)
    temporal_required: int = Field(default=3, ge=1, le=20)
    # Snapshot retention: terminal incidents older than N days (with files)
    # are purged on startup; 0 keeps everything (current behavior).
    snapshot_retention_days: int = Field(default=0, ge=0, le=3650)
    confidence_threshold: float = Field(default=0.4, ge=0.05, le=0.95)
    # Per-class detection thresholds (sprint accuracy §1). Negative (absence)
    # classes are intentionally stricter: claiming "no helmet" needs confidence.
    person_threshold: float = Field(default=0.45, ge=0.05, le=0.95)
    helmet_threshold: float = Field(default=0.50, ge=0.05, le=0.95)
    no_helmet_threshold: float = Field(default=0.60, ge=0.05, le=0.95)
    vest_threshold: float = Field(default=0.50, ge=0.05, le=0.95)
    no_vest_threshold: float = Field(default=0.60, ge=0.05, le=0.95)
    mask_threshold: float = Field(default=0.50, ge=0.05, le=0.95)
    no_mask_threshold: float = Field(default=0.60, ge=0.05, le=0.95)
    smoking_threshold: float = Field(default=0.55, ge=0.05, le=0.95)
    # Display floor (fix-visibility F2): detections below it are dropped;
    # between floor and class threshold they are returned flagged weak
    # (visible diagnostics, never violations). Violation thresholds untouched.
    display_floor: float = Field(default=0.25, ge=0.0, le=0.95)

    @property
    def class_thresholds(self) -> dict[str, float]:
        return {
            "person": self.person_threshold,
            "helmet": self.helmet_threshold,
            "no_helmet": self.no_helmet_threshold,
            "vest": self.vest_threshold,
            "no_vest": self.no_vest_threshold,
            "mask": self.mask_threshold,
            "no_mask": self.no_mask_threshold,
            "smoking": self.smoking_threshold,
        }

    onnx_allow_spinning: bool = False
    inference_threads: int = Field(default=2, ge=1, le=32)
    capture_decode_threads: int = Field(default=2, ge=1, le=8)
    target_fps: float = Field(default=6.0, ge=0.5, le=30)
    data_dir: str = "data"

    database_url: str = "sqlite:///./data/safety.db"

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Phone-as-camera mode (run.py --lan): public HTTPS URL of this server,
    # shown as QR in the UI. Empty = LAN mode off, QR hidden.
    public_url: str = ""
    # Demo access token. When set, every /api route except /health and
    # /system/info requires ?token= or the X-API-Token header. run.py --lan
    # generates one per launch and embeds it into the QR URL. Empty = open
    # local service (default; LAN mode must run only in a trusted network).
    api_token: str = ""
    # Phone pairing lifetimes (Sprint 1): one-time codes live minutes,
    # scoped phone tokens live hours and die with the camera session.
    pairing_code_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    phone_token_ttl_hours: int = Field(default=12, ge=1, le=72)
    # Extra allowed origins, comma-separated (e.g. "https://192.168.1.194:8000").
    # Merged with loopback origins for CORS and the local-origin guard.
    extra_origins: str = ""

    @property
    def origin_list(self) -> list[str]:
        base = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
        return base + [o for o in self._split(self.extra_origins) if o not in base]

    @property
    def host_list(self) -> list[str]:
        hosts = ["localhost", "127.0.0.1", "testserver"]
        for origin in self._split(self.extra_origins):
            name = urlsplit(origin).hostname
            if name and name not in hosts:
                hosts.append(name)
        return hosts

    @staticmethod
    def _split(value: str) -> list[str]:
        return [p.strip() for p in value.split(",") if p.strip()]

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    @property
    def storage_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// database URLs are supported.")
        return self.resolve(self.database_url[len(prefix) :])


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
