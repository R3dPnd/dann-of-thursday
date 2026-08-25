"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings"""

    # Project Information
    PROJECT_NAME: str = "Dann of Thursday MCP API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database (Optional for future use)
    DATABASE_URL: Optional[str] = None

    # MCP Configuration
    MCP_SERVER_URL: Optional[str] = None
    MCP_TIMEOUT: int = 30

    # Cloudflare Access (defense-in-depth JWT check; Access itself gates at
    # the edge, this just stops anything that reaches the box directly, e.g.
    # over Tailscale/LAN, from skipping auth). Leave disabled for pure
    # localhost/dev use.
    CF_ACCESS_ENABLED: bool = False
    CF_ACCESS_TEAM_DOMAIN: Optional[str] = None  # e.g. "yourteam.cloudflareaccess.com"
    CF_ACCESS_AUD: Optional[str] = None          # Application Audience (AUD) tag from the Access app

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

