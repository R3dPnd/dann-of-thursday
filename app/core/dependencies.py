"""
Dependency injection utilities
"""
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings, Settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def verify_api_key(api_key: str = Security(_api_key_header)) -> bool:
    """
    Verify the X-API-Key request header against SECRET_KEY.
    Skipped entirely when DEBUG=True or SECRET_KEY is the default placeholder.
    """
    insecure_default = "your-secret-key-change-in-production"
    if settings.DEBUG or settings.SECRET_KEY == insecure_default:
        return True

    if not api_key or api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True

