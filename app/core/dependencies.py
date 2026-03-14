"""
Dependency injection utilities
"""
from typing import Generator
from fastapi import Depends, HTTPException, status
from app.core.config import settings, Settings


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def verify_api_key(api_key: str = None) -> bool:
    """
    Verify API key (placeholder for future implementation)
    """
    # TODO: Implement proper API key verification
    if settings.DEBUG:
        return True
    # In production, verify against stored API keys
    return True

