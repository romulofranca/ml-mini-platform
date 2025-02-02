from enum import Enum


class EnvironmentEnum(str, Enum):
    """Global Enum for environment stages used across the application."""

    dev = "dev"
    staging = "staging"
    production = "production"
