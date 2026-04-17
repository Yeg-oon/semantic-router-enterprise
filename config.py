from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    
    # Infrastructure Configuration
    # If you set this, it switches to Local Mode (Ollama). 
    # If you leave it empty, it uses Cloud Mode (OpenAI).
    LOCAL_LLM_URL: Optional[str] = "http://host.docker.internal:11434/v1"
    
    # Model Selection (The "Adapted" Part)
    # Clients can swap these in their .env file without changing code
    CLASSIFIER_MODEL: str = "gpt-4o-mini"
    FAST_MODEL: str = "gpt-3.5-turbo"
    SMART_MODEL: str = "gpt-4o"
    
    # System Prompts (Defaults)
    CLASSIFIER_PROMPT: str = "Classify intent as 'simple_support' or 'complex_task'. Output only the label."
    
    # Security & Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100  # requests per minute
    RATE_LIMIT_WINDOW: int = 60      # seconds
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALLOWED_ORIGINS: List[str] = ["*"]  # CORS settings
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 30  # seconds
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = "Exception"
    
    # Logging & Monitoring
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console
    METRICS_ENABLED: bool = True
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 60  # seconds

    class Config:
        env_file = ".env"

# Initialize settings
settings = Settings()