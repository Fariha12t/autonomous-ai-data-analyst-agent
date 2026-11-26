"""Runtime settings, loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    provider: str = "offline"           # anthropic | openai | offline
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4o"
    max_rows: int = 500                  # hard cap on rows returned by any query
    max_retries: int = 3                 # SQL self-correction attempts
    max_rows_to_llm: int = 30            # result rows shared with the LLM for summarising
    output_dir: Path = ROOT / "outputs"


def load_settings(provider: str | None = None) -> Settings:
    """Build Settings from the environment. `provider` overrides auto-detection."""
    _load_dotenv()
    chosen = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if not chosen or chosen == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            chosen = "openai"
        else:
            chosen = "offline"
    return Settings(
        provider=chosen,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", Settings.anthropic_model),
        openai_model=os.getenv("OPENAI_MODEL", Settings.openai_model),
        max_rows=int(os.getenv("MAX_ROWS", Settings.max_rows)),
        max_retries=int(os.getenv("MAX_RETRIES", Settings.max_retries)),
        max_rows_to_llm=int(os.getenv("MAX_ROWS_TO_LLM", Settings.max_rows_to_llm)),
        output_dir=Path(os.getenv("OUTPUT_DIR", str(Settings.output_dir))),
    )
