#!/usr/bin/env python
"""
Sustainability Marketing Classifier - Entry Point

Run with: crewai run

Inputs are provided via environment variables:
  GOOGLE_DRIVE_URL  - Google Drive spreadsheet URL (required)
  URL_COLUMN        - Column name containing URLs (required)
  OUTPUT_FILENAME   - Output CSV filename (optional)
  BATCH_SIZE        - URLs per batch (optional, default: 5)
"""

import os
import sys
import logging
import warnings

# =============================================================================
# Suppress ALL CrewAI noise BEFORE any crewai imports
# =============================================================================

# Disable telemetry
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Event pairing mismatch.*")
warnings.filterwarnings("ignore", message=".*CrewAIEventsBus.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="crewai.*")

# Suppress noisy loggers before they initialize
for logger_name in [
    "crewai", "crewai.telemetry", "crewai.utilities", "crewai.flow",
    "litellm", "openai", "httpx", "opentelemetry", "urllib3"
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)


class _SuppressCrewAINoise(logging.Filter):
    """Filter out CrewAI event bus warnings and API usage spam."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        suppress = [
            "Event pairing mismatch", "CrewAIEventsBus", "API usage",
            "input_tokens", "output_tokens", "total_tokens", "OpenAI API"
        ]
        return not any(s in msg for s in suppress)


# Patch stdout/stderr to filter [CrewAIEventsBus] messages
class _FilteredStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if "[CrewAIEventsBus]" not in text and "API usage" not in text:
            self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _FilteredStream(sys.__stdout__)
sys.stderr = _FilteredStream(sys.__stderr__)

from automated_sequential_batch_sustainability_classifier.batch_flow import (
    run_batch_flow,
)

# Configure logging with filter
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addFilter(_SuppressCrewAINoise())
logger = logging.getLogger(__name__)


def run() -> None:
    """
    Run the sustainability classification flow.

    Inputs from environment variables:
        GOOGLE_DRIVE_URL: Google Drive spreadsheet URL
        URL_COLUMN: Column name containing URLs
        OUTPUT_FILENAME: Output CSV filename (optional)
        BATCH_SIZE: URLs per batch (optional)
    """
    google_drive_url = os.environ.get("GOOGLE_DRIVE_URL")
    url_column = os.environ.get("URL_COLUMN")
    output_filename = os.environ.get("OUTPUT_FILENAME")
    batch_size = int(os.environ.get("BATCH_SIZE", "5"))

    if not google_drive_url:
        raise ValueError("GOOGLE_DRIVE_URL environment variable is required")
    if not url_column:
        raise ValueError("URL_COLUMN environment variable is required")

    logger.info("Starting sustainability classification flow")
    logger.info(f"Google Drive URL: {google_drive_url[:60]}...")
    logger.info(f"URL Column: {url_column}")
    logger.info(f"Batch Size: {batch_size}")

    result = run_batch_flow(
        google_drive_url=google_drive_url,
        url_column=url_column,
        output_filename=output_filename,
        batch_size=batch_size,
    )

    if result.is_complete:
        logger.info(f"Complete: {result.processed_count}/{result.total_urls} URLs processed")
        if result.output_filename:
            logger.info(f"Results uploaded to Google Sheet: {result.output_filename}")
        if result.failed_urls:
            logger.warning(f"Failed: {len(result.failed_urls)} URLs")
    else:
        logger.error(f"Failed: {result.error_message}")
        raise RuntimeError(result.error_message)


if __name__ == "__main__":
    run()
