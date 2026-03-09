#!/usr/bin/env python
"""
Sustainability Marketing Classifier - Batch Processing Flow

Processes URLs from Google Drive in parallel batches using CrewAI Flows.
"""

import asyncio
import csv
import io
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from typing import List, Optional

import requests

# =============================================================================
# Suppress all CrewAI noise BEFORE importing crewai
# =============================================================================

# Suppress warnings module
warnings.filterwarnings("ignore", message=".*Event pairing mismatch.*")
warnings.filterwarnings("ignore", message=".*CrewAIEventsBus.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Suppress noisy loggers
logging.getLogger("crewai").setLevel(logging.WARNING)
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.utilities").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


class CrewAIOutputFilter(logging.Filter):
    """Filter out noisy CrewAI and API usage log messages."""

    SUPPRESSED_PATTERNS = [
        "Event pairing mismatch",
        "CrewAIEventsBus",
        "API usage",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "OpenAI API usage",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for pattern in self.SUPPRESSED_PATTERNS:
            if pattern in msg:
                return False
        return True


# Apply filter to root logger
logging.getLogger().addFilter(CrewAIOutputFilter())

from pydantic import BaseModel, Field

from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import FirecrawlScrapeWebsiteTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class URLClassification(BaseModel):
    """Classification result for a single URL."""

    url: str = Field(description="The URL that was analyzed")
    sustainability_marketing: str = Field(description="YES or NO")
    sustainability_percentage: int = Field(default=0, description="0-100 percentage")
    confidence: float = Field(default=0.5, description="0.0-1.0 confidence score")
    themes: str = Field(default="", description="Comma-separated themes")
    reason: str = Field(
        description="1-2 sentences explaining the classification with specific evidence"
    )


class BatchState(BaseModel):
    """State for the batch processing flow."""

    google_drive_url: str = ""
    url_column: str = ""
    output_filename: str = ""
    batch_size: int = 5
    max_concurrent: int = 5

    all_urls: List[str] = Field(default_factory=list)
    all_results: List[dict] = Field(default_factory=list)
    failed_urls: List[str] = Field(default_factory=list)

    total_urls: int = 0
    total_batches: int = 0
    processed_count: int = 0
    is_complete: bool = False
    error_message: str = ""


# =============================================================================
# Direct Google Sheets Reader (bypasses LLM - recommended for large sheets)
# =============================================================================


class DirectGoogleSheetsReader:
    """
    Reads URLs directly from Google Sheets using CSV export.
    This bypasses the LLM entirely, avoiding truncation issues.

    Requirements:
    - The Google Sheet must be publicly accessible (Anyone with link can view)
    - Or you must have proper OAuth credentials configured
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SustainabilityClassifier/1.0)'
        })

    def read_urls(self, google_drive_url: str, url_column: str) -> List[str]:
        """
        Extract URLs from a Google Drive spreadsheet using direct CSV export.

        Args:
            google_drive_url: Google Sheets URL
            url_column: Column name containing URLs

        Returns:
            List of URLs found in the specified column
        """
        spreadsheet_id = self._extract_spreadsheet_id(google_drive_url)

        # Try to get gid (sheet ID) from URL, default to 0 (first sheet)
        gid = self._extract_gid(google_drive_url)

        # CSV export URL format
        csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"

        logger.info(f"Fetching spreadsheet directly via CSV export: {spreadsheet_id}")
        logger.info(f"Looking for column: {url_column}")

        try:
            response = self.session.get(csv_url, timeout=60)
            response.raise_for_status()

            # Parse CSV content
            content = response.content.decode('utf-8-sig')  # Handle BOM if present
            reader = csv.DictReader(io.StringIO(content))

            # Find the URL column (case-insensitive)
            fieldnames = reader.fieldnames or []
            url_col_name = None
            for field in fieldnames:
                if field.strip().upper() == url_column.upper():
                    url_col_name = field
                    break

            if not url_col_name:
                # Try partial match
                for field in fieldnames:
                    if url_column.upper() in field.upper():
                        url_col_name = field
                        logger.info(f"Using partial column match: '{field}' for '{url_column}'")
                        break

            if not url_col_name:
                logger.error(f"Column '{url_column}' not found. Available columns: {fieldnames}")
                # Fallback: look for any column containing URLs
                logger.info("Attempting to find URLs in any column...")
                return self._extract_urls_from_any_column(content)

            urls = []
            for row in reader:
                cell = row.get(url_col_name, '').strip()
                if cell.startswith('http://') or cell.startswith('https://'):
                    urls.append(cell)

            logger.info(f"Successfully extracted {len(urls)} URLs from column '{url_col_name}'")
            return urls

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (401, 403):
                logger.warning(f"CSV export failed ({e.response.status_code}). Sheet may not be public.")
                logger.info("Falling back to LLM-based reader...")
                return self._fallback_to_llm_reader(google_drive_url, url_column)
            elif e.response.status_code == 404:
                logger.error(f"Spreadsheet not found: {spreadsheet_id}")
                raise ValueError(f"Spreadsheet not found: {spreadsheet_id}")
            else:
                logger.warning(f"CSV export failed with status {e.response.status_code}")
                logger.info("Falling back to LLM-based reader...")
                return self._fallback_to_llm_reader(google_drive_url, url_column)
        except Exception as e:
            logger.error(f"Direct CSV read failed: {e}")
            logger.info("Falling back to LLM-based reader...")
            return self._fallback_to_llm_reader(google_drive_url, url_column)

    def _extract_spreadsheet_id(self, url: str) -> str:
        """Extract spreadsheet ID from Google Sheets URL."""
        if "/spreadsheets/d/" in url:
            parts = url.split("/spreadsheets/d/")[1]
            return parts.split("/")[0].split("?")[0]
        return url

    def _extract_gid(self, url: str) -> str:
        """Extract sheet gid from URL, default to 0."""
        if "gid=" in url:
            match = re.search(r'gid=(\d+)', url)
            if match:
                return match.group(1)
        return "0"

    def _extract_urls_from_any_column(self, csv_content: str) -> List[str]:
        """Fallback: extract URLs from any column in the CSV."""
        urls = []
        reader = csv.reader(io.StringIO(csv_content))
        next(reader, None)  # Skip header

        for row in reader:
            for cell in row:
                cell = cell.strip()
                if cell.startswith('http://') or cell.startswith('https://'):
                    urls.append(cell)
                    break  # Take first URL per row

        logger.info(f"Fallback extraction found {len(urls)} URLs")
        return urls

    def _fallback_to_llm_reader(self, google_drive_url: str, url_column: str) -> List[str]:
        """Fall back to the LLM-based GoogleDriveReader."""
        reader = GoogleDriveReader()
        return reader.read_urls(google_drive_url, url_column)


# =============================================================================
# Google Sheets Uploader
# =============================================================================


class GoogleSheetsUploader:
    """Uploads results to a new Google Sheet using CrewAI Enterprise integration."""

    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.llm = LLM(model=llm_model, temperature=0.1)

    def upload_results(self, df, source_spreadsheet_id: str = None) -> str:
        """
        Upload a DataFrame to a new Google Sheet.

        Args:
            df: pandas DataFrame with results
            source_spreadsheet_id: Optional ID of source sheet for naming

        Returns:
            URL of the created Google Sheet
        """
        import pandas as pd

        # Convert DataFrame to list of lists for the API
        headers = df.columns.tolist()
        rows = df.values.tolist()

        # Prepare all data (headers + rows)
        all_data = [headers] + rows

        # Create the sheet first
        sheet_title = f"Sustainability Results {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        logger.info(f"Creating new Google Sheet: {sheet_title}")

        # Step 1: Create a new spreadsheet
        spreadsheet_id = self._create_spreadsheet(sheet_title)

        if not spreadsheet_id:
            raise RuntimeError("Failed to create new Google Sheet")

        logger.info(f"Created spreadsheet with ID: {spreadsheet_id}")

        # Step 2: Upload data in batches to avoid LLM token limits
        batch_size = 50  # Upload 50 rows at a time
        total_rows = len(all_data)

        for i in range(0, total_rows, batch_size):
            batch = all_data[i:i + batch_size]
            start_row = i + 1  # 1-indexed for Sheets

            # For first batch, include headers at row 1
            if i == 0:
                range_str = f"A1:Z{len(batch)}"
            else:
                range_str = f"A{start_row}:Z{start_row + len(batch) - 1}"

            logger.info(f"Uploading rows {i + 1}-{min(i + batch_size, total_rows)} of {total_rows}")
            self._update_sheet_values(spreadsheet_id, range_str, batch)

        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        logger.info(f"Successfully uploaded {len(rows)} results to: {sheet_url}")

        return sheet_url

    def _create_spreadsheet(self, title: str) -> Optional[str]:
        """Create a new Google Spreadsheet and return its ID."""
        agent = Agent(
            role="Google Sheets Creator",
            goal="Create a new Google Spreadsheet",
            backstory="You create Google Spreadsheets.",
            llm=self.llm,
            tools=[],
            apps=["google_sheets/create_spreadsheet"],
            verbose=True,
        )

        task = Task(
            description=f"""
            Create a new Google Spreadsheet with the title: "{title}"

            Use google_sheets/create_spreadsheet with:
            - title: "{title}"

            Return ONLY the spreadsheet ID from the response (the 'spreadsheetId' field).
            Do not include any other text, just the ID string.
            """,
            expected_output="The spreadsheet ID string only.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)

        # Extract spreadsheet ID from response
        # Look for the ID pattern (alphanumeric with dashes/underscores)
        id_patterns = [
            r'"spreadsheetId"\s*:\s*"([^"]+)"',
            r"'spreadsheetId'\s*:\s*'([^']+)'",
            r'spreadsheetId["\']?\s*:\s*["\']?([a-zA-Z0-9_-]{20,})',
            r'([a-zA-Z0-9_-]{30,50})',  # Fallback: look for long alphanumeric string
        ]

        for pattern in id_patterns:
            match = re.search(pattern, raw_output)
            if match:
                return match.group(1)

        logger.error(f"Could not extract spreadsheet ID from: {raw_output[:200]}")
        return None

    def _update_sheet_values(self, spreadsheet_id: str, range_str: str, values: List[List]) -> bool:
        """Update values in a Google Sheet."""
        # Convert values to JSON string for the prompt
        values_json = json.dumps(values)

        agent = Agent(
            role="Google Sheets Writer",
            goal="Write data to Google Sheets",
            backstory="You write data to Google Spreadsheets accurately.",
            llm=self.llm,
            tools=[],
            apps=["google_sheets/update_values"],
            verbose=True,
        )

        task = Task(
            description=f"""
            Update values in Google Sheets.

            Use google_sheets/update_values with:
            - spreadsheet_id: "{spreadsheet_id}"
            - range: "{range_str}"
            - values: {values_json}
            - valueInputOption: "RAW"

            Confirm the update was successful.
            """,
            expected_output="Confirmation that values were updated.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        return True


# =============================================================================
# LLM-based Google Drive Reader (fallback for non-public sheets)
# =============================================================================


class GoogleDriveReader:
    """Reads URLs from Google Drive using CrewAI Enterprise integration with pagination."""

    ROWS_PER_PAGE = 30  # Fetch 30 rows at a time to prevent LLM truncation

    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.llm = LLM(model=llm_model, temperature=0.1)

    def read_urls(self, google_drive_url: str, url_column: str) -> List[str]:
        """Extract URLs from a Google Drive spreadsheet using paginated reads."""
        spreadsheet_id = self._extract_spreadsheet_id(google_drive_url)
        all_urls = []
        page = 0
        max_pages = 200  # Safety limit: 200 pages * 30 rows = 6000 max rows
        consecutive_empty_pages = 0
        max_consecutive_empty = 2  # Stop after 2 consecutive empty pages

        logger.info(f"Reading spreadsheet {spreadsheet_id} in pages of {self.ROWS_PER_PAGE} rows")

        while page < max_pages:
            start_row = (page * self.ROWS_PER_PAGE) + 1
            end_row = start_row + self.ROWS_PER_PAGE - 1

            # First page includes header, subsequent pages don't need it
            if page == 0:
                range_str = f"A1:Z{end_row}"
            else:
                range_str = f"A{start_row}:Z{end_row}"

            logger.info(f"Fetching page {page + 1}: rows {start_row}-{end_row}")

            urls_from_page = self._fetch_page(spreadsheet_id, range_str, url_column, include_header=(page == 0))

            if not urls_from_page:
                consecutive_empty_pages += 1
                logger.info(f"No URLs found on page {page + 1} (consecutive empty: {consecutive_empty_pages})")
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.info(f"Stopping pagination after {max_consecutive_empty} consecutive empty pages")
                    break
            else:
                consecutive_empty_pages = 0  # Reset counter when we find URLs
                all_urls.extend(urls_from_page)
                logger.info(f"Page {page + 1}: found {len(urls_from_page)} URLs (total: {len(all_urls)})")

            page += 1

        logger.info(f"Total URLs extracted: {len(all_urls)}")
        return all_urls

    def _detect_truncation(self, text: str) -> bool:
        """Detect if the LLM truncated the response."""
        truncation_patterns = [
            r'\.\.\.\s*and\s+\d+\s+more',
            r'\.\.\.\s*\d+\s+additional',
            r'truncated',
            r'remaining\s+\d+\s+rows',
            r'continues\s+with',
            r'\[\.\.\.]\s*$',
            r'etc\.\s*$',
            r'and\s+so\s+on',
            r'\d+\s+more\s+rows',
            r'omitted\s+for\s+brevity',
        ]
        for pattern in truncation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _fetch_page(self, spreadsheet_id: str, range_str: str, url_column: str, include_header: bool, retry_count: int = 0) -> List[str]:
        """Fetch a single page of data from the spreadsheet with retry logic."""
        max_retries = 2

        agent = Agent(
            role="Google Sheets Data Extractor",
            goal="Fetch ALL spreadsheet data and return EVERY row without any summarization",
            backstory="You are a precise data extraction agent. You MUST return complete, unmodified API responses. NEVER truncate or summarize data.",
            llm=self.llm,
            tools=[],
            apps=["google_sheets/get_values"],
            verbose=True,
        )

        task = Task(
            description=f"""
            Fetch data from Google Sheets and return ALL rows.

            Use google_sheets/get_values with:
            - spreadsheet_id: "{spreadsheet_id}"
            - range: "{range_str}"
            - majorDimension: "ROWS"

            CRITICAL INSTRUCTIONS:
            1. Return the COMPLETE raw API response with ALL rows
            2. Do NOT summarize, truncate, or skip any rows
            3. Do NOT say "and X more rows" - include EVERY row
            4. Do NOT use "..." or ellipsis - show all data
            5. Output the full JSON response exactly as received
            6. If there are 50 rows, you must show all 50 rows
            """,
            expected_output="Complete raw JSON response from Google Sheets API containing ALL rows of data. No truncation.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)

        # Check for truncation
        if self._detect_truncation(raw_output) and retry_count < max_retries:
            logger.warning(f"Detected truncated response for range {range_str}, retrying ({retry_count + 1}/{max_retries})")
            return self._fetch_page(spreadsheet_id, range_str, url_column, include_header, retry_count + 1)

        urls = self._parse_urls_from_api_response(raw_output, url_column, skip_header=not include_header)

        # Log parsing details for debugging
        logger.info(f"Range {range_str}: output {len(raw_output)} chars, parsed {len(urls)} URLs")

        # Warn if we got suspiciously few URLs (likely truncation)
        if len(urls) < 10 and "Z50" in range_str and retry_count == 0:
            logger.warning(f"Low URL count ({len(urls)}) for range {range_str} - possible truncation")

        return urls

    def _extract_spreadsheet_id(self, url: str) -> str:
        """Extract spreadsheet ID from Google Sheets URL."""
        if "/spreadsheets/d/" in url:
            parts = url.split("/spreadsheets/d/")[1]
            return parts.split("/")[0]
        return url

    def _parse_urls_from_api_response(self, text: str, url_column: str, skip_header: bool = False) -> List[str]:
        """Parse URLs directly from Google Sheets API response JSON."""
        urls = []
        all_rows = []

        # Method 1: Try to find and parse complete JSON structure
        # Look for "values": followed by a nested array structure
        try:
            # Find the start of values array
            values_start = text.find('"values"')
            if values_start == -1:
                values_start = text.find("'values'")

            if values_start != -1:
                # Find the opening bracket after "values":
                bracket_start = text.find('[', values_start)
                if bracket_start != -1:
                    # Count brackets to find matching close
                    depth = 0
                    bracket_end = -1
                    for i in range(bracket_start, len(text)):
                        if text[i] == '[':
                            depth += 1
                        elif text[i] == ']':
                            depth -= 1
                            if depth == 0:
                                bracket_end = i
                                break

                    if bracket_end != -1:
                        array_str = text[bracket_start:bracket_end + 1]
                        # Clean up for JSON parsing
                        array_str = array_str.replace("'", '"')
                        try:
                            rows = json.loads(array_str)
                            if isinstance(rows, list):
                                all_rows.extend(rows)
                                logger.debug(f"Method 1: Parsed {len(rows)} rows from JSON")
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")

        # Method 2: Fallback - find individual row arrays
        if not all_rows:
            # Match rows like ["value1", "value2", ...]
            row_pattern = r'\[([^\[\]]+)\]'
            row_matches = re.findall(row_pattern, text)
            for row_match in row_matches:
                # Skip if it looks like it's not a data row
                if 'http' not in row_match.lower() and url_column.lower() not in row_match.lower():
                    continue
                try:
                    row = json.loads("[" + row_match + "]")
                    if isinstance(row, list) and len(row) > 0:
                        all_rows.append(row)
                except json.JSONDecodeError:
                    continue
            if all_rows:
                logger.debug(f"Method 2: Parsed {len(all_rows)} rows from individual arrays")

        if all_rows:
            # Find the column index for url_column (check first row for header)
            url_col_idx = None
            start_idx = 0

            if all_rows and isinstance(all_rows[0], list):
                header = all_rows[0]
                for idx, col_name in enumerate(header):
                    if str(col_name).strip().upper() == url_column.upper():
                        url_col_idx = idx
                        break

                # If we found a header match, skip it unless told otherwise
                if url_col_idx is not None and not skip_header:
                    start_idx = 1
                elif skip_header:
                    start_idx = 0  # No header in this page

            if url_col_idx is not None:
                # Extract URLs from that column
                for row in all_rows[start_idx:]:
                    if isinstance(row, list) and len(row) > url_col_idx:
                        cell = str(row[url_col_idx]).strip()
                        if cell.startswith("http://") or cell.startswith("https://"):
                            urls.append(cell)
            else:
                # Column not found by name, try to find URLs in any column
                for row in all_rows[start_idx:]:
                    if isinstance(row, list):
                        for cell in row:
                            cell_str = str(cell).strip()
                            if cell_str.startswith("http://") or cell_str.startswith("https://"):
                                urls.append(cell_str)

        # Fallback: regex extraction if no structured data found
        if not urls:
            logger.info("No structured data found, falling back to regex extraction")
            # More comprehensive URL pattern
            url_patterns = [
                r'https?://[^\s<>"\')\],}\\]+',
                r'https?://[\w\-\.]+\.[a-zA-Z]{2,}[^\s<>"\')*\],}\\]*',
            ]
            seen_urls = set()
            for url_pattern in url_patterns:
                found_urls = re.findall(url_pattern, text)
                for url in found_urls:
                    url = url.rstrip('.,;:')
                    # Clean up common trailing artifacts
                    url = re.sub(r'["\'\]\}\)]+$', '', url)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        urls.append(url)
            logger.info(f"Regex found {len(urls)} URLs")

        # Log summary
        logger.info(f"Total URLs parsed from response: {len(urls)}")
        return urls


# =============================================================================
# URL Processor
# =============================================================================


class URLProcessor:
    """Processes individual URLs for sustainability classification."""

    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.llm_model = llm_model
        self.scraper_tool = FirecrawlScrapeWebsiteTool()

    def _create_crew(self, url: str) -> Crew:
        """Create a crew for processing a single URL."""
        llm = LLM(model=self.llm_model, temperature=0.7)

        agent = Agent(
            role="Sustainability Marketing Analyst",
            goal="Scrape and classify URL for sustainability marketing",
            backstory=(
                "Expert marketing analyst specializing in sustainability messaging. "
                "Identifies when companies use environmental benefits as competitive differentiators."
            ),
            tools=[self.scraper_tool],
            llm=llm,
            verbose=True,
        )

        task = Task(
            description=f"""
            Analyze this URL for sustainability marketing:

            URL: {url}

            1. Scrape the page content
            2. Classify whether sustainability is used as a marketing message

            Classification criteria:
            - YES: Sustainability/environmental benefits explicitly used as marketing differentiator
            - NO: Sustainability absent, incidental, or purely technical

            If YES, assign percentage (10-20% minor, 25-40% clear, 50-70% primary, 80-100% core).

            Themes: Energy Efficiency; Carbon/Embodied Carbon; Green Certifications;
            Environmental Analysis Tools; Climate Resilience; Material Sustainability;
            Operational Emissions; Waste Reduction; Renewables/Clean Energy

            Provide 1-2 sentence reason citing specific evidence from the page.
            """,
            expected_output=f"""
            JSON object:
            {{"url": "{url}", "sustainability_marketing": "YES/NO", "sustainability_percentage": 0-100,
            "confidence": 0.0-1.0, "themes": "...", "reason": "..."}}
            """,
            agent=agent,
            output_pydantic=URLClassification,
        )

        return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)

    async def process_async(self, url: str) -> dict:
        """Process a single URL asynchronously."""
        try:
            crew = self._create_crew(url)
            result = await crew.kickoff_async()
            return self._parse_result(url, result)
        except Exception as e:
            logger.warning(f"Failed to process {url}: {e}")
            return self._default_result(url, str(e))

    def _parse_result(self, url: str, crew_result) -> dict:
        """Parse crew result into dictionary."""
        try:
            if hasattr(crew_result, "pydantic") and crew_result.pydantic:
                c = crew_result.pydantic
                return {
                    "url": url,
                    "sustainability_marketing": c.sustainability_marketing.upper(),
                    "sustainability_percentage": c.sustainability_percentage,
                    "confidence": c.confidence,
                    "themes": c.themes,
                    "reason": c.reason,
                }
        except Exception:
            pass

        # Fallback: try JSON extraction
        raw = str(crew_result)
        try:
            match = re.search(r"\{[^{}]*\"url\"[^{}]*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "url": url,
                    "sustainability_marketing": str(data.get("sustainability_marketing", "NO")).upper(),
                    "sustainability_percentage": int(data.get("sustainability_percentage", 0)),
                    "confidence": float(data.get("confidence", 0.5)),
                    "themes": str(data.get("themes", "")),
                    "reason": str(data.get("reason", "")),
                }
        except Exception:
            pass

        return self._default_result(url, "Could not parse result")

    def _default_result(self, url: str, reason: str) -> dict:
        """Return default classification result."""
        return {
            "url": url,
            "sustainability_marketing": "NO",
            "sustainability_percentage": 0,
            "confidence": 0.3,
            "themes": "",
            "reason": reason[:200],
        }


# =============================================================================
# Parallel Batch Processor
# =============================================================================


class ParallelBatchProcessor:
    """Processes URLs in parallel with concurrency control."""

    def __init__(self, llm_model: str = "openai/gpt-4o", max_concurrent: int = 5):
        self.processor = URLProcessor(llm_model=llm_model)
        self.max_concurrent = max_concurrent

    async def process_batch(self, urls: List[str]) -> List[dict]:
        """Process a batch of URLs in parallel."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_limit(url: str) -> dict:
            async with semaphore:
                return await self.processor.process_async(url)

        tasks = [process_with_limit(url) for url in urls]
        return await asyncio.gather(*tasks)

    def process_batch_sync(self, urls: List[str]) -> List[dict]:
        """Synchronous wrapper for batch processing."""
        return asyncio.run(self.process_batch(urls))


# =============================================================================
# Main Flow
# =============================================================================


class SustainabilityBatchFlow(Flow[BatchState]):
    """
    Flow for processing sustainability classifications in parallel batches.

    Steps:
    1. Load URLs from Google Drive
    2. Process in parallel batches
    3. Save results to CSV
    """

    def __init__(self, llm_model: str = "openai/gpt-4o"):
        super().__init__()
        self.llm_model = llm_model

    @start()
    def load_data(self) -> BatchState:
        """Load URLs from Google Drive using direct CSV export (preferred) or LLM fallback."""
        logger.info(f"Loading data from Google Drive: {self.state.google_drive_url[:60]}...")
        logger.info(f"Looking for column: {self.state.url_column}")

        state = self.state

        try:
            # Use direct CSV reader first (faster, no truncation issues)
            reader = DirectGoogleSheetsReader()
            urls = reader.read_urls(state.google_drive_url, state.url_column)

            if not urls:
                state.error_message = f"No URLs found in column '{state.url_column}'"
                logger.error(state.error_message)
                return state

            state.all_urls = urls
            state.total_urls = len(urls)
            state.total_batches = (len(urls) + state.batch_size - 1) // state.batch_size

            logger.info(f"Successfully loaded {state.total_urls} URLs")
            logger.info(f"Will process in {state.total_batches} batches of {state.batch_size}")

        except Exception as e:
            state.error_message = f"Failed to load data: {e}"
            logger.error(state.error_message, exc_info=True)

        return state

    @listen(load_data)
    def process_batches(self) -> BatchState:
        """Process all batches with parallel URL execution."""
        state = self.state

        if state.error_message or not state.all_urls:
            return state

        logger.info(f"Processing {state.total_urls} URLs in {state.total_batches} batches...")

        processor = ParallelBatchProcessor(
            llm_model=self.llm_model,
            max_concurrent=state.max_concurrent,
        )

        for batch_idx in range(state.total_batches):
            start_idx = batch_idx * state.batch_size
            end_idx = min(start_idx + state.batch_size, state.total_urls)
            batch_urls = state.all_urls[start_idx:end_idx]

            logger.info(f"Batch {batch_idx + 1}/{state.total_batches}: {len(batch_urls)} URLs")

            try:
                results = processor.process_batch_sync(batch_urls)
                state.all_results.extend(results)
                state.processed_count += len(results)
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                state.failed_urls.extend(batch_urls)

        return state

    @listen(process_batches)
    def save_results(self) -> BatchState:
        """Save results to a new Google Sheet and clean up local files."""
        state = self.state

        if not state.all_results:
            state.error_message = "No results to save"
            return state

        try:
            import pandas as pd

            df = pd.DataFrame(state.all_results)

            columns = [
                "url",
                "sustainability_marketing",
                "sustainability_percentage",
                "confidence",
                "themes",
                "reason",
            ]
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            df = df[columns]

            # Extract source spreadsheet ID for reference
            source_id = None
            if state.google_drive_url and "/spreadsheets/d/" in state.google_drive_url:
                parts = state.google_drive_url.split("/spreadsheets/d/")[1]
                source_id = parts.split("/")[0].split("?")[0]

            # Upload to new Google Sheet
            logger.info("Uploading results to new Google Sheet...")
            uploader = GoogleSheetsUploader(llm_model=self.llm_model)
            sheet_url = uploader.upload_results(df, source_spreadsheet_id=source_id)

            state.output_filename = sheet_url
            state.is_complete = True
            logger.info(f"Results uploaded to: {sheet_url}")

            # Also save a temporary local copy, then delete it
            temp_path = self._generate_filename()
            df.to_csv(temp_path, index=False)
            logger.info(f"Temporary local file created: {temp_path}")

            # Delete local file
            try:
                import os
                os.remove(temp_path)
                logger.info(f"Deleted local file: {temp_path}")
            except Exception as del_err:
                logger.warning(f"Could not delete local file {temp_path}: {del_err}")

        except Exception as e:
            state.error_message = f"Failed to save results: {e}"
            logger.error(state.error_message, exc_info=True)

            # Fallback: save locally if upload fails
            try:
                import pandas as pd
                df = pd.DataFrame(state.all_results)
                output_path = self._generate_filename()
                df.to_csv(output_path, index=False)
                logger.warning(f"Upload failed, saved locally to: {output_path}")
                state.output_filename = output_path
            except Exception:
                pass

        return state

    def _generate_filename(self) -> str:
        """Generate timestamped output filename."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return f"sustainability_results_{timestamp}.csv"


# =============================================================================
# Public API
# =============================================================================


def run_batch_flow(
    google_drive_url: str,
    url_column: str,
    output_filename: Optional[str] = None,
    batch_size: int = 5,
    max_concurrent: int = 5,
    llm_model: str = "openai/gpt-4o",
) -> BatchState:
    """
    Run the sustainability classification flow.

    Args:
        google_drive_url: Google Drive URL to the spreadsheet
        url_column: Column name containing URLs
        output_filename: Output CSV filename (auto-generated if not provided)
        batch_size: URLs per batch
        max_concurrent: Max concurrent URL processing
        llm_model: LLM model to use

    Returns:
        BatchState with results
    """
    if not google_drive_url:
        raise ValueError("google_drive_url is required")
    if not url_column:
        raise ValueError("url_column is required")

    initial_state = BatchState(
        google_drive_url=google_drive_url,
        url_column=url_column,
        output_filename=output_filename or "",
        batch_size=batch_size,
        max_concurrent=max_concurrent,
    )

    flow = SustainabilityBatchFlow(llm_model=llm_model)
    return flow.kickoff(inputs=initial_state.model_dump())
