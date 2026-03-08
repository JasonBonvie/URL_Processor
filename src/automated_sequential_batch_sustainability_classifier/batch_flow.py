#!/usr/bin/env python
"""
Batch Processing Flow for Sustainability Classification

This module implements CrewAI Flows to process large Excel files in batches,
avoiding context limits and ensuring reliable processing of all URLs.
"""

import os
import re
import json
import tempfile
import pandas as pd
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field

from crewai import LLM
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Agent, Crew, Process, Task
from crewai_tools import FirecrawlScrapeWebsiteTool


# ============================================================================
# Output Models for Structured Classification
# ============================================================================

class URLClassification(BaseModel):
    """Classification result for a single URL."""
    url: str = Field(description="The URL that was analyzed")
    sustainability_marketing: str = Field(description="YES or NO - whether sustainability is used as marketing")
    sustainability_percentage: int = Field(default=0, description="0-100 percentage of sustainability emphasis")
    confidence: float = Field(default=0.5, description="0.0-1.0 confidence score")
    themes: str = Field(default="", description="Comma-separated sustainability themes found")
    reason: str = Field(description="1-2 sentences explaining WHY this classification was made, citing specific evidence from the page content")


class BatchClassificationResult(BaseModel):
    """Classification results for a batch of URLs."""
    classifications: List[URLClassification] = Field(description="List of classification results for each URL")


# ============================================================================
# State Models
# ============================================================================

class BatchState(BaseModel):
    """State model for tracking batch processing progress."""

    # Input configuration
    file_path: str = ""  # Can be local path or Google Drive URL
    output_filename: str = ""
    batch_size: int = 10

    # Downloaded file tracking
    temp_file_path: str = ""  # Path to downloaded temp file
    is_remote: bool = False  # Whether file was downloaded from URL

    # URL data
    all_urls: list[str] = Field(default_factory=list)
    total_urls: int = 0

    # Batch tracking
    current_batch_index: int = 0
    total_batches: int = 0

    # Results accumulation
    all_results: list[dict] = Field(default_factory=list)
    batch_results: list[dict] = Field(default_factory=list)

    # Status tracking
    processed_count: int = 0
    failed_urls: list[str] = Field(default_factory=list)
    is_complete: bool = False
    error_message: str = ""


# ============================================================================
# Mini Crew for Single Batch Processing
# ============================================================================

class GoogleDriveReader:
    """Uses CrewAI Enterprise Google Drive integration to read files."""

    def __init__(self, llm_model: str = "anthropic/claude-sonnet-4-20250514"):
        self.llm = LLM(model=llm_model, temperature=0.7)

    def _create_google_drive_agent(self) -> Agent:
        """Create an agent with Google Drive enterprise apps."""
        return Agent(
            role="Google Drive File Manager",
            goal="Download and read Excel files from Google Drive",
            backstory="""You are a data manager who specializes in accessing files from
            Google Drive. You can search, download, and read Excel files to extract data.""",
            tools=[],
            llm=self.llm,
            allow_delegation=False,
            max_iter=25,
            verbose=True,
            apps=[
                "google_drive/find_file",
                "google_drive/get_file_by_id",
                "google_drive/download_file",
                "google_drive/search_files",
                "google_drive/list_files",
                "google_sheets/get_spreadsheet",
                "google_sheets/get_values",
            ],
        )

    def read_urls_from_drive(self, file_identifier: str) -> list[str]:
        """
        Read URLs from a Google Drive Excel file.

        Args:
            file_identifier: File ID, name, or path in Google Drive

        Returns:
            List of URLs from the CLEAN_LEGACY_URL column
        """
        agent = self._create_google_drive_agent()

        task = Task(
            description=f"""
            Access the Excel file from Google Drive and extract all URLs.

            File identifier: {file_identifier}

            Steps:
            1. Find or access the file using the identifier (could be file ID, name, or path)
            2. Download or read the Excel file content
            3. Find the column named 'CLEAN_LEGACY_URL' (or similar URL column)
            4. Extract ALL URLs from that column
            5. Return the complete list of URLs

            Make sure to get ALL URLs from the file, not just a sample.
            """,
            expected_output="""
            A complete list of all URLs found in the CLEAN_LEGACY_URL column.
            Format as one URL per line, like:
            https://example.com/page1
            https://example.com/page2
            https://example.com/page3
            ...
            """,
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # Parse URLs from result
        urls = []
        raw_result = str(result)
        for line in raw_result.split('\n'):
            line = line.strip()
            if line.startswith('http://') or line.startswith('https://'):
                urls.append(line)

        return urls


class SingleBatchCrew:
    """A lightweight crew that processes URLs using a single agent for scrape+classify."""

    def __init__(self, llm_model: str = "anthropic/claude-sonnet-4-20250514"):
        self.llm = LLM(model=llm_model, temperature=0.7)
        self.scraper_tool = FirecrawlScrapeWebsiteTool()

    def _create_scrape_and_classify_agent(self) -> Agent:
        """Single agent that scrapes and classifies in one flow."""
        return Agent(
            role="Sustainability Marketing Analyst",
            goal="Scrape website content and classify whether sustainability is used as a marketing message",
            backstory="""You are an expert marketing analyst who specializes in web research
            and sustainability messaging classification. You scrape websites to extract their
            marketing content, then analyze whether they use sustainability/environmental
            benefits as a competitive differentiator. You are thorough but conservative -
            only marking content as sustainability marketing when it's explicitly used as
            a selling point, not just incidental mentions.""",
            tools=[self.scraper_tool],
            llm=self.llm,
            allow_delegation=False,
            max_iter=25,
            verbose=True,
        )

    def process_batch(self, urls: list[str], batch_number: int) -> list[dict]:
        """Process a batch of URLs - scrape and classify each one."""

        agent = self._create_scrape_and_classify_agent()

        # Build the URL list for the task
        url_list = chr(10).join(f'{i+1}. {url}' for i, url in enumerate(urls))

        task = Task(
            description=f"""
            Process these {len(urls)} URLs from Batch {batch_number}. For EACH URL:

            STEP 1 - SCRAPE: Use your scraping tool to fetch the page content
            STEP 2 - CLASSIFY: Based on the scraped content, determine if sustainability is used as marketing

            URLs to process:
            {url_list}

            For each URL, after scraping, classify:

            - sustainability_marketing: "YES" if sustainability/environmental benefits are explicitly
              used as a MARKETING MESSAGE or competitive differentiator. "NO" otherwise.

            - sustainability_percentage:
              * 0 if NO
              * 10-20% if minor mention
              * 25-40% if clear but not dominant
              * 50-70% if primary value proposition
              * 80-100% if core marketing message

            - confidence: 0.0-1.0 how confident you are

            - themes (if YES, pick from): Energy Efficiency; Carbon / Embodied Carbon;
              Green Certifications; Environmental Analysis Tools; Climate Resilience;
              Material Sustainability; Operational Emissions; Waste Reduction; Renewables / Clean Energy

            - reason: 1-2 sentences citing SPECIFIC EVIDENCE from the page content you scraped.
              Example: "The page prominently displays 'Reduce your carbon footprint by 40%' as a headline
              and lists LEED certification as a key product benefit."

            IMPORTANT:
            - You MUST scrape each URL before classifying it
            - Base classifications on ACTUAL SCRAPED CONTENT, not assumptions
            - Be conservative - only "YES" for explicit sustainability marketing
            - Return ALL {len(urls)} classifications
            """,
            expected_output=f"""
            Return exactly {len(urls)} classifications as a JSON array. Each object must have:
            - url: the exact URL
            - sustainability_marketing: "YES" or "NO"
            - sustainability_percentage: integer 0-100
            - confidence: float 0.0-1.0
            - themes: string (comma-separated themes or empty)
            - reason: string (1-2 sentences with specific evidence)

            Format:
            [
              {{"url": "...", "sustainability_marketing": "YES", "sustainability_percentage": 45, "confidence": 0.85, "themes": "Energy Efficiency", "reason": "Page headline states 'Save 30% on energy costs' and promotes green building certification."}},
              {{"url": "...", "sustainability_marketing": "NO", "sustainability_percentage": 0, "confidence": 0.9, "themes": "", "reason": "Page focuses on pricing and features with no environmental messaging."}}
            ]
            """,
            agent=agent,
            output_pydantic=BatchClassificationResult,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return self._parse_batch_results(urls, result)

    def _parse_batch_results(self, urls: list[str], crew_result) -> list[dict]:
        """Parse classification results for the batch."""
        results = []
        parsed_by_url = {}

        # Try pydantic output first
        try:
            if hasattr(crew_result, 'pydantic') and crew_result.pydantic:
                batch_result = crew_result.pydantic
                if hasattr(batch_result, 'classifications'):
                    for c in batch_result.classifications:
                        url_key = c.url.lower().strip().rstrip('/')
                        parsed_by_url[url_key] = {
                            "url": c.url,
                            "sustainability_marketing": c.sustainability_marketing.upper(),
                            "sustainability_percentage": c.sustainability_percentage,
                            "confidence": c.confidence,
                            "themes": c.themes,
                            "reason": c.reason
                        }
                    print(f"  ✓ Parsed {len(parsed_by_url)} from structured output")
        except Exception as e:
            print(f"  Note: Pydantic parse issue: {e}")

        # Try JSON extraction if pydantic failed
        if not parsed_by_url:
            raw_result = str(crew_result)
            try:
                json_match = re.search(r'\[[\s\S]*\]', raw_result)
                if json_match:
                    data = json.loads(json_match.group(0))
                    for item in data:
                        if isinstance(item, dict) and 'url' in item:
                            url_key = item['url'].lower().strip().rstrip('/')
                            parsed_by_url[url_key] = {
                                "url": item.get('url', ''),
                                "sustainability_marketing": str(item.get('sustainability_marketing', 'NO')).upper(),
                                "sustainability_percentage": int(item.get('sustainability_percentage', 0)),
                                "confidence": float(item.get('confidence', 0.5)),
                                "themes": str(item.get('themes', '')),
                                "reason": str(item.get('reason', ''))
                            }
                    print(f"  ✓ Parsed {len(parsed_by_url)} from JSON")
            except Exception as e:
                print(f"  Note: JSON parse issue: {e}")

        # Match results to original URLs
        for url in urls:
            url_key = url.lower().strip().rstrip('/')

            if url_key in parsed_by_url:
                results.append(parsed_by_url[url_key])
            else:
                # Try partial match
                matched = False
                for key, value in parsed_by_url.items():
                    if url_key in key or key in url_key:
                        value['url'] = url
                        results.append(value)
                        matched = True
                        break

                if not matched:
                    results.append({
                        "url": url,
                        "sustainability_marketing": "NO",
                        "sustainability_percentage": 0,
                        "confidence": 0.3,
                        "themes": "",
                        "reason": "Could not parse classification from output"
                    })

        return results



# ============================================================================
# Main Batch Processing Flow
# ============================================================================

def is_google_drive_reference(path: str) -> bool:
    """Check if a path is a Google Drive URL or file ID."""
    if path.startswith('http://') or path.startswith('https://'):
        return 'drive.google.com' in path or 'docs.google.com' in path
    # Could also be a file ID (long alphanumeric string)
    return len(path) > 20 and path.replace('-', '').replace('_', '').isalnum()


class SustainabilityBatchFlow(Flow[BatchState]):
    """
    CrewAI Flow for processing sustainability classifications in batches.

    This flow:
    1. Downloads file from Google Drive (if URL provided) or uses local file
    2. Loads URLs from the Excel file
    3. Divides them into configurable batches
    4. Processes each batch with a dedicated crew
    5. Accumulates results and saves to CSV
    """

    def __init__(self, llm_model: str = "anthropic/claude-sonnet-4-20250514"):
        super().__init__()
        self.llm_model = llm_model
        self.temp_dir = None

    @start()
    def load_excel_data(self) -> BatchState:
        """Load URLs from Excel - either local file or Google Drive."""
        print(f"\n{'='*60}")
        print("STEP 1: Loading Excel Data")
        print(f"{'='*60}")

        state = self.state

        try:
            # Check if file_path is a Google Drive reference
            if is_google_drive_reference(state.file_path):
                print(f"📁 Detected Google Drive reference, using Enterprise integration...")
                state.is_remote = True

                # Use CrewAI Enterprise Google Drive integration
                drive_reader = GoogleDriveReader(llm_model=self.llm_model)
                urls = drive_reader.read_urls_from_drive(state.file_path)

                if urls:
                    state.all_urls = urls
                    state.total_urls = len(urls)
                    state.total_batches = (len(urls) + state.batch_size - 1) // state.batch_size
                    state.current_batch_index = 0

                    print(f"✓ Loaded {state.total_urls} URLs from Google Drive")
                    print(f"✓ Will process in {state.total_batches} batches of {state.batch_size} URLs each")
                    return state
                else:
                    state.error_message = "No URLs extracted from Google Drive file"
                    print(f"ERROR: {state.error_message}")
                    return state

            # Local file processing
            print(f"Using local file: {state.file_path}")
            excel_path = state.file_path

            # Read the Excel file
            df = pd.read_excel(excel_path)

            # Find the URL column (try common names)
            url_column = None
            possible_columns = ['CLEAN_LEGACY_URL', 'URL', 'url', 'Website', 'website', 'Link', 'link']

            for col in possible_columns:
                if col in df.columns:
                    url_column = col
                    break

            if url_column is None:
                # Try to find any column with 'url' in name
                for col in df.columns:
                    if 'url' in col.lower():
                        url_column = col
                        break

            if url_column is None:
                state.error_message = f"Could not find URL column. Available columns: {list(df.columns)}"
                print(f"ERROR: {state.error_message}")
                return state

            # Extract URLs, filter out empty/invalid entries
            urls = df[url_column].dropna().astype(str).tolist()
            urls = [url.strip() for url in urls if url.strip() and url.strip().lower() != 'nan']

            state.all_urls = urls
            state.total_urls = len(urls)
            state.total_batches = (len(urls) + state.batch_size - 1) // state.batch_size
            state.current_batch_index = 0

            print(f"✓ Loaded {state.total_urls} URLs from column '{url_column}'")
            print(f"✓ Will process in {state.total_batches} batches of {state.batch_size} URLs each")

        except FileNotFoundError:
            state.error_message = f"File not found: {state.file_path}"
            print(f"ERROR: {state.error_message}")
        except Exception as e:
            state.error_message = f"Error loading Excel file: {str(e)}"
            print(f"ERROR: {state.error_message}")

        return state

    @listen(load_excel_data)
    def process_batches(self) -> BatchState:
        """Process all batches sequentially."""
        print(f"\n{'='*60}")
        print("STEP 2: Processing Batches")
        print(f"{'='*60}")

        state = self.state

        if state.error_message:
            print(f"Skipping batch processing due to error: {state.error_message}")
            return state

        if state.total_urls == 0:
            state.error_message = "No URLs to process"
            print("ERROR: No URLs found in the file")
            return state

        # Initialize the batch crew
        batch_crew = SingleBatchCrew(llm_model=self.llm_model)

        # Process each batch
        for batch_idx in range(state.total_batches):
            state.current_batch_index = batch_idx

            # Calculate batch boundaries
            start_idx = batch_idx * state.batch_size
            end_idx = min(start_idx + state.batch_size, state.total_urls)
            batch_urls = state.all_urls[start_idx:end_idx]

            print(f"\n--- Processing Batch {batch_idx + 1}/{state.total_batches} ---")
            print(f"URLs {start_idx + 1} to {end_idx} ({len(batch_urls)} URLs)")

            try:
                # Process this batch
                batch_results = batch_crew.process_batch(batch_urls, batch_idx + 1)

                # Accumulate results
                state.all_results.extend(batch_results)
                state.processed_count += len(batch_results)

                print(f"✓ Batch {batch_idx + 1} complete: {len(batch_results)} URLs processed")
                print(f"✓ Total progress: {state.processed_count}/{state.total_urls} URLs")

            except Exception as e:
                print(f"✗ Error processing batch {batch_idx + 1}: {str(e)}")
                # Add failed URLs to tracking
                state.failed_urls.extend(batch_urls)
                # Continue with next batch
                continue

        return state

    @listen(process_batches)
    def save_results(self) -> BatchState:
        """Save all accumulated results to a CSV file."""
        print(f"\n{'='*60}")
        print("STEP 3: Saving Results")
        print(f"{'='*60}")

        state = self.state

        if state.error_message and not state.all_results:
            print(f"Cannot save results due to error: {state.error_message}")
            return state

        if not state.all_results:
            print("No results to save")
            state.error_message = "No results were generated"
            return state

        try:
            # Create DataFrame from results
            results_df = pd.DataFrame(state.all_results)

            # Ensure proper column order
            column_order = ['url', 'sustainability_marketing', 'sustainability_percentage',
                          'confidence', 'themes', 'reason']

            # Add any missing columns with defaults
            for col in column_order:
                if col not in results_df.columns:
                    results_df[col] = ""

            results_df = results_df[column_order]

            # Determine output filename
            if state.output_filename:
                output_path = state.output_filename
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                output_path = f"sustainability_classification_results_{timestamp}.csv"

            # Save to CSV
            results_df.to_csv(output_path, index=False)

            print(f"✓ Results saved to: {output_path}")
            print(f"✓ Total URLs processed: {len(state.all_results)}/{state.total_urls}")

            if state.failed_urls:
                print(f"✗ Failed URLs: {len(state.failed_urls)}")
                failed_path = output_path.replace('.csv', '_failed.txt')
                with open(failed_path, 'w') as f:
                    f.write('\n'.join(state.failed_urls))
                print(f"✓ Failed URLs saved to: {failed_path}")

            state.is_complete = True

        except Exception as e:
            state.error_message = f"Error saving results: {str(e)}"
            print(f"ERROR: {state.error_message}")

        return state

    @listen(save_results)
    def finalize(self) -> BatchState:
        """Print final summary and return state."""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")

        state = self.state

        print(f"""
Summary:
--------
- Total URLs in file: {state.total_urls}
- URLs processed successfully: {state.processed_count}
- URLs failed: {len(state.failed_urls)}
- Batches processed: {state.total_batches}
- Batch size: {state.batch_size}
- Status: {'SUCCESS' if state.is_complete else 'COMPLETED WITH ERRORS'}
""")

        if state.error_message:
            print(f"Errors encountered: {state.error_message}")

        return state


# ============================================================================
# Entry Points
# ============================================================================

def run_batch_flow(
    file_path: str,
    output_filename: Optional[str] = None,
    batch_size: int = 10,
    llm_model: str = "anthropic/claude-sonnet-4-20250514"
) -> BatchState:
    """
    Run the batch processing flow.

    Args:
        file_path: Path to the Excel file containing URLs
        output_filename: Output CSV filename (optional, auto-generated if not provided)
        batch_size: Number of URLs to process per batch (default: 10)
        llm_model: LLM model to use for processing (default: gpt-4o)

    Returns:
        BatchState with processing results and status
    """
    # Initialize the flow with starting state
    initial_state = BatchState(
        file_path=file_path,
        output_filename=output_filename or "",
        batch_size=batch_size,
    )

    # Create and run the flow
    flow = SustainabilityBatchFlow(llm_model=llm_model)

    # Kickoff returns the final state
    final_state = flow.kickoff(inputs=initial_state.model_dump())

    return final_state


def main():
    """Main entry point for batch processing."""
    import sys

    # Default values
    file_path = "Classifier Test File 1-2.xlsx"
    output_filename = None
    batch_size = 10

    # Parse command line arguments
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     Sustainability Classifier - Batch Processing Flow        ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
- Input file: {file_path}
- Output file: {output_filename or 'auto-generated'}
- Batch size: {batch_size}
""")

    result = run_batch_flow(
        file_path=file_path,
        output_filename=output_filename,
        batch_size=batch_size,
    )

    if result.is_complete:
        print("\n✓ Batch processing completed successfully!")
        return 0
    else:
        print(f"\n✗ Batch processing completed with errors: {result.error_message}")
        return 1


if __name__ == "__main__":
    exit(main())
