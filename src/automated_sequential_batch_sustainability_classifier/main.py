#!/usr/bin/env python
"""
Main entry point for the Sustainability Classifier.

Uses CrewAI Flows for batch processing of large Excel files.
"""

import sys
import os


def run():
    """
    Run the batch processing flow.

    This is the default entry point for `crewai run`.
    """
    from automated_sequential_batch_sustainability_classifier.batch_flow import run_batch_flow, is_google_drive_reference

    # Default configuration - can be local path OR Google Drive file ID/URL
    file_path = "Classifier Test File 1-2.xlsx"
    output_filename = None
    batch_size = 5

    # Check for environment variable override
    if os.environ.get('CLASSIFIER_FILE_URL'):
        file_path = os.environ.get('CLASSIFIER_FILE_URL')
    elif os.environ.get('GOOGLE_DRIVE_FILE_ID'):
        file_path = os.environ.get('GOOGLE_DRIVE_FILE_ID')

    is_remote = is_google_drive_reference(file_path)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     Sustainability Classifier - Batch Processing Flow        ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
- Input: {file_path[:60] + '...' if len(file_path) > 60 else file_path}
- Source: {'Google Drive (Enterprise Integration)' if is_remote else 'Local file'}
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
        print(f"  Processed: {result.processed_count} URLs")
        print(f"  Failed: {len(result.failed_urls)} URLs")
    else:
        print(f"\n✗ Batch processing completed with errors: {result.error_message}")


def run_batch():
    """
    Run batch processing flow with command line arguments.

    Usage: run_batch [file_path] [output_filename] [batch_size]
    """
    from automated_sequential_batch_sustainability_classifier.batch_flow import run_batch_flow, is_google_drive_reference

    # Default values - supports local path OR Google Drive file ID/URL
    file_path = "Classifier Test File 1-2.xlsx"
    output_filename = None
    batch_size = 5

    # Parse command line arguments (skip script name and command)
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    if len(args) > 0:
        file_path = args[0]
    if len(args) > 1:
        output_filename = args[1]
    if len(args) > 2:
        batch_size = int(args[2])

    is_remote = is_google_drive_reference(file_path)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     Sustainability Classifier - Batch Processing Flow        ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
- Input: {file_path[:60] + '...' if len(file_path) > 60 else file_path}
- Source: {'Google Drive (Enterprise Integration)' if is_remote else 'Local file'}
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
        print(f"  Processed: {result.processed_count} URLs")
        print(f"  Failed: {len(result.failed_urls)} URLs")
    else:
        print(f"\n✗ Batch processing completed with errors: {result.error_message}")


def run_batch_interactive():
    """
    Run batch processing with interactive prompts for configuration.
    """
    from automated_sequential_batch_sustainability_classifier.batch_flow import run_batch_flow

    print("""
╔══════════════════════════════════════════════════════════════╗
║   Sustainability Classifier - Interactive Batch Processing   ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Get file path
    default_file = "Classifier Test File 1-2.xlsx"
    file_path = input(f"Enter Excel file path [{default_file}]: ").strip()
    if not file_path:
        file_path = default_file

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' not found in current directory.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Aborted.")
            return

    # Get output filename
    output_filename = input("Enter output filename (leave blank for auto-generated): ").strip()
    if not output_filename:
        output_filename = None

    # Get batch size
    batch_size_str = input("Enter batch size [10]: ").strip()
    batch_size = int(batch_size_str) if batch_size_str else 10

    print(f"\nStarting batch processing with:")
    print(f"  - Input: {file_path}")
    print(f"  - Output: {output_filename or 'auto-generated'}")
    print(f"  - Batch size: {batch_size}")
    print()

    result = run_batch_flow(
        file_path=file_path,
        output_filename=output_filename,
        batch_size=batch_size,
    )

    if result.is_complete:
        print("\n✓ Batch processing completed successfully!")
        print(f"  Processed {result.processed_count} URLs")
    else:
        print(f"\n✗ Batch processing completed with errors: {result.error_message}")


def run_single():
    """
    Run the original single-batch crew (legacy mode).
    """
    from automated_sequential_batch_sustainability_classifier.crew import (
        AutomatedSequentialBatchSustainabilityClassifierCrew
    )

    inputs = {
        'file_path': 'Classifier Test File 1-2.xlsx',
        'batch_number': '1',
        'output_filename': 'sustainability_results.csv'
    }
    AutomatedSequentialBatchSustainabilityClassifierCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    from automated_sequential_batch_sustainability_classifier.crew import (
        AutomatedSequentialBatchSustainabilityClassifierCrew
    )

    inputs = {
        'file_path': 'sample_value',
        'batch_number': 'sample_value',
        'output_filename': 'sample_value'
    }
    try:
        AutomatedSequentialBatchSustainabilityClassifierCrew().crew().train(
            n_iterations=int(sys.argv[2]),
            filename=sys.argv[3],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    from automated_sequential_batch_sustainability_classifier.crew import (
        AutomatedSequentialBatchSustainabilityClassifierCrew
    )

    try:
        AutomatedSequentialBatchSustainabilityClassifierCrew().crew().replay(task_id=sys.argv[2])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    from automated_sequential_batch_sustainability_classifier.crew import (
        AutomatedSequentialBatchSustainabilityClassifierCrew
    )

    inputs = {
        'file_path': 'sample_value',
        'batch_number': 'sample_value',
        'output_filename': 'sample_value'
    }
    try:
        AutomatedSequentialBatchSustainabilityClassifierCrew().crew().test(
            n_iterations=int(sys.argv[2]),
            openai_model_name=sys.argv[3],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def print_usage():
    """Print usage information."""
    print("""
Sustainability Classifier - Usage
==================================

Commands:
  run              Run the batch processing flow (default)
  run_batch        Run batch processing with custom arguments
                   Usage: run_batch [file_path_or_id] [output_filename] [batch_size]
  run_interactive  Run batch processing with interactive prompts
  run_single       Run the original single-batch crew (legacy)
  train            Train the crew
  replay           Replay crew execution from a task
  test             Test the crew execution

Input Sources (uses CrewAI Enterprise Google Drive integration):
  - Local file:      "My Data.xlsx"
  - Google Drive ID: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
  - Google Drive URL: "https://drive.google.com/file/d/FILE_ID/view"

Examples:
  # Run with defaults (processes Classifier Test File 1-2.xlsx)
  crewai run

  # Run with local file
  python main.py run_batch "My Data.xlsx"

  # Run with Google Drive file ID (Enterprise integration)
  python main.py run_batch "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"

  # Run with Google Drive URL
  python main.py run_batch "https://drive.google.com/file/d/abc123/view"

  # Run with all custom options
  python main.py run_batch "My Data.xlsx" "output.csv" 5

  # Set file ID via environment variable
  export GOOGLE_DRIVE_FILE_ID="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
  crewai run

  # Interactive mode
  python main.py run_interactive
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to run() when no command specified
        run()
    else:
        command = sys.argv[1]

        if command == "run":
            run()
        elif command == "run_batch":
            run_batch()
        elif command == "run_interactive":
            run_batch_interactive()
        elif command == "run_single":
            run_single()
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        elif command in ["--help", "-h", "help"]:
            print_usage()
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
