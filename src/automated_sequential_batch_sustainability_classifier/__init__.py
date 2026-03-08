"""
Automated Sequential Batch Sustainability Classifier

A CrewAI project for classifying sustainability marketing in web content.
Supports batch processing of large files using CrewAI Flows.
"""

from automated_sequential_batch_sustainability_classifier.crew import (
    AutomatedSequentialBatchSustainabilityClassifierCrew,
)


def get_batch_flow():
    """Get the batch flow module for processing large files."""
    from automated_sequential_batch_sustainability_classifier.batch_flow import (
        SustainabilityBatchFlow,
        run_batch_flow,
    )
    return SustainabilityBatchFlow, run_batch_flow


__all__ = [
    "AutomatedSequentialBatchSustainabilityClassifierCrew",
    "get_batch_flow",
]
