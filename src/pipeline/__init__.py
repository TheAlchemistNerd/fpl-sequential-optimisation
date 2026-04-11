"""Orchestration layer — wires data → features → models → decisions."""

from src.pipeline import data_pipeline

__all__ = ["data_pipeline"]
