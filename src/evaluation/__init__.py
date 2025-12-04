"""Evaluation framework for literature search and review quality assessment."""

from .criteria import LITERATURE_CRITERIA, REVIEW_CRITERIA
from .literature_eval import LiteratureEvaluator, LiteratureEvalResult
from .review_eval import ReviewEvaluator, ReviewEvalResult
from .compare_providers import ProviderComparison, compare_all_providers
from .ab_test import (
    ABTestConfig,
    ABTestResult,
    ABTestSession,
    ABTestRunner,
    create_default_configs,
    run_ab_test,
)

__all__ = [
    "LITERATURE_CRITERIA",
    "REVIEW_CRITERIA",
    "LiteratureEvaluator",
    "LiteratureEvalResult",
    "ReviewEvaluator",
    "ReviewEvalResult",
    "ProviderComparison",
    "compare_all_providers",
    "ABTestConfig",
    "ABTestResult",
    "ABTestSession",
    "ABTestRunner",
    "create_default_configs",
    "run_ab_test",
]

