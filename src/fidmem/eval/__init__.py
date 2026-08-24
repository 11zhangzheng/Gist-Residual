"""Fair evaluation policies, raw records, metrics, and error attribution."""

from .baselines import (
    BCPolicyAdapter,
    DAggerPolicyAdapter,
    FullResidualPolicy,
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
    PromptControllerPolicy,
    QuestionOnlyPolicy,
    RulePolicy,
    TextAdaptivePolicy,
    UniformFramesPolicy,
)
from .error_taxonomy import ErrorCause, ErrorSignals, classify_error
from .metrics import (
    ResourceUsage,
    RunPoint,
    cost_at_accuracy,
    fixed_budget_accuracy,
    pareto_frontier,
    summarize_results,
)
from .runner import evaluate_run

__all__ = [
    "BCPolicyAdapter",
    "DAggerPolicyAdapter",
    "ErrorCause",
    "ErrorSignals",
    "FullResidualPolicy",
    "GistOnlyPolicy",
    "GistResidualPolicy",
    "GistVisualPolicy",
    "PromptControllerPolicy",
    "QuestionOnlyPolicy",
    "ResourceUsage",
    "RulePolicy",
    "RunPoint",
    "TextAdaptivePolicy",
    "UniformFramesPolicy",
    "classify_error",
    "cost_at_accuracy",
    "evaluate_run",
    "fixed_budget_accuracy",
    "pareto_frontier",
    "summarize_results",
]
