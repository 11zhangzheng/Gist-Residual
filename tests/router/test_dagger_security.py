import pytest

from fidmem.actions.environment import ActionObservation, MemoryEnvironment
from fidmem.router.dagger import DAggerQuestionContext

from tests.router.test_dagger_workflow import _contexts


def test_question_context_requires_forbidden_executor_in_real_environment() -> None:
    context = _contexts(1)[0]
    unsafe = MemoryEnvironment(
        events=context.environment.canonical_events,
        executor=lambda action, state: ActionObservation(),
        costs=context.environment.costs,
    )

    with pytest.raises(ValueError, match="ForbiddenObservationGenerator"):
        DAggerQuestionContext.model_validate(
            context.model_dump(mode="python") | {"environment": unsafe}
        )
