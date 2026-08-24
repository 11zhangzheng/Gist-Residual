from pathlib import Path

from fidmem.router.dagger import DAggerConfig, run_dagger

from tests.router.test_dagger_workflow import SpyTrainer, _always_stop, _contexts


def test_resume_identity_is_independent_of_context_input_order(tmp_path: Path) -> None:
    contexts = _contexts(2)
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    config = DAggerConfig(artifact_root=tmp_path)
    first = run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=source,
        trainer=SpyTrainer(),
        config=config,
    )

    resumed = run_dagger(
        train_contexts=tuple(reversed(contexts)),
        dev_contexts=tuple(reversed(contexts)),
        initial_policy=_always_stop,
        source_policy_checkpoint=source,
        trainer=SpyTrainer(),
        config=config,
    )

    assert resumed.resumed is True
    assert resumed.run_identity == first.run_identity
