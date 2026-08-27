"""Compatibility adapter for the canonical experiments observation importer."""

from fidmem.experiments.observation_import import (
    ObservationImportRecord,
    import_production_observations,
)

ProductionObservationImportRecord = ObservationImportRecord

__all__ = [
    "ProductionObservationImportRecord",
    "import_production_observations",
]
