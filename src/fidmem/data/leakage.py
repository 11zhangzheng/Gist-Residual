"""Read-only, three-layer train/evaluation video leakage audit."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, Sequence

import duckdb

from .video import probe_video, sample_frames


EmbeddingProvider = Callable[[Path, tuple[Path, ...]], Sequence[Sequence[float]]]

class LeakageAuditError(ValueError):
    """Malformed numeric audit evidence that must fail closed."""


def _validated_vector(
    vector: Sequence[float],
    *,
    label: str,
) -> tuple[tuple[float, ...], float]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise LeakageAuditError(f"{label} must be non-empty")
    if not all(math.isfinite(value) for value in values):
        raise LeakageAuditError(f"{label} values must be finite")
    norm = math.hypot(*values)
    if not math.isfinite(norm):
        raise LeakageAuditError(f"{label} norm must be finite")
    if norm <= 0:
        raise LeakageAuditError(f"{label} must be a non-zero vector")
    return values, norm


def _validated_embeddings(
    embeddings: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in embeddings)
    if not rows:
        raise LeakageAuditError("at least one frame embedding is required")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise LeakageAuditError(
            "frame embeddings must be non-empty and equal-width"
        )
    for row in rows:
        _validated_vector(row, label="frame embedding")
    return rows


@dataclass(frozen=True)
class VideoAsset:
    """The minimal immutable metadata needed to audit one source video."""

    video_id: str
    path: Path
    frame_embeddings: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.frame_embeddings is not None:
            embeddings = _validated_embeddings(self.frame_embeddings)
            object.__setattr__(
                self,
                "frame_embeddings",
                embeddings,
            )


@dataclass(frozen=True)
class LeakageFinding:
    train_video_id: str
    eval_video_id: str
    kind: str
    cosine_similarity: float | None


@dataclass(frozen=True)
class LeakageReport:
    findings: tuple[LeakageFinding, ...]
    parquet_path: Path


def _normalise_video_id(video_id: str) -> str | None:
    stem = Path(video_id).stem.casefold()
    normalised = "".join(character for character in stem if character.isalnum())
    return normalised or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centroid(embeddings: Sequence[Sequence[float]]) -> tuple[float, ...]:
    rows = _validated_embeddings(embeddings)
    values: list[float] = []
    for index in range(len(rows[0])):
        total = sum(row[index] for row in rows)
        if not math.isfinite(total):
            raise LeakageAuditError("embedding centroid arithmetic must be finite")
        value = total / len(rows)
        if not math.isfinite(value):
            raise LeakageAuditError("embedding centroid values must be finite")
        values.append(value)
    centroid, _ = _validated_vector(values, label="embedding centroid")
    return centroid


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise LeakageAuditError(
            "embedding centroids must have matching dimensions"
        )
    left_values, left_norm = _validated_vector(
        left, label="left embedding centroid"
    )
    right_values, right_norm = _validated_vector(
        right, label="right embedding centroid"
    )
    denominator = left_norm * right_norm
    if not math.isfinite(denominator) or denominator <= 0:
        raise LeakageAuditError("cosine denominator must be finite and positive")
    products = tuple(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    if not all(math.isfinite(value) for value in products):
        raise LeakageAuditError("cosine numerator terms must be finite")
    numerator = sum(products)
    if not math.isfinite(numerator):
        raise LeakageAuditError("cosine numerator must be finite")
    similarity = numerator / denominator
    if not math.isfinite(similarity):
        raise LeakageAuditError("cosine similarity must be finite")
    return similarity


class LeakageAuditor:
    """Write an audit report only; this class never mutates source data."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        sample_count: int = 8,
        near_duplicate_threshold: float = 0.985,
    ) -> None:
        if sample_count != 8:
            raise ValueError("the leakage protocol requires exactly eight sampled frames")
        self.parquet_path = Path(parquet_path)
        self.embedding_provider = embedding_provider
        self.sample_count = sample_count
        self.near_duplicate_threshold = near_duplicate_threshold

    def _embedding_centroid(self, asset: VideoAsset) -> tuple[float, ...] | None:
        if asset.frame_embeddings is not None:
            if len(asset.frame_embeddings) != self.sample_count:
                raise ValueError(
                    "precomputed frame embeddings must contain exactly eight frames"
                )
            return _centroid(asset.frame_embeddings)
        if self.embedding_provider is None:
            return None
        metadata = probe_video(asset.path)
        timestamps = tuple(
            metadata.duration_sec * (index + 0.5) / self.sample_count
            for index in range(self.sample_count)
        )
        with TemporaryDirectory(prefix="fidmem-leakage-") as directory:
            frames = sample_frames(asset.path, timestamps, directory)
            embeddings = self.embedding_provider(asset.path, frames)
            if len(embeddings) != self.sample_count:
                raise ValueError("embedding provider must return exactly eight frame embeddings")
            return _centroid(embeddings)

    def _write_parquet(self, findings: Sequence[LeakageFinding]) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect()
        try:
            connection.execute(
                "CREATE TABLE audit (train_video_id VARCHAR, eval_video_id VARCHAR, kind VARCHAR, cosine_similarity DOUBLE)"
            )
            rows = [
                (
                    finding.train_video_id,
                    finding.eval_video_id,
                    finding.kind,
                    finding.cosine_similarity,
                )
                for finding in findings
            ]
            if rows:
                connection.executemany(
                    "INSERT INTO audit VALUES (?, ?, ?, ?)",
                    rows,
                )
            quoted_path = str(self.parquet_path.resolve()).replace("'", "''")
            connection.execute(f"COPY audit TO '{quoted_path}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def audit(
        self, train: Iterable[VideoAsset], eval: Iterable[VideoAsset], *, require_coverage: bool = False
    ) -> LeakageReport:
        """Compare split pairs by normalized ID, SHA-256, then cosine similarity."""
        train_assets = tuple(train)
        eval_assets = tuple(eval)
        hashes = {asset: _sha256(asset.path) for asset in (*train_assets, *eval_assets)}
        centroids: dict[VideoAsset, tuple[float, ...] | None] = {}
        if require_coverage:
            for asset in (*train_assets, *eval_assets):
                centroid = self._embedding_centroid(asset)
                if centroid is None:
                    raise ValueError("eight-frame centroid coverage is required for every audited asset")
                centroids[asset] = centroid
        findings: list[LeakageFinding] = []

        for train_asset in train_assets:
            for eval_asset in eval_assets:
                train_id = _normalise_video_id(train_asset.video_id)
                eval_id = _normalise_video_id(eval_asset.video_id)
                if train_id is not None and train_id == eval_id:
                    findings.append(
                        LeakageFinding(train_asset.video_id, eval_asset.video_id, "id_duplicate", None)
                    )
                    continue
                if hashes[train_asset] == hashes[eval_asset]:
                    findings.append(
                        LeakageFinding(train_asset.video_id, eval_asset.video_id, "hash_duplicate", None)
                    )
                    continue
                if train_asset not in centroids:
                    centroids[train_asset] = self._embedding_centroid(train_asset)
                if eval_asset not in centroids:
                    centroids[eval_asset] = self._embedding_centroid(eval_asset)
                train_centroid = centroids[train_asset]
                eval_centroid = centroids[eval_asset]
                if train_centroid is None or eval_centroid is None:
                    continue
                similarity = _cosine(train_centroid, eval_centroid)
                if similarity >= self.near_duplicate_threshold:
                    findings.append(
                        LeakageFinding(
                            train_asset.video_id,
                            eval_asset.video_id,
                            "near_duplicate",
                            similarity,
                        )
                    )

        report = LeakageReport(tuple(findings), self.parquet_path)
        self._write_parquet(report.findings)
        return report
