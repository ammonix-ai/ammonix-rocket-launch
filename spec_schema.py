"""Structured specification schema for Ammonix platform domains.

Defines Python dataclasses that mirror the spec.yaml format. Scripts validate
and consume spec.yaml through DomainSpec.from_yaml() instead of parsing markdown.

Both spec.md (human-readable) and spec.yaml (machine-readable) coexist per domain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CohortSpec:
    """Single cohort file specification."""

    file: str
    patient_count: int
    role: str  # "inference" | "training" | "enrichment" | "validation"
    split_source: Optional[str] = None


@dataclass
class DatabaseSpec:
    """Clinical database specification (framework Section 6, input 1)."""

    source_description: str
    total_patients: int
    signal_type: str  # "launch telemetry", "EEG", etc.
    data_format: str  # "jsonl", "parquet"
    cohorts: Dict[str, CohortSpec] = field(default_factory=dict)


@dataclass
class KeyFeatureSpec:
    """A key computed feature with expected range."""

    name: str
    qpsi_key: str
    normal_range: str
    unit: str


@dataclass
class SignalProcessingSpec:
    """Signal processing framework (framework Section 6, input 2)."""

    method: str  # e.g. "causal phase-windowed features"
    expected_feature_count: int
    feature_families: Dict[str, str] = field(default_factory=dict)
    key_features: List[KeyFeatureSpec] = field(default_factory=list)


@dataclass
class BuildingBlockSpec:
    """Building block specification (framework Section 6, input 3)."""

    classifier_type: str  # "xgboost_ovr"
    training_cohort: str
    canonical_diagnosis_count: int
    clinical_categories: List[str] = field(default_factory=list)
    pipeline_stages: List[str] = field(default_factory=list)
    error_correction: List[str] = field(default_factory=list)


@dataclass
class LLMRoleSpec:
    """LLM role configuration (M1 or M2)."""

    temperature: float
    max_tokens: int
    role: str  # "diagnostician" | "architect"


@dataclass
class ProductVisionSpec:
    """Product vision (framework Section 6, input 4)."""

    product_type: str  # "static_diagnosis" | "realtime_guidance"
    llm_model: str
    m1_config: LLMRoleSpec = field(
        default_factory=lambda: LLMRoleSpec(0.0, 512, "diagnostician")
    )
    m2_config: LLMRoleSpec = field(
        default_factory=lambda: LLMRoleSpec(0.7, 6144, "architect")
    )
    frontend_type: str = "react"
    features: List[str] = field(default_factory=list)


@dataclass
class AcceptanceCriteria:
    """Output validation criteria (framework Section 6, input 5)."""

    overall_f1_target: float
    inference_time_ms: float
    validation_cohort: str
    per_diagnosis_targets: Dict[str, float] = field(default_factory=dict)
    auditability_requirements: List[str] = field(default_factory=list)


@dataclass
class DomainSpec:
    """Top-level domain specification consumed by all factory scripts."""

    domain: str
    version: str
    database: DatabaseSpec
    signal_processing: SignalProcessingSpec
    building_blocks: BuildingBlockSpec
    product_vision: ProductVisionSpec
    acceptance: AcceptanceCriteria

    @classmethod
    def from_yaml(cls, path: Path) -> "DomainSpec":
        """Load and validate a spec.yaml file.

        Args:
            path: Path to spec.yaml

        Returns:
            Validated DomainSpec instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If spec fails validation
        """
        import yaml  # lazy import — only needed when loading

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        spec = _parse_raw(raw)
        errors = spec.validate()
        if errors:
            raise ValueError(
                f"spec.yaml validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return spec

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors: List[str] = []

        if not self.domain:
            errors.append("domain is required")
        if not self.version:
            errors.append("version is required")

        # Database checks
        if self.database.total_patients < 1:
            errors.append("database.total_patients must be positive")
        if not self.database.cohorts:
            errors.append("database.cohorts must have at least one cohort")

        roles = {c.role for c in self.database.cohorts.values()}
        if "training" not in roles:
            errors.append("database.cohorts must include a 'training' cohort")

        # Signal processing
        if self.signal_processing.expected_feature_count < 1:
            errors.append("signal_processing.expected_feature_count must be positive")

        # Building blocks
        if self.building_blocks.canonical_diagnosis_count < 1:
            errors.append("building_blocks.canonical_diagnosis_count must be positive")
        if not self.building_blocks.pipeline_stages:
            errors.append("building_blocks.pipeline_stages must not be empty")

        # Acceptance
        if not (0.0 < self.acceptance.overall_f1_target <= 1.0):
            errors.append("acceptance.overall_f1_target must be in (0, 1]")
        if self.acceptance.inference_time_ms <= 0:
            errors.append("acceptance.inference_time_ms must be positive")
        if self.acceptance.validation_cohort not in self.database.cohorts:
            errors.append(
                f"acceptance.validation_cohort '{self.acceptance.validation_cohort}' "
                f"not in database.cohorts"
            )

        return errors

    def get_cohorts_by_role(self, role: str) -> Dict[str, CohortSpec]:
        """Return cohorts filtered by role."""
        return {
            name: c for name, c in self.database.cohorts.items() if c.role == role
        }

    def training_cohort(self) -> Optional[CohortSpec]:
        """Return the training cohort (or None)."""
        training = self.get_cohorts_by_role("training")
        return next(iter(training.values()), None)

    def validation_cohort(self) -> Optional[CohortSpec]:
        """Return the validation cohort referenced by acceptance criteria."""
        return self.database.cohorts.get(self.acceptance.validation_cohort)


def _parse_raw(raw: Dict[str, Any]) -> DomainSpec:
    """Parse a raw YAML dict into DomainSpec dataclasses."""
    db_raw = raw.get("database", {})
    cohorts = {}
    for name, c in db_raw.get("cohorts", {}).items():
        cohorts[name] = CohortSpec(
            file=c.get("file", ""),
            patient_count=c.get("patient_count", 0),
            role=c.get("role", ""),
            split_source=c.get("split_source"),
        )

    sp_raw = raw.get("signal_processing", {})
    key_features = []
    for kf in sp_raw.get("key_features", []):
        key_features.append(KeyFeatureSpec(
            name=kf.get("name", ""),
            qpsi_key=kf.get("qpsi_key", ""),
            normal_range=kf.get("normal_range", ""),
            unit=kf.get("unit", ""),
        ))

    bb_raw = raw.get("building_blocks", {})
    pv_raw = raw.get("product_vision", {})

    m1_raw = pv_raw.get("m1_config", {})
    m2_raw = pv_raw.get("m2_config", {})
    m1 = LLMRoleSpec(
        temperature=m1_raw.get("temperature", 0.0),
        max_tokens=m1_raw.get("max_tokens", 512),
        role=m1_raw.get("role", "diagnostician"),
    )
    m2 = LLMRoleSpec(
        temperature=m2_raw.get("temperature", 0.7),
        max_tokens=m2_raw.get("max_tokens", 6144),
        role=m2_raw.get("role", "architect"),
    )

    acc_raw = raw.get("acceptance", {})

    return DomainSpec(
        domain=raw.get("domain", ""),
        version=raw.get("version", ""),
        database=DatabaseSpec(
            source_description=db_raw.get("source_description", ""),
            total_patients=db_raw.get("total_patients", 0),
            signal_type=db_raw.get("signal_type", ""),
            data_format=db_raw.get("data_format", ""),
            cohorts=cohorts,
        ),
        signal_processing=SignalProcessingSpec(
            method=sp_raw.get("method", ""),
            expected_feature_count=sp_raw.get("expected_feature_count", 0),
            feature_families=sp_raw.get("feature_families", {}),
            key_features=key_features,
        ),
        building_blocks=BuildingBlockSpec(
            classifier_type=bb_raw.get("classifier_type", ""),
            training_cohort=bb_raw.get("training_cohort", ""),
            canonical_diagnosis_count=bb_raw.get("canonical_diagnosis_count", 0),
            clinical_categories=bb_raw.get("clinical_categories", []),
            pipeline_stages=bb_raw.get("pipeline_stages", []),
            error_correction=bb_raw.get("error_correction", []),
        ),
        product_vision=ProductVisionSpec(
            product_type=pv_raw.get("product_type", ""),
            llm_model=pv_raw.get("llm_model", ""),
            m1_config=m1,
            m2_config=m2,
            frontend_type=pv_raw.get("frontend_type", "react"),
            features=pv_raw.get("features", []),
        ),
        acceptance=AcceptanceCriteria(
            overall_f1_target=acc_raw.get("overall_f1_target", 0.0),
            inference_time_ms=acc_raw.get("inference_time_ms", 0.0),
            validation_cohort=acc_raw.get("validation_cohort", ""),
            per_diagnosis_targets=acc_raw.get("per_diagnosis_targets", {}),
            auditability_requirements=acc_raw.get("auditability_requirements", []),
        ),
    )
