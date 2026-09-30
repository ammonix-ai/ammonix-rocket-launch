"""Rule Discovery Library: automated rule formulation from labelled cases (here: launches).

Implements the 5-phase methodology:
  1. Correlation analysis
  2. Candidate rule formulation
  3. Evidence grading
  4. Validation
  5. Integration into rules.json
"""

from discovery.types import (
    CandidateRule,
    DiscoveryReport,
    EquivalenceCandidate,
    FeatureDiagnosisCorrelation,
    MutexCandidate,
)

__all__ = [
    "CandidateRule",
    "DiscoveryReport",
    "EquivalenceCandidate",
    "FeatureDiagnosisCorrelation",
    "MutexCandidate",
]
