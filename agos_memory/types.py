from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class Partition:
  """Caller-owned placement identity. It grants no authority."""

  label: str
  key: str | None = None

  def __post_init__(self) -> None:
    _required_text(self.label, error="memory_admission_partition_label_invalid")
    if self.key is not None:
      _required_text(self.key, error="memory_admission_partition_key_invalid")


@dataclass(frozen=True, slots=True)
class Proposal:
  partition: Partition
  kind: str
  text: str
  confidence: float
  source_refs: tuple[str, ...] = ()
  supersedes: tuple[str, ...] = ()
  expires_days: int | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.partition, Partition):
      raise ValueError("memory_admission_proposal_partition_invalid")
    _required_text(self.kind, error="memory_admission_proposal_kind_invalid")
    if not isinstance(self.text, str):
      raise ValueError("memory_admission_proposal_text_invalid")
    if not _unit_interval(self.confidence):
      raise ValueError("memory_admission_confidence_invalid")
    _text_tuple(self.source_refs, error="memory_admission_source_refs_invalid")
    _text_tuple(self.supersedes, error="memory_admission_supersedes_invalid")
    if self.expires_days is not None and not _positive_int(self.expires_days):
      raise ValueError("memory_admission_expiry_invalid")


@dataclass(frozen=True, slots=True)
class ExistingRecord:
  record_id: str
  partition: Partition
  kind: str
  text: str
  replaceable: bool = True

  def __post_init__(self) -> None:
    _required_text(self.record_id, error="memory_admission_existing_record_id_invalid")
    if not isinstance(self.partition, Partition):
      raise ValueError("memory_admission_existing_partition_invalid")
    _required_text(self.kind, error="memory_admission_existing_kind_invalid")
    if not isinstance(self.text, str):
      raise ValueError("memory_admission_existing_text_invalid")
    if not isinstance(self.replaceable, bool):
      raise ValueError("memory_admission_existing_replaceable_invalid")


@dataclass(frozen=True, slots=True)
class PartitionRule:
  label: str
  missing_key_reason: str | None = None
  allowed_kinds: frozenset[str] | None = None
  disallowed_kind_reason: str = "kind_not_allowed"

  def __post_init__(self) -> None:
    _required_text(self.label, error="memory_admission_partition_rule_label_invalid")
    if self.missing_key_reason is not None:
      _required_text(
        self.missing_key_reason,
        error="memory_admission_partition_rule_missing_key_reason_invalid",
      )
    if self.allowed_kinds is not None:
      _text_frozenset(
        self.allowed_kinds,
        error="memory_admission_partition_rule_allowed_kinds_invalid",
      )
    _required_text(
      self.disallowed_kind_reason,
      error="memory_admission_partition_rule_disallowed_kind_reason_invalid",
    )


@dataclass(frozen=True, slots=True)
class AdmissionRules:
  min_confidence: float = 0.5
  max_text_chars: int = 1_000
  partition_rules: tuple[PartitionRule, ...] = ()

  def __post_init__(self) -> None:
    if not _unit_interval(self.min_confidence):
      raise ValueError("memory_admission_min_confidence_invalid")
    if not _positive_int(self.max_text_chars):
      raise ValueError("memory_admission_max_text_chars_invalid")
    if not isinstance(self.partition_rules, tuple) or any(
      not isinstance(rule, PartitionRule) for rule in self.partition_rules
    ):
      raise ValueError("memory_admission_partition_rules_invalid")


@dataclass(frozen=True, slots=True)
class Accept:
  proposal: Proposal
  expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Replace:
  proposal: Proposal
  replaced_record_ids: tuple[str, ...]
  expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Reject:
  proposal: Proposal
  reason: str


@dataclass(frozen=True, slots=True)
class RetentionState:
  updated_at: datetime | None = None
  expires_at: datetime | None = None
  exposure_count: int = 0
  attributed_use_count: int = 0

  def __post_init__(self) -> None:
    _optional_aware(self.updated_at, field="updated_at")
    _optional_aware(self.expires_at, field="expires_at")
    if not _nonnegative_int(self.exposure_count):
      raise ValueError("memory_retention_exposure_count_invalid")
    if not _nonnegative_int(self.attributed_use_count):
      raise ValueError("memory_retention_attributed_use_count_invalid")
    if self.attributed_use_count > self.exposure_count:
      raise ValueError("memory_retention_attributed_use_count_invalid")


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
  min_age_days: int = 30
  min_exposures: int = 3
  max_attributed_use_ratio: float = 0.05

  def __post_init__(self) -> None:
    if not _positive_int(self.min_age_days):
      raise ValueError("memory_retention_min_age_days_invalid")
    if not _positive_int(self.min_exposures):
      raise ValueError("memory_retention_min_exposures_invalid")
    if not _unit_interval(self.max_attributed_use_ratio):
      raise ValueError("memory_retention_max_attributed_use_ratio_invalid")


@dataclass(frozen=True, slots=True)
class Retain:
  pass


@dataclass(frozen=True, slots=True)
class Omit:
  reason: Literal["expired", "unattributed", "low_attributed_use"]


@dataclass(frozen=True, slots=True)
class SelectionPath:
  lane: str
  rank: int | None
  signal: str
  relation: str | None = None

  def __post_init__(self) -> None:
    _required_text(self.lane, error="memory_selection_path_lane_invalid")
    _bounded_text(self.signal, error="memory_selection_path_signal_invalid")
    if self.rank is not None and not _bounded_rank(self.rank):
      raise ValueError("memory_selection_path_rank_invalid")
    if self.relation is not None:
      _bounded_text(self.relation, error="memory_selection_path_relation_invalid")


@dataclass(frozen=True, slots=True)
class SelectionItem:
  source: str
  source_id: str
  partition: str
  kind: str
  text: str
  content: str
  confidence: float = 1.0
  updated_at: datetime | None = None
  available_at: datetime | None = None
  revision: str = ""
  source_digest: str | None = None
  omission: Literal["retention"] | None = None
  paths: tuple[SelectionPath, ...] = ()

  def __post_init__(self) -> None:
    _required_text(self.source, error="memory_selection_source_invalid")
    if not isinstance(self.source_id, str):
      raise ValueError("memory_selection_source_id_invalid")
    _required_text(self.partition, error="memory_selection_partition_invalid")
    _required_text(self.kind, error="memory_selection_kind_invalid")
    if not isinstance(self.text, str):
      raise ValueError("memory_selection_text_invalid")
    if not isinstance(self.content, str):
      raise ValueError("memory_selection_content_invalid")
    if not _unit_interval(self.confidence):
      raise ValueError("memory_selection_confidence_invalid")
    _selection_aware(self.updated_at, field="updated_at")
    _selection_aware(self.available_at, field="available_at")
    if not isinstance(self.revision, str):
      raise ValueError("memory_selection_revision_invalid")
    if self.source_digest is not None and not isinstance(self.source_digest, str):
      raise ValueError("memory_selection_source_digest_invalid")
    if self.omission not in {None, "retention"}:
      raise ValueError("memory_selection_omission_invalid")
    if not isinstance(self.paths, tuple) or any(not isinstance(path, SelectionPath) for path in self.paths):
      raise ValueError("memory_selection_paths_invalid")


@dataclass(frozen=True, slots=True)
class SelectionRoute:
  source: str
  source_id: str
  lane: str
  rank: int
  signal: str
  relation: str | None = None

  def __post_init__(self) -> None:
    _required_text(self.source, error="memory_selection_route_source_invalid")
    _required_text(self.source_id, error="memory_selection_route_source_id_invalid")
    _required_text(self.lane, error="memory_selection_route_lane_invalid")
    if not _bounded_rank(self.rank):
      raise ValueError("memory_selection_route_rank_invalid")
    _bounded_text(self.signal, error="memory_selection_route_signal_invalid")
    if self.relation is not None:
      _bounded_text(self.relation, error="memory_selection_route_relation_invalid")


@dataclass(frozen=True, slots=True)
class SelectionPriority:
  label: str
  score: int

  def __post_init__(self) -> None:
    _required_text(self.label, error="memory_selection_priority_label_invalid")
    if not isinstance(self.score, int) or isinstance(self.score, bool):
      raise ValueError("memory_selection_priority_score_invalid")


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
  partitions: tuple[SelectionPriority, ...] = ()
  kinds: tuple[SelectionPriority, ...] = ()
  source_order: tuple[str, ...] = ()
  route_order: tuple[str, ...] = ()
  omission_order: tuple[Literal["retention", "cutoff"], ...] = ("retention", "cutoff")
  lexical_lane: str = "lexical"
  lexical_weight: int = 100
  confidence_weight: int = 20
  route_rank_ceiling: int = 100
  max_query_chars: int = 4_000
  max_paths: int = 20
  truncation_marker: str = " [truncated]"

  def __post_init__(self) -> None:
    if not isinstance(self.partitions, tuple) or any(
      not isinstance(item, SelectionPriority) for item in self.partitions
    ):
      raise ValueError("memory_selection_partition_priority_invalid")
    if not isinstance(self.kinds, tuple) or any(
      not isinstance(item, SelectionPriority) for item in self.kinds
    ):
      raise ValueError("memory_selection_kind_priority_invalid")
    for values, error in (
      (self.source_order, "memory_selection_source_order_invalid"),
      (self.route_order, "memory_selection_route_order_invalid"),
      (self.omission_order, "memory_selection_omission_order_invalid"),
    ):
      if not isinstance(values, tuple):
        raise ValueError(error)
    _unique_labels((item.label for item in self.partitions), error="memory_selection_partition_priority_duplicated")
    _unique_labels((item.label for item in self.kinds), error="memory_selection_kind_priority_duplicated")
    _unique_labels(self.source_order, error="memory_selection_source_order_duplicated")
    _unique_labels(self.route_order, error="memory_selection_route_order_duplicated")
    _unique_labels(self.omission_order, error="memory_selection_omission_order_duplicated")
    if set(self.omission_order) != {"retention", "cutoff"}:
      raise ValueError("memory_selection_omission_order_invalid")
    _required_text(self.lexical_lane, error="memory_selection_lexical_lane_invalid")
    for value, error in (
      (self.lexical_weight, "memory_selection_lexical_weight_invalid"),
      (self.confidence_weight, "memory_selection_confidence_weight_invalid"),
    ):
      if not _nonnegative_int(value):
        raise ValueError(error)
    for value, error in (
      (self.route_rank_ceiling, "memory_selection_route_rank_ceiling_invalid"),
      (self.max_query_chars, "memory_selection_query_limit_invalid"),
      (self.max_paths, "memory_selection_path_limit_invalid"),
    ):
      if not _positive_int(value):
        raise ValueError(error)
    _required_text(self.truncation_marker, error="memory_selection_truncation_marker_invalid")


@dataclass(frozen=True, slots=True)
class SelectionLimits:
  max_items: int
  max_chars: int

  def __post_init__(self) -> None:
    if not _positive_int(self.max_items):
      raise ValueError("memory_selection_item_limit_invalid")
    if not _positive_int(self.max_chars):
      raise ValueError("memory_selection_char_limit_invalid")


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
  source: str
  source_id: str
  partition: str
  kind: str
  candidate_rank: int
  score: int
  revision: str
  text_hash: str
  source_digest: str | None = None
  paths: tuple[SelectionPath, ...] = ()


@dataclass(frozen=True, slots=True)
class Selected:
  candidate: SelectionCandidate
  rank: int
  content_chars: int


@dataclass(frozen=True, slots=True)
class Omitted:
  candidate: SelectionCandidate
  reason: Literal["item_budget", "char_budget", "retention", "cutoff", "invalid"]


@dataclass(frozen=True, slots=True)
class Selection:
  content: str
  outcomes: tuple[Selected | Omitted, ...]
  truncated: bool

  @property
  def selected(self) -> tuple[Selected, ...]:
    return tuple(item for item in self.outcomes if isinstance(item, Selected))

  @property
  def source_count(self) -> int:
    return len(self.outcomes)

  @property
  def included_count(self) -> int:
    return len(self.selected)


@dataclass(frozen=True, slots=True)
class SourceDependency:
  owner: str
  revision: str
  fragment: str
  kind: str
  digest: str

  def __post_init__(self) -> None:
    for value, field in (
      (self.owner, "owner"),
      (self.revision, "revision"),
      (self.fragment, "fragment"),
      (self.kind, "kind"),
    ):
      _required_text(value, error=f"memory_support_{field}_invalid")
    _source_digest(self.digest)


@dataclass(frozen=True, slots=True)
class ReopenedSource:
  owner: str
  revision: str
  fragment: str
  kind: str
  digest: str | None
  current_revision: str | None

  def __post_init__(self) -> None:
    for value, field in (
      (self.owner, "owner"),
      (self.revision, "revision"),
      (self.fragment, "fragment"),
      (self.kind, "kind"),
    ):
      _required_text(value, error=f"memory_support_{field}_invalid")
    if self.digest is not None:
      _source_digest(self.digest)
    if self.current_revision is not None:
      _required_text(self.current_revision, error="memory_support_current_revision_invalid")


@dataclass(frozen=True, slots=True)
class Current:
  pass


@dataclass(frozen=True, slots=True)
class Replaced:
  pass


@dataclass(frozen=True, slots=True)
class Stale:
  pass


@dataclass(frozen=True, slots=True)
class Missing:
  pass


AcceptedDecision: TypeAlias = Accept | Replace
AdmissionDecision: TypeAlias = Accept | Reject | Replace
Admission: TypeAlias = tuple[AdmissionDecision, ...]
RetentionDecision: TypeAlias = Retain | Omit
SelectionOutcome: TypeAlias = Selected | Omitted
SupportDecision: TypeAlias = Current | Replaced | Stale | Missing


def _unit_interval(value: object) -> bool:
  return (
    isinstance(value, int | float)
    and not isinstance(value, bool)
    and 0 <= value <= 1
    and math.isfinite(value)
  )


def _positive_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonnegative_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_aware(value: object, *, field: str) -> None:
  if value is None:
    return
  if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
    raise ValueError(f"memory_retention_{field}_timezone_required")


def _required_text(value: object, *, error: str) -> None:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(error)


def _text_tuple(value: object, *, error: str) -> None:
  if not isinstance(value, tuple) or any(not isinstance(item, str) or not item.strip() for item in value):
    raise ValueError(error)


def _text_frozenset(value: object, *, error: str) -> None:
  if not isinstance(value, frozenset) or any(not isinstance(item, str) or not item.strip() for item in value):
    raise ValueError(error)


def _bounded_text(value: object, *, error: str) -> None:
  if not isinstance(value, str) or not value.strip() or len(value) > 512:
    raise ValueError(error)


def _bounded_rank(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 100


def _unique_labels(values: Iterable[str], *, error: str) -> None:
  labels = tuple(values)
  if any(not isinstance(value, str) or not value.strip() for value in labels):
    raise ValueError(error.replace("duplicated", "invalid"))
  if len(labels) != len(set(labels)):
    raise ValueError(error)


def _selection_aware(value: object, *, field: str) -> None:
  if value is None:
    return
  if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
    raise ValueError(f"memory_selection_{field}_timezone_required")


def _source_digest(value: object) -> None:
  if (
    not isinstance(value, str)
    or len(value) != 71
    or not value.startswith("sha256:")
    or any(character not in "0123456789abcdef" for character in value[7:])
  ):
    raise ValueError("memory_support_digest_invalid")
