from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal

from agos_memory.types import (
  Accept,
  Admission,
  AdmissionDecision,
  AdmissionRules,
  ExistingRecord,
  PartitionRule,
  Proposal,
  Reject,
  Replace,
)


UnsafeTextReason = Literal["unsafe_secret", "unsafe_reasoning"]

_SECRET_RE = re.compile(
  r"(?ix)\b(?:"
  r"(?:api[_ -]?key|password|secret|token)\s*[:=]\s*\S+"
  r"|bearer\s+[a-z0-9._~+/=-]{8,}"
  r"|sk-[a-z0-9_-]{6,}"
  r"|gh[opsu]_[a-z0-9]{6,}"
  r")"
)
_REASONING_RE = re.compile(r"(?i)(chain[- ]of[- ]thought|hidden reasoning|raw reasoning)")
_DOCISH_RE = re.compile(r"(?i)(full note|complete transcript|entire document|verbatim)")


def admit(
  proposals: Sequence[Proposal],
  existing: Sequence[ExistingRecord],
  *,
  rules: AdmissionRules,
  now: datetime,
) -> Admission:
  """Compile proposals in caller-supplied priority order.

  Earlier accepted proposals reserve their content identity and replacement
  targets before later proposals are considered.
  """
  if not isinstance(proposals, Sequence) or isinstance(proposals, str | bytes) or any(
    not isinstance(proposal, Proposal) for proposal in proposals
  ):
    raise ValueError("memory_admission_proposals_invalid")
  if not isinstance(existing, Sequence) or isinstance(existing, str | bytes) or any(
    not isinstance(record, ExistingRecord) for record in existing
  ):
    raise ValueError("memory_admission_existing_invalid")
  if not isinstance(rules, AdmissionRules):
    raise ValueError("memory_admission_rules_invalid")
  now = _utc(now)
  partition_rules = _partition_rules(rules.partition_rules)
  by_id = _existing_by_id(existing)
  active_keys = {_record_key(record) for record in existing}
  planned_keys: set[tuple[object, ...]] = set()
  planned_replacements: set[str] = set()
  decisions: list[AdmissionDecision] = []

  for raw in proposals:
    proposal = _normalize(raw)
    reason = _reject_reason(proposal, rules=rules, partition_rules=partition_rules)
    if reason is not None:
      decisions.append(_reject(proposal, reason))
      continue

    key = _proposal_key(proposal)
    replacements = _replacement_ids(proposal, by_id)
    if proposal.supersedes and len(replacements) != len(set(proposal.supersedes)):
      decisions.append(_reject(proposal, "supersedes_out_of_scope"))
      continue
    if planned_replacements.intersection(replacements):
      decisions.append(_reject(proposal, "supersedes_ambiguous"))
      continue
    if key in active_keys or key in planned_keys:
      decisions.append(_reject(proposal, "duplicate"))
      continue

    try:
      expires_at = now + timedelta(days=proposal.expires_days) if proposal.expires_days else None
    except OverflowError:
      decisions.append(_reject(proposal, "expiry_out_of_range"))
      continue
    accepted = _accepted(
      proposal,
      replacements=replacements,
      expires_at=expires_at,
    )
    planned_keys.add(key)
    planned_replacements.update(replacements)
    decisions.append(accepted)

  return tuple(decisions)


def _existing_by_id(existing: Sequence[ExistingRecord]) -> Mapping[str, ExistingRecord]:
  by_id: dict[str, ExistingRecord] = {}
  for record in existing:
    if record.record_id in by_id:
      raise ValueError("memory_admission_existing_record_id_duplicated")
    by_id[record.record_id] = record
  return by_id


def unsafe_text_reason(value: str) -> UnsafeTextReason | None:
  if not isinstance(value, str):
    raise ValueError("memory_admission_text_invalid")
  if _SECRET_RE.search(value):
    return "unsafe_secret"
  if _REASONING_RE.search(value):
    return "unsafe_reasoning"
  return None


def _partition_rules(rules: Sequence[PartitionRule]) -> Mapping[str, PartitionRule]:
  by_label = {rule.label: rule for rule in rules}
  if len(by_label) != len(rules):
    raise ValueError("memory_admission_partition_rule_duplicated")
  return by_label


def _reject_reason(
  proposal: Proposal,
  *,
  rules: AdmissionRules,
  partition_rules: Mapping[str, PartitionRule],
) -> str | None:
  if not proposal.text:
    return "empty_text"
  if len(proposal.text) > rules.max_text_chars:
    return "text_too_long"
  if proposal.confidence < rules.min_confidence:
    return "low_confidence"
  partition_rule = partition_rules.get(proposal.partition.label)
  if partition_rule is not None:
    if partition_rule.missing_key_reason is not None and proposal.partition.key is None:
      return partition_rule.missing_key_reason
    if partition_rule.allowed_kinds is not None and proposal.kind not in partition_rule.allowed_kinds:
      return partition_rule.disallowed_kind_reason
  if unsafe_reason := unsafe_text_reason(proposal.text):
    return unsafe_reason
  if _DOCISH_RE.search(proposal.text):
    return "belongs_in_docs"
  return None


def _replacement_ids(
  proposal: Proposal,
  by_id: Mapping[str, ExistingRecord],
) -> tuple[str, ...]:
  replacements = []
  for record_id in sorted(set(proposal.supersedes)):
    record = by_id.get(record_id)
    if record is None or not record.replaceable:
      continue
    if record.partition != proposal.partition or record.kind != proposal.kind:
      continue
    replacements.append(record_id)
  return tuple(replacements)


def _record_key(record: ExistingRecord) -> tuple[object, ...]:
  return record.partition, record.kind, _text_key(record.text)


def _proposal_key(proposal: Proposal) -> tuple[object, ...]:
  return proposal.partition, proposal.kind, _text_key(proposal.text)


def _text_key(value: str) -> str:
  return _normalize_text(value).casefold()


def _normalize(proposal: Proposal) -> Proposal:
  return Proposal(
    partition=proposal.partition,
    kind=proposal.kind,
    text=_normalize_text(proposal.text),
    confidence=proposal.confidence,
    source_refs=tuple(proposal.source_refs),
    supersedes=tuple(proposal.supersedes),
    expires_days=proposal.expires_days,
  )


def _accepted(
  proposal: Proposal,
  *,
  replacements: tuple[str, ...],
  expires_at: datetime | None,
) -> Accept | Replace:
  if replacements:
    return Replace(
      proposal=proposal,
      replaced_record_ids=replacements,
      expires_at=expires_at,
    )
  return Accept(proposal=proposal, expires_at=expires_at)


def _reject(proposal: Proposal, reason: str) -> Reject:
  return Reject(proposal=proposal, reason=reason)


def _normalize_text(value: str) -> str:
  return " ".join(value.split())


def _utc(value: datetime) -> datetime:
  if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
    raise ValueError("memory_admission_now_timezone_required")
  return value.astimezone(timezone.utc)
