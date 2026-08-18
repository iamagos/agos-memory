from __future__ import annotations

from datetime import datetime, timezone

from agos_memory.types import Omit, Retain, RetentionDecision, RetentionPolicy, RetentionState


def retain(
  state: RetentionState,
  *,
  policy: RetentionPolicy,
  now: datetime,
) -> RetentionDecision:
  """Decide whether one already-authorized record is eligible for context."""

  now = _utc(now)
  if state.expires_at is not None and state.expires_at <= now:
    return Omit("expired")
  if state.updated_at is None or (now - state.updated_at).days < policy.min_age_days:
    return Retain()
  if state.exposure_count < policy.min_exposures:
    return Retain()
  if state.attributed_use_count == 0:
    return Omit("unattributed")
  ratio_numerator, ratio_denominator = policy.max_attributed_use_ratio.as_integer_ratio()
  if state.attributed_use_count * ratio_denominator <= state.exposure_count * ratio_numerator:
    return Omit("low_attributed_use")
  return Retain()


def _utc(value: datetime) -> datetime:
  if value.tzinfo is None or value.utcoffset() is None:
    raise ValueError("memory_retention_cutoff_timezone_required")
  return value.astimezone(timezone.utc)
