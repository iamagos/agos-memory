from datetime import datetime, timedelta, timezone

import pytest

from agos_memory.retain import retain
from agos_memory.types import Omit, Retain, RetentionPolicy, RetentionState

NOW = datetime(2026, 5, 17, tzinfo=timezone.utc)
POLICY = RetentionPolicy()


def test_retain_omits_expired_context() -> None:
  state = RetentionState(expires_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

  assert retain(state, policy=POLICY, now=NOW) == Omit("expired")


def test_retain_freezes_inclusive_policy_boundaries() -> None:
  exactly_old = NOW - timedelta(days=POLICY.min_age_days)

  assert retain(
    RetentionState(expires_at=NOW),
    policy=POLICY,
    now=NOW,
  ) == Omit("expired")
  assert retain(
    RetentionState(expires_at=NOW + timedelta(microseconds=1)),
    policy=POLICY,
    now=NOW,
  ) == Retain()
  assert retain(
    RetentionState(updated_at=exactly_old, exposure_count=POLICY.min_exposures),
    policy=POLICY,
    now=NOW,
  ) == Omit("unattributed")
  assert retain(
    RetentionState(
      updated_at=exactly_old + timedelta(microseconds=1),
      exposure_count=POLICY.min_exposures,
    ),
    policy=POLICY,
    now=NOW,
  ) == Retain()
  assert retain(
    RetentionState(updated_at=exactly_old, exposure_count=POLICY.min_exposures - 1),
    policy=POLICY,
    now=NOW,
  ) == Retain()
  assert retain(
    RetentionState(
      updated_at=exactly_old,
      exposure_count=20,
      attributed_use_count=1,
    ),
    policy=POLICY,
    now=NOW,
  ) == Omit("low_attributed_use")
  assert retain(
    RetentionState(
      updated_at=exactly_old,
      exposure_count=19,
      attributed_use_count=1,
    ),
    policy=POLICY,
    now=NOW,
  ) == Retain()


def test_retain_omits_stale_unattributed_context() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    exposure_count=6,
  )

  assert retain(state, policy=POLICY, now=NOW) == Omit("unattributed")


def test_retain_honors_policy_age_threshold() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    exposure_count=6,
  )
  policy = RetentionPolicy(
    min_age_days=60,
    min_exposures=3,
    max_attributed_use_ratio=0.05,
  )

  assert retain(state, policy=policy, now=NOW) == Retain()


def test_retain_omits_stale_low_attributed_use_context() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    exposure_count=20,
    attributed_use_count=1,
  )

  assert retain(state, policy=POLICY, now=NOW) == Omit("low_attributed_use")


def test_retain_derives_attributed_use_ratio_from_counts() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    exposure_count=10,
    attributed_use_count=1,
  )

  assert retain(state, policy=POLICY, now=NOW) == Retain()


def test_retain_is_total_for_large_age_policy() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    exposure_count=10,
    attributed_use_count=1,
  )

  assert retain(state, policy=RetentionPolicy(min_age_days=10**12), now=NOW) == Retain()


def test_retain_is_total_for_large_signal_counts() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    exposure_count=10**1000,
    attributed_use_count=10**999,
  )

  assert retain(state, policy=POLICY, now=NOW) == Retain()


def test_retain_keeps_recent_context() -> None:
  state = RetentionState(
    updated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    exposure_count=20,
  )

  assert retain(state, policy=POLICY, now=NOW) == Retain()


def test_retain_rejects_naive_cutoff() -> None:
  with pytest.raises(ValueError, match="memory_retention_cutoff_timezone_required"):
    retain(RetentionState(), policy=POLICY, now=datetime(2026, 5, 17))


@pytest.mark.parametrize(
  ("field", "value"),
  (
    ("updated_at", datetime(2026, 5, 17)),
    ("expires_at", datetime(2026, 5, 17)),
  ),
)
def test_retention_state_requires_aware_time(field: str, value: datetime) -> None:
  with pytest.raises(ValueError, match=f"memory_retention_{field}_timezone_required"):
    RetentionState(**{field: value})


@pytest.mark.parametrize(
  "state",
  (
    {"exposure_count": -1},
    {"attributed_use_count": -1},
  ),
)
def test_retention_state_rejects_invalid_signals(state: dict[str, object]) -> None:
  with pytest.raises(ValueError, match="memory_retention_.*_invalid"):
    RetentionState(**state)  # type: ignore[arg-type]


def test_retention_state_rejects_more_uses_than_exposures() -> None:
  with pytest.raises(ValueError, match="memory_retention_attributed_use_count_invalid"):
    RetentionState(exposure_count=1, attributed_use_count=2)


@pytest.mark.parametrize(
  "policy",
  (
    {"min_age_days": 0},
    {"min_exposures": 0},
    {"max_attributed_use_ratio": float("nan")},
    {"max_attributed_use_ratio": 1.1},
  ),
)
def test_retention_policy_rejects_invalid_rules(policy: dict[str, object]) -> None:
  with pytest.raises(ValueError, match="memory_retention_.*_invalid"):
    RetentionPolicy(**policy)  # type: ignore[arg-type]
