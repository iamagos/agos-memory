from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from agos_memory.admit import admit, unsafe_text_reason
from agos_memory.types import (
  Accept,
  AdmissionRules,
  ExistingRecord,
  Partition,
  PartitionRule,
  Proposal,
  Reject,
  Replace,
)


NOW = datetime(2026, 5, 16, tzinfo=timezone.utc)
SHARED = Partition("shared")
PERSONAL = Partition("personal", "person-1")
RULES = AdmissionRules(
  partition_rules=(
    PartitionRule(
      label="personal",
      missing_key_reason="missing_person",
      allowed_kinds=frozenset(("preference", "instruction")),
      disallowed_kind_reason="kind_not_personal",
    ),
    PartitionRule(label="session", missing_key_reason="missing_session"),
  ),
)


def _proposal(
  *,
  partition: Partition = SHARED,
  kind: str = "fact",
  text: str = "Revenue is stable.",
  confidence: float = 0.8,
  source_refs: tuple[str, ...] = (),
  supersedes: tuple[str, ...] = (),
  expires_days: int | None = None,
) -> Proposal:
  return Proposal(
    partition=partition,
    kind=kind,
    text=text,
    confidence=confidence,
    source_refs=source_refs,
    supersedes=supersedes,
    expires_days=expires_days,
  )


def _record(
  *,
  record_id: str = "record-existing",
  partition: Partition = SHARED,
  kind: str = "fact",
  text: str = "Revenue is stable.",
  replaceable: bool = True,
) -> ExistingRecord:
  return ExistingRecord(
    record_id=record_id,
    partition=partition,
    kind=kind,
    text=text,
    replaceable=replaceable,
  )


def test_admit_returns_one_ordered_immutable_outcome_per_proposal() -> None:
  duplicate = _proposal(text="  RENT   ROLL IS NORMALIZED. ")
  accepted = _proposal(
    kind="decision",
    text=" Use   DSCR 1.25. ",
    confidence=0.9,
    source_refs=("source-1",),
    expires_days=2,
  )
  replacement = _proposal(
    partition=PERSONAL,
    kind="preference",
    text="Show the downside case first.",
    supersedes=("record-old",),
  )

  outcomes = admit(
    (duplicate, accepted, replacement),
    (
      _record(record_id="record-duplicate", text="rent roll is normalized."),
      _record(
        record_id="record-old",
        partition=PERSONAL,
        kind="preference",
        text="Show the base case first.",
      ),
    ),
    rules=RULES,
    now=NOW,
  )

  assert outcomes == (
    Reject(
      proposal=_proposal(text="RENT ROLL IS NORMALIZED."),
      reason="duplicate",
    ),
    Accept(
      proposal=_proposal(
        kind="decision",
        text="Use DSCR 1.25.",
        confidence=0.9,
        source_refs=("source-1",),
        expires_days=2,
      ),
      expires_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    ),
    Replace(
      proposal=replacement,
      replaced_record_ids=("record-old",),
    ),
  )
  with pytest.raises(FrozenInstanceError):
    outcomes[1].proposal.text = "changed"  # type: ignore[misc]


def test_admit_rejects_existing_and_planned_duplicates() -> None:
  outcomes = admit(
    (
      _proposal(text="RENT ROLL IS NORMALIZED."),
      _proposal(text="Debt matures in 2029."),
      _proposal(text=" debt   matures in 2029. "),
    ),
    (_record(text=" rent   roll is normalized. "),),
    rules=RULES,
    now=NOW,
  )

  assert [type(outcome) for outcome in outcomes] == [Reject, Accept, Reject]
  assert [outcome.reason for outcome in outcomes if isinstance(outcome, Reject)] == [
    "duplicate",
    "duplicate",
  ]


def test_admit_replaces_once_then_rejects_ambiguous_fan_out() -> None:
  outcomes = admit(
    (
      _proposal(text="Debt matures in 2028.", supersedes=("record-old",)),
      _proposal(text="Debt matures in 2029.", supersedes=("record-old",)),
    ),
    (_record(record_id="record-old", text="Debt matures in 2027."),),
    rules=RULES,
    now=NOW,
  )

  assert isinstance(outcomes[0], Replace)
  assert outcomes[0].replaced_record_ids == ("record-old",)
  assert outcomes[1] == Reject(
    proposal=_proposal(text="Debt matures in 2029.", supersedes=("record-old",)),
    reason="supersedes_ambiguous",
  )


@pytest.mark.parametrize(
  "existing",
  (
    (),
    (_record(record_id="record-old", replaceable=False),),
    (_record(record_id="record-old", partition=Partition("other")),),
    (_record(record_id="record-old", kind="decision"),),
  ),
)
def test_admit_rejects_unavailable_replacement(
  existing: tuple[ExistingRecord, ...],
) -> None:
  proposal = _proposal(supersedes=("record-old",))

  assert admit((proposal,), existing, rules=RULES, now=NOW) == (
    Reject(proposal=proposal, reason="supersedes_out_of_scope"),
  )


@pytest.mark.parametrize(
  ("proposal", "reason"),
  (
    (_proposal(text=""), "empty_text"),
    (_proposal(text="x" * 1_001), "text_too_long"),
    (_proposal(text="Too uncertain.", confidence=0.2), "low_confidence"),
    (
      _proposal(partition=Partition("personal"), kind="preference", text="Prefer concise answers."),
      "missing_person",
    ),
    (
      _proposal(partition=PERSONAL, kind="fact", text="Revenue is stable."),
      "kind_not_personal",
    ),
    (
      _proposal(partition=Partition("session"), kind="handoff", text="Check the quote."),
      "missing_session",
    ),
    (_proposal(text="API key is sk-secret"), "unsafe_secret"),
    (_proposal(text="Hidden reasoning says approve it."), "unsafe_reasoning"),
    (_proposal(text="Copy the complete transcript."), "belongs_in_docs"),
  ),
)
def test_admit_rejects_invalid_proposal(proposal: Proposal, reason: str) -> None:
  assert admit((proposal,), (), rules=RULES, now=NOW) == (
    Reject(proposal=proposal, reason=reason),
  )


def test_admit_rejects_unrepresentable_expiry_without_aborting_later_proposals() -> None:
  out_of_range = _proposal(text="Expires too late.", expires_days=10**12)
  accepted = _proposal(text="Still evaluated.")

  assert admit((out_of_range, accepted), (), rules=RULES, now=NOW) == (
    Reject(proposal=out_of_range, reason="expiry_out_of_range"),
    Accept(proposal=accepted),
  )


def test_admit_rejects_duplicate_partition_rules() -> None:
  rules = AdmissionRules(
    partition_rules=(PartitionRule("shared"), PartitionRule("shared")),
  )

  with pytest.raises(ValueError, match="memory_admission_partition_rule_duplicated"):
    admit((), (), rules=rules, now=NOW)


@pytest.mark.parametrize(
  ("partition", "error"),
  (
    (lambda: Partition(""), "memory_admission_partition_label_invalid"),
    (lambda: Partition("shared", ""), "memory_admission_partition_key_invalid"),
  ),
)
def test_partition_rejects_invalid_shape(partition: object, error: str) -> None:
  with pytest.raises(ValueError, match=error):
    partition()  # type: ignore[operator]


@pytest.mark.parametrize(
  ("values", "error"),
  (
    ({"partition": "shared"}, "memory_admission_proposal_partition_invalid"),
    ({"kind": []}, "memory_admission_proposal_kind_invalid"),
    ({"text": None}, "memory_admission_proposal_text_invalid"),
    ({"source_refs": ["source-1"]}, "memory_admission_source_refs_invalid"),
    ({"source_refs": ("",)}, "memory_admission_source_refs_invalid"),
    ({"supersedes": ("record-1", 2)}, "memory_admission_supersedes_invalid"),
  ),
)
def test_proposal_rejects_invalid_shape(values: dict[str, object], error: str) -> None:
  with pytest.raises(ValueError, match=error):
    _proposal(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
  ("values", "error"),
  (
    ({"record_id": ""}, "memory_admission_existing_record_id_invalid"),
    ({"partition": "shared"}, "memory_admission_existing_partition_invalid"),
    ({"kind": None}, "memory_admission_existing_kind_invalid"),
    ({"text": []}, "memory_admission_existing_text_invalid"),
    ({"replaceable": 1}, "memory_admission_existing_replaceable_invalid"),
  ),
)
def test_existing_record_rejects_invalid_shape(values: dict[str, object], error: str) -> None:
  with pytest.raises(ValueError, match=error):
    _record(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
  ("values", "error"),
  (
    ({"label": ""}, "memory_admission_partition_rule_label_invalid"),
    (
      {"label": "shared", "missing_key_reason": ""},
      "memory_admission_partition_rule_missing_key_reason_invalid",
    ),
    (
      {"label": "shared", "allowed_kinds": ("fact",)},
      "memory_admission_partition_rule_allowed_kinds_invalid",
    ),
    (
      {"label": "shared", "allowed_kinds": frozenset(("",))},
      "memory_admission_partition_rule_allowed_kinds_invalid",
    ),
    (
      {"label": "shared", "disallowed_kind_reason": ""},
      "memory_admission_partition_rule_disallowed_kind_reason_invalid",
    ),
  ),
)
def test_partition_rule_rejects_invalid_shape(values: dict[str, object], error: str) -> None:
  with pytest.raises(ValueError, match=error):
    PartitionRule(**values)  # type: ignore[arg-type]


def test_rules_reject_invalid_partition_rule_shape() -> None:
  with pytest.raises(ValueError, match="memory_admission_partition_rules_invalid"):
    AdmissionRules(partition_rules=[PartitionRule("shared")])  # type: ignore[arg-type]


@pytest.mark.parametrize(
  ("proposals", "existing", "rules", "error"),
  (
    (None, (), RULES, "memory_admission_proposals_invalid"),
    ((object(),), (), RULES, "memory_admission_proposals_invalid"),
    ((), None, RULES, "memory_admission_existing_invalid"),
    ((), (object(),), RULES, "memory_admission_existing_invalid"),
    ((), (), None, "memory_admission_rules_invalid"),
  ),
)
def test_admit_rejects_invalid_boundary_shape(
  proposals: object,
  existing: object,
  rules: object,
  error: str,
) -> None:
  with pytest.raises(ValueError, match=error):
    admit(proposals, existing, rules=rules, now=NOW)  # type: ignore[arg-type]


def test_unsafe_text_reason_rejects_invalid_text() -> None:
  with pytest.raises(ValueError, match="memory_admission_text_invalid"):
    unsafe_text_reason(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", (float("nan"), float("inf"), -0.1, 1.1, True))
def test_proposal_rejects_invalid_confidence(confidence: object) -> None:
  with pytest.raises(ValueError, match="memory_admission_confidence_invalid"):
    _proposal(confidence=confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize("expires_days", (-1, 0, 1.5, True))
def test_proposal_rejects_invalid_expiry(expires_days: object) -> None:
  with pytest.raises(ValueError, match="memory_admission_expiry_invalid"):
    _proposal(expires_days=expires_days)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_confidence", (float("nan"), float("inf"), -0.1, 1.1, True))
def test_rules_reject_invalid_confidence(min_confidence: object) -> None:
  with pytest.raises(ValueError, match="memory_admission_min_confidence_invalid"):
    AdmissionRules(min_confidence=min_confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_text_chars", (0, -1, 1.5, True))
def test_rules_reject_invalid_text_bound(max_text_chars: object) -> None:
  with pytest.raises(ValueError, match="memory_admission_max_text_chars_invalid"):
    AdmissionRules(max_text_chars=max_text_chars)  # type: ignore[arg-type]


def test_admit_rejects_naive_time() -> None:
  with pytest.raises(ValueError, match="memory_admission_now_timezone_required"):
    admit((_proposal(),), (), rules=RULES, now=datetime(2026, 5, 16))


def test_admit_rejects_invalid_time_shape() -> None:
  with pytest.raises(ValueError, match="memory_admission_now_timezone_required"):
    admit((_proposal(),), (), rules=RULES, now=None)  # type: ignore[arg-type]


def test_admit_rejects_duplicate_existing_ids() -> None:
  existing = (
    _record(record_id="record-duplicate", replaceable=False),
    _record(record_id="record-duplicate", text="Other text."),
  )

  with pytest.raises(ValueError, match="memory_admission_existing_record_id_duplicated"):
    admit((_proposal(supersedes=("record-duplicate",)),), existing, rules=RULES, now=NOW)

  with pytest.raises(ValueError, match="memory_admission_existing_record_id_duplicated"):
    admit((_proposal(supersedes=("record-duplicate",)),), tuple(reversed(existing)), rules=RULES, now=NOW)
