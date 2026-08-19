from datetime import datetime, timedelta, timezone

import pytest

from agos_memory.select import SelectionIdentityError, normalize_query, select
from agos_memory.types import (
  Omitted,
  Selected,
  Selection,
  SelectionCandidate,
  SelectionItem,
  SelectionLimits,
  SelectionPath,
  SelectionPolicy,
  SelectionPriority,
  SelectionRoute,
)


NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)
POLICY = SelectionPolicy(
  partitions=(
    SelectionPriority("narrow", 30),
    SelectionPriority("shared", 20),
    SelectionPriority("broad", 0),
  ),
  kinds=(
    SelectionPriority("instruction", 50),
    SelectionPriority("decision", 40),
    SelectionPriority("fact", 10),
  ),
  source_order=("record", "guidance", "observation"),
  route_order=("keyword", "entity", "semantic"),
)


def test_select_compiles_one_exact_immutable_plan() -> None:
  content = "- [record:debt] shared/decision: Committee DSCR floor is 1.25x."
  result = select(
    (
      _item(
        "debt",
        kind="decision",
        text="Committee DSCR floor is 1.25x.",
        content=content,
        confidence=0.9,
        updated_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        revision="revision-debt",
      ),
      _item(
        "policy",
        source="guidance",
        partition="broad",
        kind="instruction",
        text="Always show insurance sensitivity for committee.",
        confidence=1,
        updated_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
        revision="revision-policy",
      ),
      _item(
        "old",
        text="Old committee memory.",
        omission="retention",
        revision="revision-old",
      ),
    ),
    query="committee DSCR insurance",
    limits=SelectionLimits(max_items=1, max_chars=1_000),
    policy=POLICY,
    now=NOW,
  )

  assert result == Selection(
    content=content,
    outcomes=(
      Selected(
        candidate=SelectionCandidate(
          source="record",
          source_id="debt",
          partition="shared",
          kind="decision",
          candidate_rank=1,
          score=278,
          revision="revision-debt",
          text_hash="sha256:3251591c68abdf7eea44656953344e65406c9699aca14e79236eb466960e7a6c",
        ),
        rank=1,
        content_chars=len(content),
      ),
      Omitted(
        candidate=SelectionCandidate(
          source="guidance",
          source_id="policy",
          partition="broad",
          kind="instruction",
          candidate_rank=2,
          score=270,
          revision="revision-policy",
          text_hash="sha256:3a66753a29cb429d92989c4188e629a949cea09c8155259f41ac1fb0463bf205",
        ),
        reason="item_budget",
      ),
      Omitted(
        candidate=SelectionCandidate(
          source="record",
          source_id="old",
          partition="shared",
          kind="fact",
          candidate_rank=3,
          score=150,
          revision="revision-old",
          text_hash="sha256:9ee9544ec4feee4a550183cfb19cc57dd71f25bce0c0fefc50c1c27fcdff48b3",
        ),
        reason="retention",
      ),
    ),
    truncated=True,
  )


def test_select_is_permutation_invariant_and_keeps_source_qualified_identity() -> None:
  items = (
    _item("shared", source="record", kind="decision", text="Shared decision."),
    _item("shared", source="guidance", kind="instruction", text="Shared instruction."),
    _item("later", kind="fact", text="Shared fact."),
  )
  kwargs = {
    "query": "shared",
    "limits": SelectionLimits(max_items=3, max_chars=1_000),
    "policy": POLICY,
    "now": NOW,
  }

  first = select(items, **kwargs)
  second = select(tuple(reversed(items)), **kwargs)

  assert second == first
  assert [(item.candidate.source, item.candidate.source_id) for item in first.selected] == [
    ("guidance", "shared"),
    ("record", "shared"),
    ("record", "later"),
  ]


def test_select_routes_only_provided_items_and_preserves_bounded_evidence() -> None:
  temporal = SelectionPath(
    lane="temporal",
    rank=None,
    signal="reviewed_at=2026-05-19T00:00:00+00:00",
    relation="supported_by:source-1",
  )
  item = _item("routed", text="The lender requested a floor.", paths=(temporal,))
  routes = tuple(
    SelectionRoute(
      source="record",
      source_id="routed",
      lane="entity",
      rank=rank,
      signal=f"entity-{rank}",
    )
    for rank in range(1, 4)
  )
  routes = (*routes, SelectionRoute("record", "missing", "entity", 1, "missing"))
  policy = SelectionPolicy(
    partitions=POLICY.partitions,
    kinds=POLICY.kinds,
    source_order=POLICY.source_order,
    route_order=POLICY.route_order,
    max_paths=2,
  )

  result = select(
    (item,),
    routes=tuple(reversed(routes)),
    query="unmatched",
    limits=SelectionLimits(max_items=1, max_chars=1_000),
    policy=policy,
    now=NOW,
    include_paths=True,
  )

  assert result.selected[0].candidate.source_id == "routed"
  assert result.outcomes[0].candidate.paths == (
    SelectionPath("entity", 1, "entity-1"),
    temporal,
  )


def test_select_keeps_current_paths_before_bounded_historical_evidence() -> None:
  historical = (
    SelectionPath("history", 1, "old-1"),
    SelectionPath("history", 2, "old-2"),
  )
  result = select(
    (_item("routed", text="The lender requested a floor.", paths=historical),),
    routes=(SelectionRoute("record", "routed", "entity", 1, "current"),),
    query="unmatched",
    limits=SelectionLimits(max_items=1, max_chars=1_000),
    policy=SelectionPolicy(max_paths=2),
    now=NOW,
    include_paths=True,
  )

  assert result.selected[0].candidate.paths == (
    SelectionPath("entity", 1, "current"),
    SelectionPath("history", 2, "old-2"),
  )


def test_select_accounts_for_cutoff_invalid_and_budgets() -> None:
  result = select(
    (
      _item("first", text="Current budget fact.", content="A" * 80),
      _item("second", text="Current budget detail."),
      _item("invalid", text="Current budget invalid.", content=""),
      _item("future", text="Current budget later.", available_at=NOW + timedelta(seconds=1)),
      _item("retained-out", text="Current budget old.", omission="retention"),
    ),
    query="current budget",
    limits=SelectionLimits(max_items=2, max_chars=75),
    policy=POLICY,
    now=NOW,
  )

  assert result.content.endswith(" [truncated]")
  assert ["selected" if isinstance(outcome, Selected) else outcome.reason for outcome in result.outcomes] == [
    "selected",
    "invalid",
    "char_budget",
    "retention",
    "cutoff",
  ]
  assert result.included_count == 1
  assert result.source_count == 5


def test_select_omits_unmatched_inputs_without_inventing_candidates() -> None:
  result = select(
    (_item("rent", text="Rent concessions increased."),),
    query="elevator certificate",
    limits=SelectionLimits(max_items=1, max_chars=100),
    policy=POLICY,
    now=NOW,
  )

  assert result == Selection(content="", outcomes=(), truncated=False)


def test_select_rejects_duplicate_identity_within_one_source() -> None:
  with pytest.raises(SelectionIdentityError) as exc_info:
    select(
      (_item(" duplicate "), _item("duplicate")),
      query="record",
      limits=SelectionLimits(max_items=1, max_chars=100),
      policy=POLICY,
      now=NOW,
    )

  assert exc_info.value.sources == ("record",)


def test_select_rejects_invalid_boundary_values() -> None:
  with pytest.raises(ValueError, match="memory_selection_confidence_invalid"):
    _item("record", confidence=float("nan"))
  with pytest.raises(ValueError, match="memory_selection_updated_at_timezone_required"):
    _item("record", updated_at=datetime(2026, 5, 20))
  with pytest.raises(ValueError, match="memory_selection_now_timezone_required"):
    select(
      (_item("record"),),
      query="record",
      limits=SelectionLimits(max_items=1, max_chars=100),
      policy=POLICY,
      now=datetime(2026, 5, 20),
    )
  with pytest.raises(ValueError, match="memory_selection_item_limit_invalid"):
    SelectionLimits(max_items=0, max_chars=100)


def test_normalize_query_keeps_bounded_head_and_tail() -> None:
  query = normalize_query("head " + "x" * 5_000 + " tail", max_chars=4_000)

  assert len(query) == 4_000
  assert query.startswith("head ")
  assert query.endswith(" tail")
  assert normalize_query("head tail", max_chars=1) == "h"
  assert normalize_query("head tail", max_chars=2) == "hl"


def test_normalize_query_rejects_invalid_query_shape() -> None:
  with pytest.raises(ValueError, match="memory_selection_query_invalid"):
    normalize_query(None, max_chars=4_000)  # type: ignore[arg-type]


def _item(
  source_id: str,
  *,
  source: str = "record",
  partition: str = "shared",
  kind: str = "fact",
  text: str = "Current record.",
  content: str | None = None,
  confidence: float = 1,
  updated_at: datetime | None = None,
  available_at: datetime | None = None,
  revision: str = "",
  omission: str | None = None,
  paths: tuple[SelectionPath, ...] = (),
) -> SelectionItem:
  return SelectionItem(
    source=source,
    source_id=source_id,
    partition=partition,
    kind=kind,
    text=text,
    content=content if content is not None else f"- [{source}:{source_id}] {text}",
    confidence=confidence,
    updated_at=updated_at,
    available_at=available_at,
    revision=revision,
    omission=omission,  # type: ignore[arg-type]
    paths=paths,
  )
