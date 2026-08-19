from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

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
  SelectionOutcome,
)


_PendingReason = Literal[
  "pending",
  "selected",
  "item_budget",
  "char_budget",
  "retention",
  "cutoff",
  "invalid",
]
_TERM_RE = re.compile(r"[A-Za-z0-9_]{3,}")


class SelectionIdentityError(ValueError):
  def __init__(self, sources: Sequence[str]) -> None:
    self.sources = tuple(sources)
    super().__init__("memory_selection_source_identity_duplicated")


@dataclass(slots=True)
class _Candidate:
  item: SelectionItem
  score: int
  paths: tuple[SelectionPath, ...]
  candidate_rank: int = 0
  selected: bool = False
  rank: int | None = None
  content_chars: int = 0
  reason: _PendingReason = "pending"


def select(
  items: Sequence[SelectionItem],
  *,
  routes: Sequence[SelectionRoute] = (),
  query: str,
  limits: SelectionLimits,
  policy: SelectionPolicy,
  now: datetime,
  include_paths: bool = False,
) -> Selection:
  """Compile finite authorized inputs into one immutable selection plan."""

  now = _utc(now)
  _require_unique_identities(items, policy=policy)
  if not items:
    return Selection(content="", outcomes=(), truncated=False)

  route_map = _route_map(routes, policy=policy)
  query_terms = terms(normalize_query(query, max_chars=policy.max_query_chars))
  matched_terms = {
    _identity(item): query_terms & terms(item.text)
    for item in items
  }
  current = tuple(item for item in items if item.omission is None and not _future(item, now=now))
  candidates = [
    _candidate(
      item,
      score,
      routes=route_map.get(_identity(item), ()),
      matched_terms=matched_terms[_identity(item)],
      policy=policy,
      include_paths=include_paths,
    )
    for item, score in _ordered(current, routes=route_map, matched_terms=matched_terms, policy=policy)
    if matched_terms[_identity(item)] or route_map.get(_identity(item))
  ]
  omissions = {
    "retention": tuple(item for item in items if item.omission == "retention"),
    "cutoff": tuple(item for item in items if item.omission is None and _future(item, now=now)),
  }
  for reason in policy.omission_order:
    candidates.extend(
      _candidate(
        item,
        score,
        routes=route_map.get(_identity(item), ()),
        matched_terms=matched_terms[_identity(item)],
        policy=policy,
        include_paths=include_paths,
        reason=reason,
      )
      for item, score in _ordered(
        omissions[reason],
        routes=route_map,
        matched_terms=matched_terms,
        policy=policy,
      )
    )
  for candidate_rank, candidate in enumerate(candidates, start=1):
    candidate.candidate_rank = candidate_rank

  content: list[str] = []
  selected_count = 0
  used = 0
  blocked_reason: Literal["item_budget", "char_budget"] | None = None
  truncated = False

  for candidate in candidates:
    if candidate.reason in {"retention", "cutoff"}:
      continue
    item = candidate.item
    source_id = item.source_id.strip()
    if not source_id or not item.content.strip():
      candidate.reason = "invalid"
      continue
    if blocked_reason is not None:
      candidate.reason = blocked_reason
      continue
    if selected_count >= limits.max_items:
      candidate.reason = "item_budget"
      blocked_reason = "item_budget"
      truncated = True
      continue
    fitted = _append_content(
      content,
      item.content,
      used=used,
      max_chars=limits.max_chars,
      marker=policy.truncation_marker,
    )
    if fitted is None:
      candidate.reason = "char_budget"
      blocked_reason = "char_budget"
      truncated = True
      continue

    content.append(fitted)
    selected_count += 1
    candidate.selected = True
    candidate.rank = selected_count
    candidate.content_chars = len(fitted)
    candidate.reason = "selected"
    used += (1 if len(content) > 1 else 0) + len(fitted)
    if fitted != item.content:
      blocked_reason = "char_budget"
      truncated = True

  return Selection(
    content="\n".join(content),
    outcomes=tuple(_outcome(candidate) for candidate in candidates),
    truncated=truncated,
  )


def normalize_query(value: str, *, max_chars: int) -> str:
  if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
    raise ValueError("memory_selection_query_limit_invalid")
  if not isinstance(value, str):
    raise ValueError("memory_selection_query_invalid")
  query = " ".join(value.split())
  if len(query) <= max_chars:
    return query
  head = max_chars // 2
  if max_chars == 1:
    return query[:1]
  if max_chars == 2:
    return query[:head] + query[-(max_chars - head):]
  return f"{query[:head]} {query[-(max_chars - head - 1):]}"


def terms(value: str) -> set[str]:
  return {match.group(0).lower() for match in _TERM_RE.finditer(value)}


def _require_unique_identities(
  items: Sequence[SelectionItem],
  *,
  policy: SelectionPolicy,
) -> None:
  values: dict[str, list[str]] = {}
  for item in items:
    values.setdefault(item.source, []).append(item.source_id.strip())
  duplicated = tuple(
    source
    for source in sorted(values, key=lambda value: (_rank(value, policy.source_order), value))
    if len(values[source]) != len(set(values[source]))
  )
  if duplicated:
    raise SelectionIdentityError(duplicated)


def _ordered(
  items: Sequence[SelectionItem],
  *,
  routes: dict[tuple[str, str], tuple[SelectionRoute, ...]],
  matched_terms: dict[tuple[str, str], set[str]],
  policy: SelectionPolicy,
) -> list[tuple[SelectionItem, int]]:
  scored = tuple(
    (
      item,
      _score(item, matched_terms=matched_terms[_identity(item)], policy=policy)
      + _route_score(routes.get(_identity(item), ()), policy=policy),
    )
    for item in items
  )
  return sorted(
    scored,
    key=lambda entry: (
      -entry[1],
      _priority_rank(entry[0].partition, policy.partitions),
      _priority_rank(entry[0].kind, policy.kinds),
      -_updated_at_unix(entry[0].updated_at),
      _rank(entry[0].source, policy.source_order),
      entry[0].source_id,
    ),
  )


def _score(item: SelectionItem, *, matched_terms: set[str], policy: SelectionPolicy) -> int:
  return (
    len(matched_terms) * policy.lexical_weight
    + _priority_score(item.partition, policy.partitions)
    + _priority_score(item.kind, policy.kinds)
    + round(item.confidence * policy.confidence_weight)
  )

def _route_map(
  routes: Sequence[SelectionRoute],
  *,
  policy: SelectionPolicy,
) -> dict[tuple[str, str], tuple[SelectionRoute, ...]]:
  grouped: dict[tuple[str, str], set[SelectionRoute]] = {}
  for route in routes:
    grouped.setdefault((route.source, route.source_id.strip()), set()).add(route)
  return {
    key: tuple(sorted(values, key=lambda route: _route_key(route, policy=policy)))
    for key, values in grouped.items()
  }


def _route_score(routes: Sequence[SelectionRoute], *, policy: SelectionPolicy) -> int:
  return max(
    (max(1, policy.route_rank_ceiling + 1 - route.rank) for route in routes),
    default=0,
  )


def _route_key(route: SelectionRoute, *, policy: SelectionPolicy) -> tuple[int, int, str, str]:
  return (
    _rank(route.lane, policy.route_order),
    route.rank,
    route.signal,
    route.relation or "",
  )


def _candidate(
  item: SelectionItem,
  score: int,
  *,
  routes: Sequence[SelectionRoute],
  matched_terms: set[str],
  policy: SelectionPolicy,
  include_paths: bool,
  reason: _PendingReason = "pending",
) -> _Candidate:
  paths = (
    _paths(item, routes=routes, matched_terms=matched_terms, policy=policy)
    if include_paths
    else ()
  )
  return _Candidate(item=item, score=score, paths=paths, reason=reason)


def _paths(
  item: SelectionItem,
  *,
  routes: Sequence[SelectionRoute],
  matched_terms: set[str],
  policy: SelectionPolicy,
) -> tuple[SelectionPath, ...]:
  paths: list[SelectionPath] = []
  matched = sorted(matched_terms)
  if matched:
    paths.append(SelectionPath(
      lane=policy.lexical_lane,
      rank=None,
      signal=("matched:" + ",".join(matched[:20]))[:512],
    ))
  paths.extend(
    SelectionPath(
      lane=route.lane,
      rank=route.rank,
      signal=route.signal,
      relation=route.relation,
    )
    for route in routes
  )
  historical_limit = policy.max_paths - 1 if paths else policy.max_paths
  historical = item.paths[-historical_limit:] if historical_limit else ()
  available = policy.max_paths - len(historical)
  return (*paths[:available], *historical)


def _outcome(candidate: _Candidate) -> SelectionOutcome:
  item = candidate.item
  evidence = SelectionCandidate(
    source=item.source,
    source_id=item.source_id.strip(),
    partition=item.partition,
    kind=item.kind,
    candidate_rank=candidate.candidate_rank,
    score=candidate.score,
    revision=item.revision,
    text_hash=_text_hash(item.text),
    source_digest=item.source_digest,
    paths=candidate.paths,
  )
  if candidate.selected:
    assert candidate.rank is not None
    return Selected(
      candidate=evidence,
      rank=candidate.rank,
      content_chars=candidate.content_chars,
    )
  reason = "invalid" if candidate.reason == "pending" else candidate.reason
  assert reason != "selected"
  return Omitted(candidate=evidence, reason=reason)


def _append_content(
  content: list[str],
  value: str,
  *,
  used: int,
  max_chars: int,
  marker: str,
) -> str | None:
  separator = 1 if content else 0
  remaining = max_chars - used - separator
  if remaining <= 0:
    return None
  fitted = _fit(value, max_chars=remaining, marker=marker)
  return fitted or None


def _fit(value: str, *, max_chars: int, marker: str) -> str:
  if len(value) <= max_chars:
    return value
  if max_chars <= len(marker):
    return ""
  return value[: max_chars - len(marker)].rstrip() + marker


def _identity(item: SelectionItem) -> tuple[str, str]:
  return item.source, item.source_id.strip()


def _future(item: SelectionItem, *, now: datetime) -> bool:
  return item.available_at is not None and item.available_at > now


def _priority_score(value: str, priorities: Sequence[SelectionPriority]) -> int:
  return next((item.score for item in priorities if item.label == value), 0)


def _priority_rank(value: str, priorities: Sequence[SelectionPriority]) -> int:
  return next((rank for rank, item in enumerate(priorities) if item.label == value), len(priorities))


def _rank(value: str, order: Sequence[str]) -> int:
  try:
    return order.index(value)
  except ValueError:
    return len(order)


def _updated_at_unix(value: datetime | None) -> int:
  if value is None:
    return 0
  current = value.astimezone(timezone.utc)
  return (
    current.toordinal() * 86_400
    + current.hour * 3_600
    + current.minute * 60
    + current.second
  )


def _text_hash(value: str) -> str:
  text = " ".join(value.split())
  return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc(value: datetime) -> datetime:
  if value.tzinfo is None or value.utcoffset() is None:
    raise ValueError("memory_selection_now_timezone_required")
  return value.astimezone(timezone.utc)
