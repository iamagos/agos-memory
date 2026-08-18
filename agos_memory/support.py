from __future__ import annotations

from hashlib import sha256

from agos_memory.types import (
  Current,
  Missing,
  ReopenedSource,
  Replaced,
  SourceDependency,
  Stale,
  SupportDecision,
)


def support(
  expected: SourceDependency,
  reopened: ReopenedSource | None,
) -> SupportDecision:
  """Compare one exact dependency with one already-reopened source."""

  if reopened is None or _identity(reopened) != _identity(expected):
    return Missing()
  if reopened.digest != expected.digest:
    return Stale()
  if reopened.current_revision is None:
    return Missing()
  if reopened.current_revision == expected.revision:
    return Current()
  return Replaced()


def source_digest(text: str) -> str:
  if not isinstance(text, str):
    raise ValueError("memory_support_text_invalid")
  return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def _identity(source: SourceDependency | ReopenedSource) -> tuple[str, str, str, str]:
  return source.owner, source.revision, source.fragment, source.kind
