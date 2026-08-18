from dataclasses import FrozenInstanceError

import pytest

from agos_memory.support import source_digest, support
from agos_memory.types import (
  Current,
  Missing,
  ReopenedSource,
  Replaced,
  SourceDependency,
  Stale,
)


DIGEST = source_digest(" Debt matures in 2029. ")
EXPECTED = SourceDependency(
  owner="file-1",
  revision="version-1",
  fragment="chunk-1",
  kind="raw",
  digest=DIGEST,
)


def test_support_returns_one_exact_immutable_state() -> None:
  assert support(EXPECTED, _reopened()) == Current()
  assert support(EXPECTED, _reopened(current_revision="version-2")) == Replaced()
  assert support(EXPECTED, _reopened(digest=source_digest("Debt matures in 2027."))) == Stale()
  assert support(EXPECTED, _reopened(digest=None)) == Stale()
  assert support(EXPECTED, _reopened(current_revision=None)) == Missing()
  assert support(EXPECTED, None) == Missing()

  with pytest.raises(FrozenInstanceError):
    EXPECTED.digest = source_digest("changed")  # type: ignore[misc]


@pytest.mark.parametrize(
  ("field", "value"),
  (
    ("owner", "file-2"),
    ("revision", "version-2"),
    ("fragment", "chunk-2"),
    ("kind", "derived"),
  ),
)
def test_support_treats_a_different_exact_identity_as_missing(field: str, value: str) -> None:
  values = {
    "owner": "file-1",
    "revision": "version-1",
    "fragment": "chunk-1",
    "kind": "raw",
    "digest": DIGEST,
    "current_revision": "version-1",
  }
  values[field] = value

  assert support(EXPECTED, ReopenedSource(**values)) == Missing()  # type: ignore[arg-type]


def test_source_digest_preserves_exact_bytes() -> None:
  assert DIGEST == "sha256:89ec09f1cccae495afe5bcf930db00f97940ace1173c1dd64df541791d20a871"
  assert DIGEST != source_digest("Debt matures in 2029.")


@pytest.mark.parametrize("field", ("owner", "revision", "fragment", "kind"))
def test_support_values_require_identity(field: str) -> None:
  values = {
    "owner": "file-1",
    "revision": "version-1",
    "fragment": "chunk-1",
    "kind": "raw",
    "digest": DIGEST,
  }
  values[field] = " "

  with pytest.raises(ValueError, match=f"memory_support_{field}_invalid"):
    SourceDependency(**values)


@pytest.mark.parametrize("digest", ("", "sha256:ABC", "md5:" + "a" * 32, "sha256:" + "g" * 64))
def test_support_values_require_canonical_sha256(digest: str) -> None:
  with pytest.raises(ValueError, match="memory_support_digest_invalid"):
    SourceDependency(
      owner="file-1",
      revision="version-1",
      fragment="chunk-1",
      kind="raw",
      digest=digest,
    )


def test_reopened_source_requires_canonical_state() -> None:
  with pytest.raises(ValueError, match="memory_support_digest_invalid"):
    _reopened(digest="sha256:" + "A" * 64)
  with pytest.raises(ValueError, match="memory_support_current_revision_invalid"):
    _reopened(current_revision=" ")


def test_source_digest_rejects_non_text() -> None:
  with pytest.raises(ValueError, match="memory_support_text_invalid"):
    source_digest(b"text")  # type: ignore[arg-type]


def _reopened(
  *,
  digest: str | None = DIGEST,
  current_revision: str | None = "version-1",
) -> ReopenedSource:
  return ReopenedSource(
    owner="file-1",
    revision="version-1",
    fragment="chunk-1",
    kind="raw",
    digest=digest,
    current_revision=current_revision,
  )
