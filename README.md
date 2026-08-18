# agos-memory

A small, deterministic decision kernel for already-authorized memory values.
It performs no I/O and grants no authority.

```text
Host acquisition -> finite authorized values -> pure decisions -> Host effects
                                           admit / retain / select / support
```

The package owns four decisions:

| Function | Decision |
| --- | --- |
| `admit()` | Accept, reject, or replace proposed memory. |
| `retain()` | Retain or omit one record from the current context. |
| `select()` | Rank and fit finite candidates with complete outcome evidence. |
| `support()` | Compare an exact dependency with an already-reopened source. |

Storage, authorization, retrieval, embeddings, models, prompts, graphs, and
receipts belong to the application integrating the kernel.

## Install

With pip:

```bash
python -m pip install agos-memory
```

With uv:

```bash
uv add agos-memory
```

## Example

```python
from agos_memory.support import source_digest, support
from agos_memory.types import Current, ReopenedSource, SourceDependency

text = "Debt matures in 2029."
digest = source_digest(text)
expected = SourceDependency(
    owner="document-1",
    revision="version-1",
    fragment="chunk-1",
    kind="raw",
    digest=digest,
)
reopened = ReopenedSource(
    owner="document-1",
    revision="version-1",
    fragment="chunk-1",
    kind="raw",
    digest=digest,
    current_revision="version-1",
)

assert support(expected, reopened) == Current()
```

Inputs are explicit immutable values. Time, limits, policy, partitions, routes,
and reopened source state come from the caller. Labels and scores never create
authority.

## Develop

```bash
uv sync --locked --all-groups
uv run pytest
uv run ruff check .
uv build --no-sources
```

The wheel and source archive are tested as external packages on Python 3.11,
3.12, and 3.13. Runtime dependencies remain empty.

## Research

Reproducible public experiments live in
[`agos-memory-lab`](https://github.com/iamagos/agos-memory-lab). Benchmark
dependencies, datasets, model calls, and results stay outside this kernel.

## Security

See [SECURITY.md](SECURITY.md). Please report vulnerabilities privately rather
than opening a public issue.

## License

Copyright 2026 I am Agos, Inc. Licensed under the Apache License, Version 2.0.
