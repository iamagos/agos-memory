# Contributing

Make one thing clearly better.

This package is a pure decision kernel. It receives already-authorized finite
values and returns deterministic decisions. If a proposal needs storage,
retrieval, models, prompts, authorization, providers, graph ownership, or I/O,
it belongs in the integrating application.

## Before a pull request

- Read `README.md`, `AGENTS.md`, and the owning module and tests.
- Explain why the change should exist and why the current primitives cannot
  express it.
- Keep prerequisite refactors separate from behavior changes.
- Remove unrelated formatting, generated churn, and compatibility paths.
- Disclose material AI assistance and review every resulting line and claim.

Open a pull request only when the change is coherent and ready to merge.

## What belongs

- A smaller or clearer expression of an existing decision.
- A regression fix with an executable law.
- A necessary typed input, outcome, or omission with one obvious owner.
- A public behavior whose determinism is explicit in its inputs.

Runtime dependencies remain empty. New public surface has a higher burden than
composition from existing values and functions. Labels, scores, environment,
time, and hidden state never create authority.

Do not add an engine, store, retriever, provider registry, compatibility facade,
or second representation of an existing fact.

## Proof

Run the complete local gate:

```sh
uv sync --locked --all-groups
uv run pytest
uv run ruff check .
uv run mypy agos_memory
uv build --no-sources
```

Bug fixes need a regression law. Refactors must preserve exact outcomes.
Performance claims need a reproducible before-and-after benchmark. CI also tests
the wheel and source archive as external packages on Python 3.11, 3.12, and
3.13.
