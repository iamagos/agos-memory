# Security

## Reporting

Report vulnerabilities through this repository's private security-advisory
form. Do not open a public issue for an undisclosed vulnerability.

Include the affected version, impact, reproduction steps, and any suggested
fix. We will acknowledge a report after triage and coordinate disclosure when
a fix is available.

## Scope

`agos-memory` is a pure decision kernel. It performs no I/O and grants no
authority. Security reports should distinguish kernel behavior from the
authorization, storage, retrieval, model, and execution layers owned by an
integrating application.

Only the latest published `0.x` release receives security fixes before 1.0.
