# Security policy

## Scope
This project processes **public, aggregate, district-level data only**. It must never ingest
personal, household-level or otherwise sensitive data.

## Reporting a vulnerability
Please open a private security advisory on GitHub (Security tab) rather than a public issue.
Include steps to reproduce and the affected version. Expect an initial reply within 7 days.

## Practices in this repo
- Secrets live in `.env` (git-ignored); `.env.example` documents the names only.
- `pre-commit` runs gitleaks, private-key detection and large-file checks.
- CI runs Ruff, tests, Bandit (code) and pip-audit (dependencies) with read-only permissions.
- Never load pickle/joblib files from untrusted sources; prefer ONNX, `skops` or native formats.
- Raw data is immutable and versioned with DVC; verify checksums after download.
- Respect each dataset's licence (e.g. NFHS/DHS access terms); do not redistribute restricted data.
