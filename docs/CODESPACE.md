# HLEO — GitHub Codespaces

HLEO is Codespace-ready via `.devcontainer/`.

## Flow

```
GitHub repository
→ Create Codespace
→ devcontainer build (Python 3.13)
→ postCreateCommand: bash scripts/setup.sh
→ database ready (Postgres from docker-compose "db" service)
→ boot / health check / HLEO READY
```

## What the devcontainer provides

- `.devcontainer/Dockerfile` — `mcr.microsoft.com/devcontainers/python:1-3.13-bookworm`
  with `libpq-dev` and `gcc` (system deps for `psycopg2-binary`).
- `.devcontainer/devcontainer.json` — merges the root `docker-compose.yml`
  (which starts Postgres) and the devcontainer compose override; runs
  `scripts/setup.sh` after creation; forwards port 8000.
- `.devcontainer/docker-compose.yml` — `devcontainer` service mounting the
  workspace and using the shared `db` service.

## Commands inside the Codespace

```bash
# One-shot full boot + verification
bash scripts/bootloader.sh

# Or step by step
bash scripts/install.sh
bash scripts/setup.sh
bash scripts/start.sh
bash scripts/health-check.sh
```

The app is served on port **8000** (forwarded automatically by the
devcontainer configuration).

## Notes

- `postCreateCommand` runs `setup.sh`, not `bootloader.sh`, so the container
  creation stays fast and deterministic; start the app manually afterwards.
- No secret is committed: copy `.env.example` to `.env` and set only the keys
  you need. Without an LLM key, LLM-dependent endpoints return 503 by design.
