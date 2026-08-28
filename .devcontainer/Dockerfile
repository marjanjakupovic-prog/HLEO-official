FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

# libpq-dev + gcc: required by psycopg2-binary at build time; kept as a
# documented system dependency even though the binary wheel is preferred.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*
