#!/usr/bin/env bash
# Build the predicators-sandbox Docker image.
#
# Run from anywhere inside the repository:
#   bash docker/build.sh
#
# Rebuild when PyPI dependencies in setup.py change.
# No rebuild needed for predicators source changes (bind-mounted at runtime).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building predicators-sandbox from ${REPO_ROOT} ..."
docker build \
    --tag predicators-sandbox \
    --file "${REPO_ROOT}/docker/Dockerfile" \
    "${REPO_ROOT}"
echo "Done. Image tagged: predicators-sandbox"
