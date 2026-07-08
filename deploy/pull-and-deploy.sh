#!/usr/bin/env bash
set -euo pipefail
cd /home/avi/avinash-agent-backend

BEFORE=$(git rev-parse HEAD)
git fetch origin main
git reset --hard origin/main
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" != "$AFTER" ]; then
  echo "New commit $AFTER, redeploying"
  docker compose up -d --build
fi
