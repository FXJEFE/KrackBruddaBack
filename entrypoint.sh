#!/bin/sh
# secrets only at runtime: /run/secrets/kraken_env (docker secret) or --env-file; never baked into the image
set -e
[ -f /run/secrets/kraken_env ] && set -a && . /run/secrets/kraken_env && set +a
case "$1" in
  bridge)  exec python -m bridge ;;
  feature) exec python -m models.feature_service ;;
  trader)  exec python -m trader ;;
  dashboard) exec python /app/hub/dashboard_hub.py ;;
  agent)   exec python /app/agent/server.py ;;
  *) exec "$@" ;;
esac
