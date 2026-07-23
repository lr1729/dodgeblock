#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "run-ppo-v2.sh now selects the v4 training contract; use run-ppo-v4.sh directly" >&2
exec "$root/rl/run-ppo-v4.sh"
