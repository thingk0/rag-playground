#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ZAI_API_KEY="your_key" ./claude-glm-once.sh
#   ./claude-glm-once.sh "your_key"
#   ZAI_API_KEY="your_key" ./claude-glm-once.sh --resume

ZAI_KEY="bda9e1f243a9472d802c0de345ed18c3.uz3Xh6Ki8k3BD4Co"

if [[ -z "$ZAI_KEY" ]]; then
  cat <<'USAGE'
Usage:
  ZAI_API_KEY="your_key" ./claude-glm-once.sh
  ./claude-glm-once.sh "your_key"
  ZAI_API_KEY="your_key" ./claude-glm-once.sh --resume

This script starts Claude Code for this run only with Z.ai GLM mapped through the Anthropic-compatible endpoint.
USAGE
  exit 1
fi

if [[ "${1:-}" == "$ZAI_KEY" ]]; then
  shift || true
fi

export ANTHROPIC_AUTH_TOKEN="$ZAI_KEY"
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export API_TIMEOUT_MS="3000000"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="glm-4.5-air"
export ANTHROPIC_DEFAULT_SONNET_MODEL="glm-5.1"
export ANTHROPIC_DEFAULT_OPUS_MODEL="glm-5.1"

exec claude "$@"
