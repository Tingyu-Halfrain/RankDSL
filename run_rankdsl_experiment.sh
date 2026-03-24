#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_rankdsl_experiment() {
  python run_rankdsl_experiment.py "$@"
}

run_rankdsl_experiment_deepseek() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    python run_rankdsl_experiment.py --help
    return 0
  fi

  local env_file="${DEEPSEEK_ENV_FILE:-.env.deepseek}"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi

  local api_key="${DEEPSEEK_API_KEY:-${RANKDSL_API_KEY:-}}"
  local base_url="${DEEPSEEK_BASE_URL:-${RANKDSL_BASE_URL:-https://api.deepseek.com/v1}}"
  local model="${DEEPSEEK_MODEL:-deepseek-chat}"

  if [[ -z "${api_key}" ]]; then
    echo "Missing DeepSeek API key. Set DEEPSEEK_API_KEY or RANKDSL_API_KEY, or create .env.deepseek." >&2
    return 1
  fi

  RANKDSL_API_KEY="$api_key" \
  RANKDSL_BASE_URL="$base_url" \
    python run_rankdsl_experiment.py \
      --llm-mode api \
      --model "$model" \
      "$@"
}

show_help() {
  cat <<'EOF'
Usage:
  ./run_rankdsl_experiment.sh [python-args...]
  ./run_rankdsl_experiment.sh deepseek [python-args...]

Commands:
  deepseek   Load DeepSeek API config from env or .env.deepseek, then run the experiment in api mode.

DeepSeek env resolution order:
  1. .env.deepseek (or $DEEPSEEK_ENV_FILE if set)
  2. DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  3. RANKDSL_API_KEY / RANKDSL_BASE_URL

.env.deepseek example:
  DEEPSEEK_API_KEY=sk-xxxx
  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
  DEEPSEEK_MODEL=deepseek-chat
EOF
}

if [[ "${1:-}" == "deepseek" ]]; then
  shift
  run_rankdsl_experiment_deepseek "$@"
elif [[ "${1:-}" == "--help-deepseek" ]]; then
  show_help
else
  run_rankdsl_experiment "$@"
fi
