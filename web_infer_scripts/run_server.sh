#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Cross-machine clients cannot connect to localhost. Keep this as 0.0.0.0
# unless you intentionally want to serve only on one specific network interface.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Default to the existing action model checkpoint in this workspace.
# Override these from the shell when serving another trained model.
CONFIG="${CONFIG:-configs/ltx_model/policy_model_lerobot.yaml}"
WEIGHT="${WEIGHT:-/mnt/pfs/s7fsio/code/Genie-Envisioner/OUTPUTS/ltx/action_video/2026_07_07_16_31_47/step_20000}"
DOMAIN_NAME="${DOMAIN_NAME:-habitat_lerobot}"

# Must match diffusion_model.config.action_in_channels and the client state/action layout.
ACTION_DIM="${ACTION_DIM:-2}"
DENOISE_STEP_VALUE="${DENOISE_STEP:-10}"
THRESHOLD="${THRESHOLD:-200}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -e "${WEIGHT}" ]]; then
    echo "Weight path not found: ${WEIGHT}" >&2
    exit 1
fi

echo "Starting Genie-Envisioner web inference server"
echo "  host        : ${HOST}"
echo "  port        : ${PORT}"
echo "  config      : ${CONFIG}"
echo "  weight      : ${WEIGHT}"
echo "  domain_name : ${DOMAIN_NAME}"
echo "  action_dim  : ${ACTION_DIM}"
echo "  denoise_step: ${DENOISE_STEP_VALUE}"
echo "  threshold   : ${THRESHOLD}"

python3 web_infer_scripts/main_server.py \
    -c "${CONFIG}" \
    -w "${WEIGHT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --domain_name "${DOMAIN_NAME}" \
    --denoise_step "${DENOISE_STEP_VALUE}" \
    --action_dim "${ACTION_DIM}" \
    --threshold "${THRESHOLD}" \
    --add_state
