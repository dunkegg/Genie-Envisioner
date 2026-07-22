#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8011}"
CONFIG="${CONFIG:-configs/ltx_model/za_adaptor_tdmpc_server.yaml}"
ADAPTOR_CHECKPOINT="${ADAPTOR_CHECKPOINT:-OUTPUTS/ltx/za_adaptor_tdmpc/2026_07_09_17_38_25/step_20000/za_adaptor.pt}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
TDMPC_PROJECT_ROOT="${TDMPC_PROJECT_ROOT:-}"
TDMPC_MODEL_PATH="${TDMPC_MODEL_PATH:-}"
TDMPC_CFG_ARGS="${TDMPC_CFG_ARGS:-}"
TDMPC_GPU="${TDMPC_GPU:-}"
ACTION_FORMAT="${ACTION_FORMAT:-real}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${ADAPTOR_CHECKPOINT}" ]]; then
    echo "Adaptor checkpoint not found: ${ADAPTOR_CHECKPOINT}" >&2
    exit 1
fi

echo "Starting Genie-Envisioner Za adaptor server"
echo "  host              : ${HOST}"
echo "  port              : ${PORT}"
echo "  config            : ${CONFIG}"
echo "  adaptor checkpoint: ${ADAPTOR_CHECKPOINT}"
echo "  device            : ${DEVICE}"
echo "  dtype             : ${DTYPE}"
echo "  tdmpc project root: ${TDMPC_PROJECT_ROOT:-<from yaml>}"
echo "  tdmpc model path  : ${TDMPC_MODEL_PATH:-<from yaml>}"
echo "  action format     : ${ACTION_FORMAT}"

CMD=(
python3 web_infer_scripts/main_za_adaptor_server.py
    -c "${CONFIG}"
    -w "${ADAPTOR_CHECKPOINT}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --action_format "${ACTION_FORMAT}"
)

if [[ -n "${TDMPC_PROJECT_ROOT}" ]]; then
    CMD+=(--tdmpc_project_root "${TDMPC_PROJECT_ROOT}")
fi
if [[ -n "${TDMPC_MODEL_PATH}" ]]; then
    CMD+=(--tdmpc_model_path "${TDMPC_MODEL_PATH}")
fi
if [[ -n "${TDMPC_CFG_ARGS}" ]]; then
    CMD+=(--tdmpc_cfg_args "${TDMPC_CFG_ARGS}")
fi
if [[ -n "${TDMPC_GPU}" ]]; then
    CMD+=(--tdmpc_gpu "${TDMPC_GPU}")
fi

"${CMD[@]}"
