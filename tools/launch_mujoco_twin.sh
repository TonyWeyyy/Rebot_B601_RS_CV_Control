#!/usr/bin/env bash
# 一键启动 Seeed reBot Arm B601-RS MuJoCo Studio 3D 实时数字孪生服务
# 启动后浏览器打开 http://localhost:8080/ ，再运行 python3 palm_follow_v14.py 即可双机同步运动。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STUDIO_PY="${MUJOCO_STUDIO_PYTHON:-$HOME/.venvs/mujoco-studio/bin/python}"
STUDIOD="${STUDIOD_SCRIPT:-$HOME/.gemini/config/plugins/robotics/skills/mujoco-studio-sidecar/scripts/studiod.py}"

if [[ ! -x "${STUDIO_PY}" ]]; then
  echo "[错误] 未找到 MuJoCo Studio Python: ${STUDIO_PY}" >&2
  echo "请先安装: uv venv ~/.venvs/mujoco-studio --python 3.12 && uv pip install --python ~/.venvs/mujoco-studio/bin/python 'mujoco>=4.0.1'" >&2
  exit 1
fi

exec "${STUDIO_PY}" "${STUDIOD}" --workspace_dir="${PROJECT_ROOT}/mujoco_twin" "$@"
