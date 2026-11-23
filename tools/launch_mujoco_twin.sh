#!/usr/bin/env bash
# 一键启动 Seeed reBot Arm B601-RS MuJoCo Studio 3D 实时数字孪生
#
# 用法：
#   ./tools/launch_mujoco_twin.sh --native              # 本地原生桌面 3D 窗口（推荐本地查看，无需浏览器）
#   ./tools/launch_mujoco_twin.sh --web                 # 独立浏览器 Web 模式（http://localhost:8080/）
#   ./tools/launch_mujoco_twin.sh --sidecar             # Antigravity 侧边栏守护模式（studiod）
#   ./tools/launch_mujoco_twin.sh --native --udp-port 19870
#
# 未指定模式时默认以 --native（本地原生桌面窗口）启动。
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

MODE="native"
PASS_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --native)
      MODE="native"
      ;;
    --web)
      MODE="web"
      ;;
    --sidecar)
      MODE="sidecar"
      ;;
    *)
      PASS_ARGS+=("${arg}")
      ;;
  esac
done

case "${MODE}" in
  native)
    exec "${STUDIO_PY}" "${PROJECT_ROOT}/mujoco_twin/run_twin.py" --native "${PASS_ARGS[@]}"
    ;;
  web)
    exec "${STUDIO_PY}" "${PROJECT_ROOT}/mujoco_twin/run_twin.py" --web "${PASS_ARGS[@]}"
    ;;
  sidecar)
    if [[ ! -f "${STUDIOD}" ]]; then
      echo "[错误] 未找到 Antigravity studiod.py: ${STUDIOD}" >&2
      echo "请改用 --native 或 --web 独立模式启动。" >&2
      exit 1
    fi
    exec "${STUDIO_PY}" "${STUDIOD}" --workspace_dir="${PROJECT_ROOT}/mujoco_twin" "${PASS_ARGS[@]}"
    ;;
esac
