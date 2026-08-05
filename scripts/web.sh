#!/usr/bin/env bash
# Robot Pipeline Web 控制台一键启动脚本
#
# 用法:
#   bash scripts/web.sh                 # 默认 0.0.0.0:8000, token=dev-token
#   ROBOT_WEB_TOKEN=xxx bash scripts/web.sh   # 自定义访问令牌
#   ROBOT_WEB_PORT=9000 bash scripts/web.sh   # 自定义端口
#
# 浏览器访问: http://<本机IP>:<端口>/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TOKEN="${ROBOT_WEB_TOKEN:-123}"
PORT="${ROBOT_WEB_PORT:-8000}"
HOST="${ROBOT_WEB_HOST:-0.0.0.0}"

echo "=============================================="
echo " Robot Pipeline Web 控制台"
echo "  地址:  http://${HOST}:${PORT}/"
echo "  Token: ${TOKEN}"
echo "  (局域网内其他电脑可用本机 IP 访问)"
echo "=============================================="
# 端口占用检测：若已有一个实例在运行，提示并退出，避免 Errno 98
if ss -tln | grep -q "[:.]${PORT} "; then
  pid=$(ss -tlnp 2>/dev/null | grep "[:.]${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
  echo "⚠️  端口 ${PORT} 已被占用 (pid=${pid:-未知})，可能已有实例在运行。"
  echo "   如需重启: 先关闭旧实例: kill -9 ${pid:-<pid>} 或者 killall uvicorn"
  exit 1
fi
# 前端未构建则先构建
if [ ! -d "web/frontend/dist" ]; then
  echo "[web] 前端未构建, 正在构建..."
  (cd web/frontend && npm run build)
fi

ROBOT_WEB_TOKEN="${TOKEN}" exec uv run uvicorn web.main:app --host "${HOST}" --port "${PORT}"
