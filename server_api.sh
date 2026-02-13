#!/usr/bin/env bash

# Control script for whisper.cpp HTTP API (FastAPI + uvicorn)
# Usage:
#   ./server_api.sh start
#   ./server_api.sh stop
#   ./server_api.sh restart
#   ./server_api.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="whispercpp_env"
HOST="0.0.0.0"
PORT="8700"
PID_FILE="${SCRIPT_DIR}/api/whisper_api.pid"
LOG_FILE="${SCRIPT_DIR}/api/whisper_api.log"

WHISPER_BIN_DEFAULT="./build/bin/whisper-cli"
WHISPER_MODEL_DEFAULT="medium"
WHISPER_MODELS_DIR_DEFAULT="./models"

ensure_api_dir() {
  mkdir -p "${SCRIPT_DIR}/api"
}

is_running_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  if ps -p "${pid}" > /dev/null 2>&1; then
    return 0
  fi
  return 1
}

port_pids() {
  # Return PIDs listening on PORT (IPv4/IPv6)
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -i :"${PORT}" 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${PORT}" 2>/dev/null | awk 'NR>1 {print $NF}' | sed 's/.*pid=\([0-9]\+\).*/\1/' || true
  else
    return 0
  fi
}

start_server() {
  ensure_api_dir

  if [[ -f "${PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_running_pid "${old_pid}"; then
      echo "Server already running with PID ${old_pid}"
      return 0
    else
      rm -f "${PID_FILE}"
    fi
  fi

  # Clean any existing process on the port before starting
  local pids
  pids="$(port_pids || true)"
  if [[ -n "${pids}" ]]; then
    echo "Killing existing process(es) on port ${PORT}: ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 1
  fi

  echo "Starting whisper.cpp HTTP API on ${HOST}:${PORT} ..."

  # Environment variables for app.py
  export WHISPER_BIN="${WHISPER_BIN_DEFAULT}"
  export WHISPER_MODEL="${WHISPER_MODEL_DEFAULT}"
  export WHISPER_MODELS_DIR="${WHISPER_MODELS_DIR_DEFAULT}"

  # Use conda run to avoid relying on shell-specific activation
  nohup conda run -n "${ENV_NAME}" uvicorn api.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 2 \
    >> "${LOG_FILE}" 2>&1 &

  local new_pid=$!
  echo "${new_pid}" > "${PID_FILE}"
  echo "Started with PID ${new_pid}, logs: ${LOG_FILE}"
}

stop_server() {
  echo "Stopping whisper.cpp HTTP API ..."

  local stopped_any=0

  # Stop by PID file if present
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_running_pid "${pid}"; then
      echo "Killing PID from pidfile: ${pid}"
      kill "${pid}" 2>/dev/null || true
      stopped_any=1
    fi
    rm -f "${PID_FILE}"
  fi

  # Also clean any processes still holding the port
  local pids
  pids="$(port_pids || true)"
  if [[ -n "${pids}" ]]; then
    echo "Killing leftover process(es) on port ${PORT}: ${pids}"
    kill ${pids} 2>/dev/null || true
    stopped_any=1
  fi

  if [[ "${stopped_any}" -eq 0 ]]; then
    echo "No running server processes found."
  else
    echo "Stop signal sent. Processes should exit shortly."
  fi
}

status_server() {
  local pid=""
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_running_pid "${pid}"; then
      echo "Server status: RUNNING (PID ${pid})"
    else
      echo "Server status: NOT RUNNING (stale pidfile ${PID_FILE})"
    fi
  else
    echo "Server status: NOT RUNNING (no pidfile)"
  fi

  local pids
  pids="$(port_pids || true)"
  if [[ -n "${pids}" ]]; then
    echo "Port ${PORT} is IN USE by PID(s): ${pids}"
  else
    echo "Port ${PORT} is FREE"
  fi
}

restart_server() {
  stop_server
  sleep 1
  start_server
}

case "${1-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    restart_server
    ;;
  status)
    status_server
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac

