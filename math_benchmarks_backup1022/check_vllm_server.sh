#!/bin/bash
# Check vLLM server status and metrics

HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"

echo "============================================================"
echo "VLLM SERVER STATUS"
echo "============================================================"
echo "Checking: http://${HOST}:${PORT}"
echo ""

# Check if server is responding
if curl -s http://${HOST}:${PORT}/health >/dev/null 2>&1; then
    echo "✓ Server is ONLINE"
else
    echo "✗ Server is OFFLINE or not responding"
    exit 1
fi

# Get models
echo ""
echo "Available models:"
curl -s http://${HOST}:${PORT}/v1/models | python3 -m json.tool 2>/dev/null || echo "  (could not retrieve)"

# Check for PID file
echo ""
if [ -f "vllm_server.pid" ]; then
    PID=$(cat vllm_server.pid)
    echo "Server PID: ${PID}"
    if kill -0 ${PID} 2>/dev/null; then
        echo "✓ Process is running"
    else
        echo "✗ PID file exists but process is not running"
    fi
else
    echo "No PID file found (server may have been started manually)"
fi

echo ""
echo "============================================================"
echo "Endpoints:"
echo "  Health:  http://${HOST}:${PORT}/health"
echo "  Models:  http://${HOST}:${PORT}/v1/models"
echo "  Metrics: http://${HOST}:${PORT}/metrics"
echo "  API:     http://${HOST}:${PORT}/v1/completions"
echo "============================================================"
