#!/bin/bash

# Run FunASR Triton Inference Server
# Usage: ./run_server.sh [options]

set -e
export CUDA_VISIBLE_DEVICES="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="${SCRIPT_DIR}/model_repo_funasr"

# Default settings
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002
LOG_VERBOSE=0
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-repo)
            MODEL_REPO="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --grpc-port)
            GRPC_PORT="$2"
            shift 2
            ;;
        --metrics-port)
            METRICS_PORT="$2"
            shift 2
            ;;
        --verbose)
            LOG_VERBOSE=1
            shift
            ;;
        --gpu)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-repo PATH    Model repository path (default: ./model_repo_funasr)"
            echo "  --http-port PORT     HTTP port (default: 8000)"
            echo "  --grpc-port PORT     gRPC port (default: 8001)"
            echo "  --metrics-port PORT  Metrics port (default: 8002)"
            echo "  --gpu DEVICE         GPU device ID (default: 0)"
            echo "  --verbose            Enable verbose logging"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

echo "=========================================="
echo "FunASR Triton Inference Server"
echo "=========================================="
echo "Model Repository: ${MODEL_REPO}"
echo "HTTP Port: ${HTTP_PORT}"
echo "gRPC Port: ${GRPC_PORT}"
echo "Metrics Port: ${METRICS_PORT}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Check if model repository exists
if [ ! -d "${MODEL_REPO}" ]; then
    echo "Error: Model repository not found: ${MODEL_REPO}"
    exit 1
fi

# Build tritonserver command
CMD="tritonserver"
CMD="${CMD} --model-repository=${MODEL_REPO}"
CMD="${CMD} --http-port=${HTTP_PORT}"
CMD="${CMD} --grpc-port=${GRPC_PORT}"
CMD="${CMD} --metrics-port=${METRICS_PORT}"
CMD="${CMD} --allow-http=true"
CMD="${CMD} --allow-grpc=true"
CMD="${CMD} --allow-metrics=true"

if [ "${LOG_VERBOSE}" -eq 1 ]; then
    CMD="${CMD} --log-verbose=1"
fi

echo "Starting server..."
echo "Command: ${CMD}"
echo ""

exec ${CMD}
