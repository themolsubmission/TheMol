#!/bin/bash
#
# Multi-Target Ligand Generation with Dual Optimization (Docking + SA)
# This script runs:
# 1. Starts Uni-Dock ZMQ server
# 2. Generates molecules with baseline and optimized methods
# 3. Compares results and saves outputs
#

set -e  # Exit on error

# Default parameters (can be overridden via environment variables)
POPULATION_SIZE=${POPULATION_SIZE:-50}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
SA_WEIGHT=${SA_WEIGHT:-1.0}
NUM_SAMPLES=${NUM_SAMPLES:-100}
TARGET=${TARGET:-}
NUM_TARGETS=${NUM_TARGETS:-}
GPU_ID=${GPU_ID:-0}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Multi-Target Dual Optimization Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}GPU: ${GPU_ID}${NC}"
echo -e "${GREEN}Population Size: ${POPULATION_SIZE}${NC}"
echo -e "${GREEN}Generations: ${NUM_GENERATIONS}${NC}"
echo -e "${GREEN}SA Weight: ${SA_WEIGHT}${NC}"
echo -e "${GREEN}Final Samples: ${NUM_SAMPLES}${NC}"
if [ ! -z "$TARGET" ]; then
    echo -e "${GREEN}Target: ${TARGET}${NC}"
fi
if [ ! -z "$NUM_TARGETS" ]; then
    echo -e "${GREEN}Num Targets: ${NUM_TARGETS}${NC}"
fi
echo ""

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UNIDOCK_DIR="${UNIDOCK_DIR:-./gnina-torch}"  # Path to Uni-Dock installation
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIDOCK_LOG="${LOG_DIR}/unidock_server_dual_opt_${TIMESTAMP}.log"
GENERATION_LOG="${LOG_DIR}/dual_optimization_${TIMESTAMP}.log"

echo -e "${YELLOW}Logs will be saved to: ${LOG_DIR}${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$UNIDOCK_PID" ]; then
        echo -e "${YELLOW}Stopping Uni-Dock server (PID: $UNIDOCK_PID)...${NC}"
        kill $UNIDOCK_PID 2>/dev/null || true
        wait $UNIDOCK_PID 2>/dev/null || true
        echo -e "${GREEN}Uni-Dock server stopped${NC}"
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Step 1: Start Uni-Dock ZMQ Server
echo -e "${BLUE}Step 1: Checking Uni-Dock ZMQ Server${NC}"

EXISTING_SERVER_PID=$(ps aux | grep "[p]ython.*unidock_zmq_server" | awk '{print $2}' | head -1)

if [ ! -z "$EXISTING_SERVER_PID" ]; then
    echo -e "${YELLOW}Uni-Dock server already running (PID: ${EXISTING_SERVER_PID})${NC}"
    echo -e "${GREEN}Using existing server${NC}"
    UNIDOCK_PID=""
else
    echo -e "${YELLOW}Starting new Uni-Dock server...${NC}"
    echo -e "${YELLOW}Activating unidock_env conda environment...${NC}"

    cd "${UNIDOCK_DIR}"

    source ~/anaconda3/bin/activate unidock_env
    nohup python -u unidock_zmq_server.py > "${UNIDOCK_LOG}" 2>&1 &
    UNIDOCK_PID=$!

    echo -e "${GREEN}Uni-Dock server started (PID: ${UNIDOCK_PID})${NC}"
    echo -e "${YELLOW}  Log file: ${UNIDOCK_LOG}${NC}"

    echo -e "${YELLOW}Waiting for Uni-Dock server to initialize...${NC}"
    sleep 3

    if ps -p $UNIDOCK_PID > /dev/null; then
        echo -e "${GREEN}Uni-Dock server is running${NC}"
        if grep -q "listening on port" "${UNIDOCK_LOG}" 2>/dev/null; then
            echo -e "${GREEN}Uni-Dock server is ready and listening${NC}"
        else
            echo -e "${YELLOW}Server started but waiting for ready signal...${NC}"
            sleep 2
        fi
    else
        echo -e "${RED}Failed to start Uni-Dock server${NC}"
        echo -e "${RED}Check log file: ${UNIDOCK_LOG}${NC}"
        exit 1
    fi
fi

echo ""

# Step 2: Run Dual Optimization
echo -e "${BLUE}Step 2: Running Dual Optimization${NC}"
echo -e "${YELLOW}Activating lf_cfm_cma conda environment...${NC}"

cd "${SCRIPT_DIR}"

source ~/anaconda3/bin/activate lf_cfm_cma

echo -e "${GREEN}Starting dual optimization...${NC}"
echo -e "${YELLOW}  Log file: ${GENERATION_LOG}${NC}"
echo ""

# Build command with parameters
CMD="python -u test_sampling_multi_target_v3.py"
CMD="${CMD} --population-size ${POPULATION_SIZE}"
CMD="${CMD} --num-generations ${NUM_GENERATIONS}"
CMD="${CMD} --sa-weight ${SA_WEIGHT}"
CMD="${CMD} --num-final-samples ${NUM_SAMPLES}"
CMD="${CMD} --gpu ${GPU_ID}"

if [ ! -z "$TARGET" ]; then
    CMD="${CMD} --target ${TARGET}"
fi

if [ ! -z "$NUM_TARGETS" ]; then
    CMD="${CMD} --num-targets ${NUM_TARGETS}"
fi

echo -e "${YELLOW}Running: ${CMD}${NC}"
echo ""

eval "${CMD}" 2>&1 | tee "${GENERATION_LOG}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Dual optimization completed!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "${YELLOW}Results saved to: ./multi_target_dual_results/${NC}"
    echo -e "${YELLOW}Generation log: ${GENERATION_LOG}${NC}"
    echo -e "${YELLOW}Uni-Dock log: ${UNIDOCK_LOG}${NC}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}Optimization failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo -e "${YELLOW}Check log file: ${GENERATION_LOG}${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}All done!${NC}"
