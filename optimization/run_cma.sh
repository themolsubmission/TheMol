#!/bin/bash
#
# CMA-ES Molecular Optimization with Uni-Dock
# This script runs the complete optimization pipeline:
# 1. Starts Uni-Dock ZMQ server
# 2. Runs CMA-ES optimization with molecular generation and docking
# 3. Generates final molecules from optimized latent space
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}CMA-ES Molecular Optimization Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UNIDOCK_DIR="/home/csy/work1/gnina-torch"
LOG_DIR="${SCRIPT_DIR}/logs"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIDOCK_LOG="${LOG_DIR}/unidock_server_${TIMESTAMP}.log"
OPTIMIZATION_LOG="${LOG_DIR}/cma_optimization_${TIMESTAMP}.log"

echo -e "${YELLOW}Logs will be saved to: ${LOG_DIR}${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # Kill Uni-Dock server if running
    if [ ! -z "$UNIDOCK_PID" ]; then
        echo -e "${YELLOW}Stopping Uni-Dock server (PID: $UNIDOCK_PID)...${NC}"
        kill $UNIDOCK_PID 2>/dev/null || true
        wait $UNIDOCK_PID 2>/dev/null || true
        echo -e "${GREEN}Uni-Dock server stopped${NC}"
    fi

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Step 1: Start Uni-Dock ZMQ Server
echo -e "${BLUE}Step 1: Checking Uni-Dock ZMQ Server${NC}"

# Check if Uni-Dock server is already running
EXISTING_SERVER_PID=$(ps aux | grep "[p]ython.*unidock_zmq_server" | awk '{print $2}' | head -1)

if [ ! -z "$EXISTING_SERVER_PID" ]; then
    echo -e "${YELLOW}⚠ Uni-Dock server already running (PID: ${EXISTING_SERVER_PID})${NC}"
    echo -e "${GREEN}✓ Using existing server${NC}"
    UNIDOCK_PID=""  # Don't kill existing server on exit
else
    echo -e "${YELLOW}Starting new Uni-Dock server...${NC}"
    echo -e "${YELLOW}Activating unidock_env conda environment...${NC}"

    cd "${UNIDOCK_DIR}"

    # Start Uni-Dock server in background
    source /home/csy/anaconda3/bin/activate unidock_env
    nohup python -u unidock_zmq_server.py > "${UNIDOCK_LOG}" 2>&1 &
    UNIDOCK_PID=$!

    echo -e "${GREEN}✓ Uni-Dock server started (PID: ${UNIDOCK_PID})${NC}"
    echo -e "${YELLOW}  Log file: ${UNIDOCK_LOG}${NC}"

    # Wait for server to initialize
    echo -e "${YELLOW}Waiting for Uni-Dock server to initialize...${NC}"
    sleep 3

    # Check if server is running
    if ps -p $UNIDOCK_PID > /dev/null; then
        echo -e "${GREEN}✓ Uni-Dock server is running${NC}"

        # Check server log for confirmation
        if grep -q "listening on port" "${UNIDOCK_LOG}" 2>/dev/null; then
            echo -e "${GREEN}✓ Uni-Dock server is ready and listening${NC}"
        else
            echo -e "${YELLOW}⚠ Server started but waiting for ready signal...${NC}"
            sleep 2
        fi
    else
        echo -e "${RED}✗ Failed to start Uni-Dock server${NC}"
        echo -e "${RED}Check log file: ${UNIDOCK_LOG}${NC}"
        exit 1
    fi
fi

echo ""

# Step 2: Run CMA-ES Optimization
echo -e "${BLUE}Step 2: Running CMA-ES Optimization${NC}"
echo -e "${YELLOW}Activating lf_cfm_cma conda environment...${NC}"

cd "${SCRIPT_DIR}"

# Activate conda environment and run optimization
source /home/csy/anaconda3/bin/activate lf_cfm_cma

echo -e "${GREEN}✓ Starting molecular generation and optimization...${NC}"
echo -e "${YELLOW}  Log file: ${OPTIMIZATION_LOG}${NC}"
echo -e "${YELLOW}  This may take a while (50 generations × 100 molecules)...${NC}"
echo ""

# Run the optimization script
python test_sampling_optimize_fromNoise.py 2>&1 | tee "${OPTIMIZATION_LOG}"

# Check if optimization completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✓ Optimization completed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "${YELLOW}Results saved to: ./test_direct_sampling/${NC}"
    echo -e "${YELLOW}Optimization log: ${OPTIMIZATION_LOG}${NC}"
    echo -e "${YELLOW}Uni-Dock log: ${UNIDOCK_LOG}${NC}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}✗ Optimization failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo -e "${YELLOW}Check log file: ${OPTIMIZATION_LOG}${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}All done!${NC}"
