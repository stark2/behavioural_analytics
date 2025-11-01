#!/bin/bash

# Configuration
PYTHON_PATH="/Users/david/miniforge3/bin/python"
SCRIPT_PATH="/Users/david/Documents/Data/python-workspace/behavioural_analytics/behavio_client.py"
CSV_FILE="session_mickey_1.csv"
USERNAME="Mickey Mouse"
CONTRACT="795131459"
TOPN=10
INTERVAL=5

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Behavioral Analytics Monitor"
echo "=========================================="
echo "Monitoring every ${INTERVAL} seconds..."
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Counter for iterations
counter=0

# Trap Ctrl+C to exit gracefully
trap 'echo -e "\n${RED}Monitoring stopped.${NC}"; exit 0' INT

while true; do
    counter=$((counter + 1))
    
    # Get current datetime
    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${BLUE}[Call #${counter}] ${datetime}${NC}"
    
    # Execute the command and capture output
    output=$("$PYTHON_PATH" "$SCRIPT_PATH" eval \
        --csv "$CSV_FILE" \
        --username "$USERNAME" \
        --contract "$CONTRACT" \
        --topN "$TOPN" 2>&1)
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        # Extract similarity_score using grep and awk
        similarity_score=$(echo "$output" | grep -o '"similarity_score": [0-9.]*' | awk '{print $2}')
        
        if [ -n "$similarity_score" ]; then
            echo -e "${GREEN}Similarity Score: ${similarity_score}${NC}"
        else
            echo -e "${YELLOW}Similarity score not found in response${NC}"
        fi
    else
        echo -e "${RED}Error executing command${NC}"
        echo "$output"
    fi
    
    echo "----------------------------------------"
    echo ""
    
    # Wait for specified interval
    sleep "$INTERVAL"
done