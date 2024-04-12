#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\r"
    done
    printf "    \r"
}

echo -e "${GREEN}Syncing...${NC}"

API_KEY=$OPENAI_API_KEY
if [ -z "$API_KEY" ]; then
    echo -e "${RED}OPENAI_API_KEY is not set. Please export the OPENAI_API_KEY before running this script.${NC}"
    exit 1
fi

echo -e "\nEnter your project's directory:"

read PROJECT_DIRECTORY

PYTHON_SCRIPT_PATH="feature_transcribe/prepare_code.py"

echo -e "\nRunning Python script to sync the project..."
python3 $PYTHON_SCRIPT_PATH --directory "$PROJECT_DIRECTORY" --api_key $API_KEY &
spinner $!

echo -e "${GREEN}Script execution completed. Now you can run ${YELLOW}sh run.sh ${GREEN} ${NC}"


