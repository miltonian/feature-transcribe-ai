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

echo -e "${GREEN}Starting the Feature Transcription Script...${NC}"

API_KEY=$OPENAI_API_KEY
if [ -z "$API_KEY" ]; then
    echo -e "${RED}OPENAI_API_KEY is not set. Please export the OPENAI_API_KEY before running this script.${NC}"
    exit 1
fi

MODEL="gpt-4-turbo-preview"
echo -e "${YELLOW}Model selected: $MODEL${NC}"

# Initialize the first_run flag
first_run=true

# Adding a loop to continuously accept input
while true; do
    # Check if it's the first run
    if [ "$first_run" = true ]; then
        echo -e "\nEnter the feature description (type 'exit' to quit):"
        first_run=false  # Set first_run to false after the first input
    else
        echo -e "\nContinue typing... (type 'exit' to quit | type 'new' to reset the context):"
    fi

    read FEATURE_DESCRIPTION

    # Check if user wants to exit the script
    if [[ "$FEATURE_DESCRIPTION" == "exit" ]]; then
        echo "Exiting script."
        break
    fi

    # Check if user wants to exit the script
    if [[ "$FEATURE_DESCRIPTION" == "new" ]]; then
        echo "Started new conversation."
        python3 "feature_transcribe/start_new_conversation.py"
    else
        PYTHON_SCRIPT_PATH="feature_transcribe/feature_to_code.py"

        echo -e "\nProcessing the feature description..."
        python3 $PYTHON_SCRIPT_PATH --feature "$FEATURE_DESCRIPTION" --api_key $API_KEY --model $MODEL &
        spinner $!

        echo -e "${GREEN}Execution completed.${NC}"
    fi

    
done
