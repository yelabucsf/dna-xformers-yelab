#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Function to display usage
usage() {
    echo "Usage: $0 <to_s3|from_s3> <data|models> [subdirectory]"
    echo "Example: $0 to_s3 data results/analysis"
    exit 1
}

# Check if correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
fi

DIRECTION=$1
MAIN_DIR=$2
SUBDIR=${3:-}

# Run S3 sync script
if [ -z "$SUBDIR" ]; then
    python scripts/s3_sync.py "$DIRECTION" --dir "$MAIN_DIR"
else
    python scripts/s3_sync.py "$DIRECTION" --dir "$MAIN_DIR" --subdir "$SUBDIR"
fi

# Deactivate virtual environment
deactivate