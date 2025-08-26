#!/bin/bash

# Define paths to the three Python scripts
TRAIN_SCRIPT="train_features.py"
TEST_SCRIPT="test_features.py"
VAL_SCRIPT="val_features.py"

# Check if a script file exists
check_script() {
    if [ ! -f "$1" ]; then
        echo "Error: Script file $1 not found!"
        exit 1
    fi
}

# Verify all required scripts exist
check_script "$TRAIN_SCRIPT"
check_script "$TEST_SCRIPT"
check_script "$VAL_SCRIPT"

# Function to run a script with status checks
run_script() {
    local script_name=$1
    local description=$2

    echo "Starting $description..."
    python "$script_name"
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "$description failed with exit code $exit_code!"
        exit $exit_code
    fi

    echo "$description completed successfully."
}

# Run all feature extraction scripts in sequence
run_script "$TRAIN_SCRIPT" "training set feature extraction"
run_script "$TEST_SCRIPT" "test set feature extraction"
run_script "$VAL_SCRIPT" "validation set feature extraction"

echo "All feature extraction scripts have been executed successfully."
exit 0