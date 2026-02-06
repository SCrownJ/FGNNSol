#!/bin/bash

PYTHON_CMD="python"

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python environment not found. Please check if Python is installed or modify the PYTHON_CMD variable in the script."
    exit 1
fi

echo "===== Starting to run construction_features.py ====="
$PYTHON_CMD construction_features.py

if [ $? -ne 0 ]; then
    echo "Error: construction_features.py execution failed!"
    exit 1
fi

echo "===== Starting to run add_feature1.py ====="
$PYTHON_CMD add_feature1.py

if [ $? -ne 0 ]; then
    echo "Error: add_feature1.py execution failed!"
    exit 1
fi

echo "===== Starting to run csv_to_npy.py ====="
$PYTHON_CMD csv_to_npy.py

if [ $? -ne 0 ]; then
    echo "Error: csv_to_npy.py execution failed!"
    exit 1
fi

echo "===== All feature processing steps completed successfully! ====="
exit 0
