#!/bin/bash

# Set the root directory
ROOT_DIR="CSLR-VSL"

# Define an array of directories to create inside CSLR-VSL
directories=(
    "$ROOT_DIR/data"
    "$ROOT_DIR/data/external"
    "$ROOT_DIR/data/interim"
    "$ROOT_DIR/data/processed"
    "$ROOT_DIR/data/raw"
)

echo "Checking and creating folder structure inside $ROOT_DIR..."

# Loop through each directory and check if it exists
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "Folder '$dir' already exists."
    else
        mkdir -p "$dir"
        echo "Created folder: $dir"
    fi
done

echo "Folder structure setup completed!"
