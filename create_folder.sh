#!/bin/bash

# Set the root directory to the current working directory
ROOT_DIR=$(pwd)

# Define an array of directories to create inside the current working directory
directories=(
    "data"
    "data/external"
    "data/interim"
    "data/processed"
    "data/raw"
)

echo "Checking and creating folder structure inside $ROOT_DIR..."

# Loop through each directory and check if it exists
for dir in "${directories[@]}"; do
    if [ -d "$ROOT_DIR/$dir" ]; then
        echo "Folder '$ROOT_DIR/$dir' already exists."
    else
        mkdir -p "$ROOT_DIR/$dir"
        echo "Created folder: $ROOT_DIR/$dir"
    fi
done

echo "Folder structure setup completed!"
