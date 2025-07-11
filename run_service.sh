#!/usr/bin/env bash

rm -r mech

# Load env vars
set -o allexport; source .1env; set +o allexport

# Remove previous builds
# if [ -d "mech" ]; then
#     echo $PASSWORD | sudo -S sudo rm -Rf mech;
# fi

# Push packages and fetch service
# make formatters
# make generators
make clean

autonomy push-all

autonomy fetch --local --service valory/mech && cd mech

# Build the image
autonomy build-image

# Copy keys and build the deployment
cp $PWD/../keys.json .

autonomy deploy build --n 1 -ltm

# Get the latest abci_build directory
BUILD_DIR=$(ls -d abci_build_* | sort -V | tail -n1)

# Run the deployment with the found directory
autonomy deploy run --build-dir $BUILD_DIR