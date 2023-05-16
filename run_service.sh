#!/usr/bin/env bash

# Load env vars
export $(grep -v '^#' .1env | xargs)

# Remove previous builds
# if [ -d "mech" ]; then
#     echo $PASSWORD | sudo -S sudo rm -Rf mech;
# fi

# Push packages and fetch service
# make formatters
# make generators
make clean

autonomy push-all

autonomy fetch --local --service eightballer/mech && cd mech

# Build the image
autonomy build-image

# Copy keys and build the deployment
cp /home/david/Valory/repos/mech/keys.json ./keys.json

autonomy deploy build -ltm

# Run the deployment
autonomy deploy run --build-dir abci_build/