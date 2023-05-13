#!/usr/bin/env bash

# Load env vars
export $(grep -v '^#' .env | xargs)

# Remove previous builds
# if [ -d "governatooorr" ]; then
#     echo $PASSWORD | sudo -S sudo rm -Rf governatooorr;
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
cp /home/david/Valory/env/keys.json ./keys.json

autonomy deploy build -ltm

# Run the deployment
autonomy deploy run --build-dir abci_build/