#!/bin/bash

# Config
volume_dir=/work/$USER/gnn-baselines/data
volumes="primevul"

# Create docker volumes
echo "[+] Create docker volumes"
for volume in ${volumes}; do
    echo "[*] Create volume: ${volume}"
    docker volume create ${volume} --opt type=none --opt device=${volume_dir}/${volume} --opt o=bind
done

# Build images
echo "[+] Build images"
pushd docker/ > /dev/null
docker compose build
popd > /dev/null
