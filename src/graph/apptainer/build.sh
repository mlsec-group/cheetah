#!/bin/bash

image_id=$(docker images docker-gnn-baselines:latest -q --no-trunc)

TAR_TMP_FILE=/tmp/docker-gnn-baselines-$(id -u).tar
echo "[+] Saving $image_id to tar"
docker save $image_id  -o $TAR_TMP_FILE
echo "[+] Building Apptainer container"
apptainer build apptainer/gnn-baselines.sif docker-archive:$TAR_TMP_FILE
echo "[+] Cleanup of docker tar image"
rm -f $TAR_TMP_FILE
echo "[+] All done!"
