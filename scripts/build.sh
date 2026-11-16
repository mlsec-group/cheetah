#!/bin/bash
#SBATCH --job-name=build-cheetah
#SBATCH --partition=cpu-5h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=data/logs/build-cheetah/%j.out
#SBATCH --error=data/logs/build-cheetah/%j.err

# Builds the container images and downloads the external inputs. Needs
# internet access. Every step is skipped if its result already exists.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh

mkdir -p "${apptainer_cachedir}"
export APPTAINER_CACHEDIR="${apptainer_cachedir}"
export APPTAINER_TMPDIR="${apptainer_cachedir}"

BIND_OPTS=()
if [ "${slurm}" -eq 1 ]; then
  BIND_OPTS+=(-B /dev/shm:/tmp)
fi

# Container images
for image in llm metrics cpgtools gnn; do
  if [ -f "${sif_dir}/${image}.sif" ]; then
    echo "[+] ${image}.sif already exists. Skipping build."
  else
    echo "[+] Building ${image}.sif"
    apptainer build --fakeroot "${BIND_OPTS[@]}" "${sif_dir}/${image}.sif" "${image}.def"
  fi
done

# PrimeVul
raw_dir="${data_dir}/${dataset}/raw"
mkdir -p "${raw_dir}"
download() {
  if [ ! -f "$2" ]; then
    echo "[+] Downloading $(basename "$2")"
    wget "$1" -O "$2.part"
    mv "$2.part" "$2"
  fi
}
download 'https://drive.usercontent.google.com/download?id=12b1QkCwW0SC6l9KvxSmMe4jHF7VhjwCa&confirm=t' "${raw_dir}/primevul_train.jsonl"
download 'https://drive.usercontent.google.com/download?id=1490USYtUtb5n3i3m3n2LaSfjTCiKhPoO' "${raw_dir}/primevul_valid.jsonl"
download 'https://drive.usercontent.google.com/download?id=1ABV5cIdtyNAzKlGxjW_BsFZOp9MFd-AH' "${raw_dir}/primevul_test.jsonl"

# PDBERT checkpoint
if [ ! -d "${pdbert_dir}" ]; then
  echo "[+] Downloading PDBERT"
  tmp_dir="$(mktemp -d)"
  wget "https://zenodo.org/records/10140638/files/PDBERT_data.zip?download=1" -O "${tmp_dir}/PDBERT_data.zip"
  unzip -q "${tmp_dir}/PDBERT_data.zip" -d "${tmp_dir}"
  mv "${tmp_dir}/PDBERT_data/data/models/pdbert-base" "${pdbert_dir}"
  rm -rf "${tmp_dir}"
fi

# SAST tools (semgrep, rats, cppcheck): git submodule ./sast.
if [ ! -f sast/apptainer_build.sh ]; then
  echo "[!] ./sast is empty. Run: git submodule update --init" >&2
  exit 1
fi
for tool in semgrep rats cppcheck; do
  if [ -f "sast/${tool}/${tool}.sif" ]; then
    echo "[+] ${tool}.sif already exists. Skipping build."
  else
    APPTAINER_BUILD_OPTS="--fakeroot ${BIND_OPTS[*]}" ./sast/apptainer_build.sh "${tool}"
  fi
done
