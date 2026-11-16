# shellcheck shell=bash
# Source from the repository root:  . scripts/lib/run_in_sif.sh
#
# _cheetah_run [apptainer exec options] <image.sif> <command...>
#
# Runs a command in an image with the repository's src/ on PYTHONPATH. The
# current directory is mounted at the same path inside the container, so the
# repository is never installed into an image. DRY_RUN=1 prints the command.
_cheetah_run() {
  local opts=()
  while [ $# -gt 0 ] && [[ "$1" != *.sif ]]; do
    opts+=("$1")
    shift
  done
  if [ $# -lt 2 ]; then
    echo "[!] usage: _cheetah_run [apptainer options] <image.sif> <command...>" >&2
    return 2
  fi
  local sif="$1"
  shift
  local cmd=(apptainer exec "${opts[@]}" --env "PYTHONPATH=${PWD}/src" "${sif}" "$@")
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    echo "[dry-run] ${cmd[*]}" >&2
    return 0
  fi
  "${cmd[@]}"
}
