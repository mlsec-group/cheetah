# shellcheck shell=bash
# Source after scripts/config.sh:  . scripts/lib/submit.sh
#
# submit <script> [--after <jobid[:jobid...]>] [--after-any <jobid[:jobid...]>] [script args...]
#
# slurm=1: submits the script with sbatch and prints the job id. The job starts
#          once the --after jobs have succeeded and the --after-any jobs have
#          ended, whether they succeeded or not.
# slurm=0: runs the script in this shell, once per array task if the script
#          has an "#SBATCH --array" line, and prints nothing.
#
# CHEETAH_ARRAY=<range> overrides the array range of every array script
# (e.g. CHEETAH_ARRAY=0-0 for a smoke test). DRY_RUN=1 prints what would run.
submit() {
  local script="$1"
  shift
  local dependency=""
  while [ "${1:-}" = "--after" ] || [ "${1:-}" = "--after-any" ]; do
    if [ -n "$2" ]; then
      local kind=afterok
      [ "$1" = "--after-any" ] && kind=afterany
      dependency="${dependency:+${dependency},}${kind}:$2"
    fi
    shift 2
  done

  local array
  array="$(sed -n 's/^#SBATCH --array=//p' "${script}")"
  if [ -n "${array}" ] && [ -n "${CHEETAH_ARRAY:-}" ]; then
    array="${CHEETAH_ARRAY}"
  fi

  # Slurm opens the log files before the script starts, so create their directories now.
  sed -n 's/^#SBATCH --\(output\|error\)=//p' "${script}" | xargs -r -n1 dirname | xargs -r mkdir -p

  if [ "${slurm}" -eq 1 ]; then
    local args=(--parsable --kill-on-invalid-dep=yes)
    [ -n "${dependency}" ] && args+=("--dependency=${dependency}")
    [ -n "${array}" ] && args+=("--array=${array}")
    if [ "${DRY_RUN:-0}" -eq 1 ]; then
      echo "[dry-run] sbatch ${args[*]} ${script} $*" >&2
      basename "${script%.*}"
      return 0
    fi
    sbatch "${args[@]}" "${script}" "$@"
  elif [ -n "${array}" ]; then
    local first="${array%-*}" last="${array#*-}" task
    for task in $(seq "${first}" "${last}"); do
      echo "[+] ${script} (task ${task})" >&2
      SLURM_ARRAY_TASK_ID="${task}" bash "${script}" "$@" >&2
    done
  else
    echo "[+] ${script}" >&2
    bash "${script}" "$@" >&2
  fi
}

# join_ids <id...>: colon-joined job ids, skipping empty ones.
join_ids() {
  local out="" id
  for id in "$@"; do
    [ -n "${id}" ] && out="${out:+${out}:}${id}"
  done
  echo "${out}"
}
