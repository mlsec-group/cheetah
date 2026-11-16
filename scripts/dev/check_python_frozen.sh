#!/bin/bash
# Guard for functional integrity: fails if experiment code differs from the
# baseline commit, except for files listed in scripts/dev/approved_patches.txt
# (one path per line; every entry needs an approved item in
# docs/reproduction-report.md). Run from the repository root.
set -euo pipefail

baseline=7b9bac138d94d5ae9a62d334c669458a702b646d
# src/prompting/ is not part of the baseline: it was ported from the authors'
# prompting archive later and is frozen at the commit that added it.
prompting_baseline=a02492a916c357cf735086f9cf208efbdd9c27e1
allowlist=scripts/dev/approved_patches.txt

changed=$(
  {
    git diff --name-only "${baseline}" -- src ':!src/prompting'
    git ls-files --others --exclude-standard -- src ':!src/prompting'
    if [ -n "${prompting_baseline}" ]; then
      git diff --name-only "${prompting_baseline}" -- src/prompting
      git ls-files --others --exclude-standard -- src/prompting
    fi
  } | grep -E '\.(py|sc)$|^src/graph/cpgtools/joern_scripts/' \
    | grep -vxF -f <(grep -v '^#' "${allowlist}") || true
)

if [ -n "${changed}" ]; then
  echo "[!] Experiment code differs from baseline ${baseline:0:7} without approval:"
  echo "${changed}" | sed 's/^/    /'
  exit 1
fi
echo "[+] Experiment code matches baseline ${baseline:0:7} (approved patches excepted)."
