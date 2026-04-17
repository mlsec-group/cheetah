#!/bin/bash
# Orchestration: run metric extraction then autosklearn jobs (see individual scripts).
set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh

echo "[+] Step 1/4: MCPP extraction (metrics.calculate)"
./scripts/metric_analysis/calculate.sh

echo "[+] Step 2/4: autosklearn train (array or loop — metrics/train.sh)"
./scripts/metric_analysis/train.sh

echo "[+] Step 3/4: optional isolation / leave-one-out"
echo "    ./scripts/metric_analysis/isolation.sh"
echo "    ./scripts/metric_analysis/leaveoneout.sh"

echo "[+] Step 4/4: aggregate autosklearn outputs"
./scripts/metric_analysis/aggregate.sh

echo "[+] metrics/run.sh pipeline finished."
