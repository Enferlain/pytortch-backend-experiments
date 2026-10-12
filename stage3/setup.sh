#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Preserve the exact established FP stack and existing compiler/toolchains.
bash stage2/setup.sh
