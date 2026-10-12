#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .stage0-sources/cubecl/.git ]]; then
    bash stage0/setup.sh
fi
python - <<'PY'
from pathlib import Path
import subprocess
root=Path('.stage0-sources/cubecl')
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='6bba5fdf04206d9c3710dfbc46aa9850450ab405'
patch=Path('stage2/cubecl-ieee.patch').read_bytes()
diff=subprocess.check_output(['git','-C',str(root),'diff'])
if not diff:
    subprocess.run(['git','-C',str(root),'apply','../../stage2/cubecl-ieee.patch'],check=True)
elif diff!=patch:
    raise SystemExit('CubeCL checkout differs from the exact Stage 2 IEEE patch; preserve and inspect local work')
assert subprocess.check_output(['git','-C',str(root),'diff'])==patch
PY
cargo build --manifest-path stage0/Cargo.toml --locked --features llvm,hip-probe
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
