#!/usr/bin/env bash
set -eu
task_root=$(git rev-parse --show-toplevel)
g++ -std=c++17 -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1 \
  "$task_root/stage5/reduction_state_sampling.cpp" \
  -I "$task_root/stage0/python-deps/nvidia/cu13/include" \
  -I "$task_root/stage0/python-deps/torch/include" \
  -L "$task_root/stage0/python-deps/nvidia/cu13/lib" \
  -L "$task_root/stage0/python-deps/torch/lib" \
  -Wl,-rpath,"$task_root/stage0/python-deps/nvidia/cu13/lib" \
  -Wl,-rpath,"$task_root/stage0/python-deps/torch/lib" \
  -l:libcupti.so.13 -lc10 -o /tmp/stage5-state-sampling.so
