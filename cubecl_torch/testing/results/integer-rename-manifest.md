# Manifest changes from the integer op rename (integer-value-arithmetic 2.2)

Group 2.1 replaced the effects path's internal op spellings `iadd`, `imul`, `imin`, `imax`, `and`
and `or` with the shared `add`, `mul`, `minimum`, `maximum`, `bitwise_and` and `bitwise_or`
(int64/bool operands). The manifest check against the group-1 manifest (2026-10-25) reported 4
changed entries, 2 corpus cases × 2 policies, each in one region:

| Case | Policy | Region | Artifact | Renamed ops | Old descriptor digest reproduced | `cubecl.ir` | `kernel.cu` |
|---|---|---|---|---|---|---|---|
| `embedding_backward-float32-257x7` | baseline | 2 | `508823ccd153bc72` | 5 | yes | identical | identical |
| `embedding_backward-float32-257x7` | first_use | 2 | `508823ccd153bc72` | 5 | yes | identical | identical |
| `training-float32-5x9` | baseline | 7 | `3ebe034670173d0c` | 5 | yes | identical | identical |
| `training-float32-5x9` | first_use | 7 | `3ebe034670173d0c` | 5 | yes | identical | identical |

## Method

For every artifact of the changed cases whose semantic descriptor contains a renamed op:
1. The new spellings were mapped back to the old ones.
2. The result's canonical digest was compared with the region's descriptor digest in the old
   manifest. It matched in every case, so the spelling is the only descriptor change.
3. The back-translated packet was compiled with the group-1 compiler binary (commit `b9fdbce3`),
   and its `cubecl.ir` and `kernel.cu` were compared with the new artifact's. The entry-point name
   (`cubecl5_<packet digest>`, which changes with the descriptor) was replaced by a placeholder on
   both sides. Both files are identical.

`kernel.ptx` was not recompiled. NVRTC's PTX is a deterministic function of the CUDA source and
options, and those differ only in the entry-point name. The manifest's IR/CUDA/PTX digests change
only because they hash files that contain that name.

No other corpus case changed: the manifest check (`manifest_check check --jobs 6`) listed only these
4 entries. The manifest was re-recorded after this review, and a check against the new manifest PASSed.
