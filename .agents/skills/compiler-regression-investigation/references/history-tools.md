# Verified history interfaces
Verified 2026-10-13 using official Git documentation. Use an isolated checkout, preserve user work and verify actual executable/revisions/paths before running templates. GOOD, BAD, PATTERN, path/to/source and predicate.py are placeholders.

| Command/interface | Source | Use |
| --- | --- | --- |
| `git log -S 'PATTERN' -- path/to/source` | [git-log pickaxe -S](https://git-scm.com/docs/git-log) | Find changes in occurrence count; moves/macros can obscure origin |
| `git log -G 'PATTERN' -- path/to/source` | [git-log pickaxe -G](https://git-scm.com/docs/git-log) | Find added/removed lines matching regex; inspect patches |
| `git blame -L START,END -- path/to/source` | [git-blame -L](https://git-scm.com/docs/git-blame) | Find attributed line commits; blame is not proof of original intent |
| `git bisect start BAD GOOD` | [git-bisect start](https://git-scm.com/docs/git-bisect) | Establish known tested endpoints |
| `git bisect run python predicate.py` | [git-bisect run](https://git-scm.com/docs/git-bisect) | Run a locally created and validated predicate; not an upstream-provided script |
| `git bisect log` / `git bisect reset` | [git-bisect log/reset](https://git-scm.com/docs/git-bisect) | Save sequence and restore starting state |

For bisect run, 0 means good; 1–127 except 125 mean bad; 125 means untestable/skip; other exit statuses abort. Unexpected interpreter/setup failures must not accidentally become bad. Write the harness to distinguish expected failures, transient failures and unavailable builds. Performance overlap or flakiness means the predicate is not ready. Skipped revisions may leave a range; do not claim a unique culprit.

Use connected GitHub to inspect commit/PR/test diffs and discussion or to obtain revision-pinned source. If local history is shallow/unavailable, report that limit; do not pretend a full bisect was completed. Context7 supplies documentation orientation, not commit attribution.
