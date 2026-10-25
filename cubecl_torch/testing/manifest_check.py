"""Structural manifest: record and check the package's generated code per corpus case.

The manifest holds, per (case, policy), the ordered region records the
equivalence harness builds (role, descriptor digest, chosen plan, IR/CUDA/PTX
digests of every compiled candidate, candidate sets, selections). A later
change that alters any of them is reported by case, region and file.

    python -m cubecl_torch.testing.manifest_check check [--names a,b] [--jobs 6]
    python -m cubecl_torch.testing.manifest_check record [--jobs 6]
    python -m cubecl_torch.testing.manifest_check record --from results/equivalence-full.json

``record`` without ``--from`` compiles the whole corpus with this package only.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
FILES = ("cubecl.ir", "kernel.cu", "kernel.ptx")


def entries(results):
    """``{"case|policy": regions}`` from package results."""
    return {
        f"{r['case']}|{r['policy']}": r["regions"]
        for r in results
        if r["impl"] == "package"
    }


def record(results, path=MANIFEST):
    failed = sorted(
        f"{r['case']}|{r['policy']}" for r in results if r.get("error") is not None
    )
    if failed:
        raise ValueError(f"refusing to record errored cases: {', '.join(failed)}")
    data = dict(sorted(entries(results).items()))
    path.write_text(json.dumps({"format": 1, "cases": data}, indent=1, sort_keys=True))
    return len(data)


def _artifact_changes(old, new):
    old = {c["plan"]: c for c in old or ()}
    new = {c["plan"]: c for c in new or ()}
    changes = [f"candidate {p} added" for p in sorted(set(new) - set(old))]
    changes += [f"candidate {p} removed" for p in sorted(set(old) - set(new))]
    for plan in sorted(set(old) & set(new)):
        changed = [f for f in FILES if old[plan][f] != new[plan][f]]
        if changed:
            changes.append(f"candidate {plan}: {', '.join(changed)}")
    return changes


def diff(manifest, current):
    """``{"case|policy": [change, ...]}`` for every case whose records changed."""
    report = {}
    for key in sorted(set(manifest) | set(current)):
        if key not in current:
            report[key] = ["case not run"]
            continue
        if key not in manifest:
            report[key] = ["case not in manifest"]
            continue
        old, new = manifest[key], current[key]
        changes = []
        if len(old) != len(new):
            changes.append(f"region count {len(old)} -> {len(new)}")
        for i, (a, b) in enumerate(zip(old, new, strict=False)):
            for field in sorted(set(a) | set(b)):
                if a.get(field) == b.get(field):
                    continue
                if field == "compiled":
                    changes += [
                        f"region {i} {c}" for c in _artifact_changes(a[field], b[field])
                    ]
                elif field == "artifact":
                    changed = [
                        f
                        for f in FILES
                        if (a[field] or {}).get(f) != (b[field] or {}).get(f)
                    ]
                    changes.append(
                        f"region {i} chosen artifact: {', '.join(changed) or 'plan'}"
                    )
                else:
                    changes.append(f"region {i} {field} changed")
        if changes:
            report[key] = changes
    return report


def check(names=None, policies=("baseline", "first_use"), jobs=6, work=None):
    from cubecl_torch.testing import equivalence

    manifest = json.loads(MANIFEST.read_text())["cases"]
    if names:
        manifest = {k: v for k, v in manifest.items() if k.split("|")[0] in names}
    names = sorted({k.split("|")[0] for k in manifest})
    if not names:
        raise ValueError("no manifest cases selected")
    # Without --work the equivalence runner owns, and removes, a temporary one.
    results = equivalence.run(
        names, list(policies), jobs=jobs, work=work, impls=("package",)
    )
    report = diff(manifest, entries(results))
    for r in results:
        if r["error"] is not None:
            report.setdefault(f"{r['case']}|{r['policy']}", []).append(
                f"error: {r['error']}"
            )
    return report


def compile_corpus(policies=("baseline", "first_use"), jobs=6, work=None):
    """Package results for every corpus case, as ``record`` takes them."""
    from cubecl_torch.testing import equivalence
    from stage5.corpus import cases

    names = [c.name for c in cases()]
    return equivalence.run(
        names, list(policies), jobs=jobs, work=work, impls=("package",)
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--from", dest="source")
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--work")
    c = sub.add_parser("check")
    c.add_argument("--names", default="")
    c.add_argument("--jobs", type=int, default=6)
    c.add_argument("--work")
    args = parser.parse_args(argv)
    if args.command == "record":
        if args.source:
            results = json.loads(Path(args.source).read_text())["results"]
        else:
            results = compile_corpus(jobs=args.jobs, work=args.work)
        print("recorded", record(results), "cases to", MANIFEST)
        return 0
    names = [n for n in args.names.split(",") if n]
    report = check(names, jobs=args.jobs, work=args.work)
    for key, changes in report.items():
        print(key)
        for change in changes:
            print("   ", change)
    print("manifest check:", "PASS" if not report else f"{len(report)} cases changed")
    return 0 if not report else 1


if __name__ == "__main__":
    sys.exit(main())
