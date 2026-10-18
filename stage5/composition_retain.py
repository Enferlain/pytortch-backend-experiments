"""Retain selected ordinary artifacts for reproducible offline composition audit."""

from pathlib import Path
import shutil

from stage5.composition_trial import OUT, read, sha, write


def main():
    timing = read(OUT / "timing.json")
    assert timing["complete"]
    paths = {Path(a["directory"]) for row in timing["rows"]
             for artifacts in row["artifacts"].values() for a in artifacts}
    paths.update(Path(a["directory"]) for row in read(OUT / "preflight.json")["records"]
                 for a in row["selected"])
    retained = []
    for source in sorted(paths):
        if source.is_relative_to(OUT):
            continue
        target = OUT / "ordinary" / source.name
        target.mkdir(parents=True, exist_ok=True)
        for original in sorted(source.iterdir()):
            if original.is_file():
                destination = target / original.name
                if destination.exists():
                    assert sha(destination) == sha(original), str(destination)
                else:
                    shutil.copyfile(original, destination)
        retained.append(dict(source=str(source), retained=str(target), files={
            p.name: sha(p) for p in target.iterdir() if p.is_file()}))
    write("retention.json", dict(complete=True, rows=retained, source_sha256=sha(__file__)))
    print("PASS", len(retained), "selected ordinary artifact directories retained/verified")


if __name__ == "__main__":
    main()
