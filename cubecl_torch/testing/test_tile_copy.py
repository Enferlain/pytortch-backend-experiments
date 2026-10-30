"""Our cubecl-tile copy stays traceable to upstream (cubecl-tile-stage-1 task 1.2)."""

import io
import re
import shutil
import subprocess
import tarfile

import pytest

from cubecl_torch.runtime import native
from cubecl_torch.testing.fixtures import REPO

COPY = REPO / "cubecl_torch" / "native" / "cubecl-tile"
CHECKOUT = REPO / ".stage0-sources" / "cubecl"
PREFIX = "crates/cubecl-tile/"


def recorded(copy):
    """The upstream revision and the files listed as changed in UPSTREAM.md."""
    text = (copy / "UPSTREAM.md").read_text()
    revision = re.search(
        r"^Upstream revision: ([0-9a-f]{40})$", text, re.MULTILINE
    ).group(1)
    section = text.split("## Local changes", 1)[1].split("\n## ", 1)[0]
    return revision, set(re.findall(r"^- `(src/[^`]+)`", section, re.MULTILINE))


def upstream_sources(revision):
    """``{relative path: bytes}`` of upstream's ``src`` at ``revision``."""
    archive = subprocess.run(
        ["git", "-C", str(CHECKOUT), "archive", revision, PREFIX + "src"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return {
            member.name.removeprefix(PREFIX): tar.extractfile(member).read()
            for member in tar.getmembers()
            if member.isfile()
        }


def record_mismatches(copy):
    """Files changed but not listed, or listed but not changed: a stale entry
    would otherwise hide a later edit to its file."""
    revision, listed = recorded(copy)
    upstream = upstream_sources(revision)
    ours = {
        path.relative_to(copy).as_posix(): path.read_bytes()
        for path in (copy / "src").rglob("*")
        if path.is_file()
    }
    changed = {
        name
        for name in upstream.keys() | ours.keys()
        if upstream.get(name) != ours.get(name)
    }
    return changed ^ listed


def checkout_present():
    return (CHECKOUT / ".git").exists()


skip_without_checkout = pytest.mark.skipif(
    not checkout_present(), reason="needs the CubeCL checkout"
)


def test_record_names_the_audited_pin():
    assert recorded(COPY)[0] == native.CUBECL_PIN


@skip_without_checkout
def test_copy_matches_its_record():
    assert record_mismatches(COPY) == set()


@skip_without_checkout
def test_unlisted_edit_and_stale_entry_are_caught(tmp_path):
    copy = tmp_path / "cubecl-tile"
    shutil.copytree(COPY, copy, ignore=shutil.ignore_patterns("target"))
    edited = copy / "src" / "lib.rs"
    original = edited.read_bytes()
    edited.write_bytes(original + b"\n// local edit\n")
    assert record_mismatches(copy) == {"src/lib.rs"}

    record = copy / "UPSTREAM.md"
    record.write_text(
        record.read_text().replace(
            "\n## Merging", "- `src/lib.rs`: a listed edit\n\n## Merging"
        )
    )
    assert record_mismatches(copy) == set()

    edited.write_bytes(original)
    assert record_mismatches(copy) == {"src/lib.rs"}
