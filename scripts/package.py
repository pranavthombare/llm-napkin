#!/usr/bin/env python3
"""Package an already-built native executable; Python is only a release-time dependency."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)


def package(target, binary, out_dir, tag=None):
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    if tag and tag != f"v{version}":
        raise ValueError(f"tag {tag} does not match Cargo.toml version v{version}")
    if target not in TARGETS:
        raise ValueError(f"unsupported release target: {target}")
    binary = Path(binary).resolve()
    actual = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if actual != f"llm-napkin {version}":
        raise ValueError(f"binary version mismatch: {actual!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    executable = "llm-napkin.exe" if "windows" in target else "llm-napkin"
    entries = [(binary, executable)] + [(ROOT / name, name) for name in (
        "README.md", "LICENSE", "docs/reference.md", "docs/distributing.md"
    )]
    extension = "zip" if "windows" in target else "tar.gz"
    archive = out_dir / f"llm-napkin-{target}.{extension}"
    if extension == "zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for source, name in entries:
                output.write(source, name)
    else:
        with tarfile.open(archive, "w:gz") as output:
            for source, name in entries:
                info = output.gettarinfo(str(source), name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if name == executable else 0o644
                with source.open("rb") as content:
                    output.addfile(info, content)
    with archive.open("rb") as content:
        digest = hashlib.file_digest(content, "sha256").hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    (out_dir / "SHA256SUMS").write_text("".join(p.read_text() for p in sorted(out_dir.glob("*.sha256"))))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    filename = "llm-napkin.exe" if "windows" in args.target else "llm-napkin"
    binary = args.binary or ROOT / "target" / args.target / "release" / filename
    print(package(args.target, binary, args.out_dir, os.environ.get("RELEASE_TAG")))


if __name__ == "__main__":
    main()
