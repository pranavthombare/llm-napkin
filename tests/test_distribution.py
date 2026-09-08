"""Offline release-package and installer checks (no real network or home-directory writes)."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package", ROOT / "scripts/package.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)
VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="napkin distribution ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "installed bin"
        self.dist = self.root / "dist"
        self.binary = self.root / "llm-napkin"
        self.binary.write_text(f"#!/bin/sh\nprintf 'llm-napkin {VERSION}\\n'\n")
        self.binary.chmod(0o755)
        self.mock_bin = self.root / "tools"
        self.mock_bin.mkdir()
        self.platform("Linux", "x86_64")
        self.env = {**os.environ, "PATH": str(self.mock_bin) + os.pathsep + os.environ["PATH"]}

    def platform(self, system, machine):
        uname = self.mock_bin / "uname"
        uname.write_text(f'#!/bin/sh\ncase "$1" in -s) echo {system};; -m) echo {machine};; esac\n')
        uname.chmod(0o755)

    def archive(self, target="x86_64-unknown-linux-musl"):
        return packager.package(target, self.binary, self.dist)

    def install(self, *args, offline=True):
        command = ["sh", str(ROOT / "install.sh"), "--bin-dir", str(self.bin_dir)]
        if offline:
            command += ["--archive-dir", str(self.dist)]
        return subprocess.run(command + list(args), env=self.env, capture_output=True, text=True)

    def test_package_contains_executable_docs_and_matching_checksum(self):
        archive = self.archive()
        with tarfile.open(archive) as package:
            self.assertEqual(set(package.getnames()), {
                "llm-napkin", "README.md", "LICENSE", "docs/reference.md", "docs/distributing.md"
            })
            self.assertEqual(package.getmember("llm-napkin").mode, 0o755)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual((self.dist / "SHA256SUMS").read_text(), f"{checksum}  {archive.name}\n")

    def test_install_and_update_with_spaces_in_destination(self):
        self.archive()
        self.bin_dir.mkdir()
        (self.bin_dir / "llm-napkin").write_text("old binary")
        result = self.install("--version", VERSION)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.check_output([str(self.bin_dir / "llm-napkin"), "--version"], text=True)
        self.assertEqual(result.strip(), f"llm-napkin {VERSION}")
        self.assertFalse(list(self.bin_dir.glob(".llm-napkin.*")))

    def test_checksum_failure_leaves_existing_installation_intact(self):
        archive = self.archive()
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        self.bin_dir.mkdir()
        (self.bin_dir / "llm-napkin").write_text("keep this")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertEqual((self.bin_dir / "llm-napkin").read_text(), "keep this")

    def test_missing_duplicate_and_wrong_version_checksums_are_rejected(self):
        self.archive()
        sums = self.dist / "SHA256SUMS"
        original = sums.read_text()
        for text in ["", original + original]:
            with self.subTest(manifest=text):
                sums.write_text(text)
                self.assertNotEqual(self.install().returncode, 0)
        sums.write_text(original)
        result = self.install("--version", "v999.0.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version does not match", result.stderr)
        self.assertFalse(self.bin_dir.exists())

    def test_platform_selection_for_each_unix_archive(self):
        for system, machine, target in [
            ("Linux", "x86_64", "x86_64-unknown-linux-musl"),
            ("Linux", "aarch64", "aarch64-unknown-linux-musl"),
            ("Darwin", "x86_64", "x86_64-apple-darwin"),
            ("Darwin", "arm64", "aarch64-apple-darwin"),
        ]:
            with self.subTest(target=target):
                self.platform(system, machine)
                self.archive(target)
                result = self.install()
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_windows_zip_and_release_version_guard(self):
        archive = self.archive("x86_64-pc-windows-msvc")
        with zipfile.ZipFile(archive) as package:
            self.assertIn("llm-napkin.exe", package.namelist())
        with self.assertRaisesRegex(ValueError, "does not match"):
            packager.package("x86_64-unknown-linux-musl", self.binary, self.dist, "v999.0.0")

    def test_invalid_options_and_unsupported_platform_fail_without_installing(self):
        for args in [("--unknown",), ("--version",), ("--version", "../../bad"), ("--bin-dir", "")]:
            with self.subTest(args=args):
                self.assertNotEqual(self.install(*args).returncode, 0)
        self.platform("FreeBSD", "x86_64")
        self.assertIn("unsupported platform", self.install().stderr)
        self.assertFalse(self.bin_dir.exists())

    def test_symlink_binary_is_not_installed(self):
        archive = self.archive()
        with tarfile.open(archive, "w:gz") as package:
            info = tarfile.TarInfo("llm-napkin")
            info.type = tarfile.SYMTYPE
            info.linkname = "/bin/sh"
            package.addfile(info)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.dist / "SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.bin_dir.exists())

    def test_download_urls_and_checksum_verification_with_mock_curl(self):
        self.archive()
        curl = self.mock_bin / "curl"
        curl.write_text('''#!/bin/sh
set -eu
for arg in "$@"; do
    case "$arg" in https://*) url=$arg;; esac
done
while [ "$1" != -o ]; do shift; done
shift
printf '%s\\n' "$url" >> "$NAPKIN_TEST_LOG"
cp "$NAPKIN_TEST_DIST/${url##*/}" "$1"
''')
        curl.chmod(0o755)
        self.env["NAPKIN_TEST_DIST"] = str(self.dist)
        log = self.root / "urls.txt"
        self.env["NAPKIN_TEST_LOG"] = str(log)
        result = self.install("--version", VERSION, offline=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        urls = log.read_text().splitlines()
        self.assertEqual(len(urls), 2)
        self.assertTrue(all(f"/releases/download/v{VERSION}/" in url for url in urls))
        log.unlink()
        result = self.install(offline=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all("/releases/latest/download/" in url for url in log.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
