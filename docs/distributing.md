# Distributing llm-napkin

The CLI ships as one native executable. End users do not need Rust or Python.
Install from the checkout with `cargo install --path . --locked`, or build and
share a release archive using the workflow below.

## Prepare a release

1. Update `version` in `Cargo.toml` and regenerate `Cargo.lock` with `cargo check`.
2. Update `CHANGELOG.md` and `docs/release-notes.md`.
3. Run the checks listed in the README, commit the changes and push the branch.
4. Optionally run the **Release** workflow manually in GitHub Actions. It tests
   and builds every target and uploads downloadable Actions artifacts; manual
   runs do not publish a release.
5. Tag the tested commit with the matching version and push the tag:

   ```sh
   git tag -a v0.1.0 -m "llm-napkin v0.1.0"
   git push origin v0.1.0
   ```

The tag triggers native builds on Linux x86_64/ARM64, macOS Intel/Apple Silicon,
and Windows x86_64. Each job runs the Rust tests on its target, builds the binary,
checks `--version`, and packages it. Unix jobs also install and run their archive.
Only after all jobs succeed does the workflow upload every archive, `SHA256SUMS`
and `install.sh` to a draft, then publish the completed GitHub Release. A
prerelease tag containing `-` is marked as a prerelease. Published assets are not
overwritten by reruns; a new version is required. No crates.io token is needed.

The initial native CLI version is **v0.1.0**. The v0.0.x tags/releases are from
the earlier VS Code extension and do not provide native CLI installers.

## Package locally

Python 3.11+ is required for the packaging helper. It packages an already-built
binary and checks that its reported version matches `Cargo.toml`.

On an x86_64 Debian/Ubuntu Linux build host:

```sh
sudo apt-get install musl-tools
rustup target add x86_64-unknown-linux-musl
cargo build --release --locked --target x86_64-unknown-linux-musl
python3 scripts/package.py --target x86_64-unknown-linux-musl
```

The helper writes the archive, a per-archive `.sha256` file and a combined
`SHA256SUMS` file under `dist/`. Linux release targets use static musl linking to
avoid requiring a specific host glibc version. Use a native build host for the
chosen target; the GitHub matrix supplies these hosts automatically.

Install the resulting Linux/macOS archive without a network request:

```sh
sh install.sh --archive-dir dist --bin-dir /tmp/napkin-demo
/tmp/napkin-demo/llm-napkin --version
/tmp/napkin-demo/llm-napkin HuggingFaceTB/SmolLM2-135M-Instruct -i 2048 -o 512 -b 4
```

Share the matching archive and `SHA256SUMS` with another machine of the same
OS/architecture. The installer verifies the checksum and checks that the binary
runs before replacing an existing installation. It does not change shell profiles.

## Verify a downloaded archive manually

Download the platform archive and `SHA256SUMS` from the same release. Compare
its hash with the matching line in the manifest:

```sh
# Linux
sha256sum llm-napkin-x86_64-unknown-linux-musl.tar.gz
# macOS
shasum -a 256 llm-napkin-aarch64-apple-darwin.tar.gz
```

```powershell
# Windows PowerShell
Get-FileHash .\llm-napkin-x86_64-pc-windows-msvc.zip -Algorithm SHA256
Expand-Archive .\llm-napkin-x86_64-pc-windows-msvc.zip -DestinationPath .\llm-napkin
.\llm-napkin\llm-napkin.exe --help
```

Checksums detect corrupt or mismatched downloads. Signing/notarization is not
configured for these archives. macOS may require allowing a downloaded app to
run through the system's Privacy & Security settings.
