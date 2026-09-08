#!/bin/sh
# Install a native release, or a verified archive supplied locally.
set -eu

napkin_fail() { printf 'llm-napkin: %s\n' "$*" >&2; exit 1; }
napkin_version=latest
napkin_bin_dir=${HOME:?HOME must be set}/.local/bin
napkin_archive_dir=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version|--bin-dir|--archive-dir)
            [ "$#" -ge 2 ] || napkin_fail "missing value for $1"
            case "$1" in
                --version) napkin_version=$2 ;;
                --bin-dir) napkin_bin_dir=$2 ;;
                --archive-dir) napkin_archive_dir=$2 ;;
            esac
            shift 2 ;;
        -h|--help)
            printf '%s\n' 'Usage: sh install.sh [--version v0.1.0] [--bin-dir DIR] [--archive-dir DIR]' \
                'Defaults: latest release, ~/.local/bin. No administrator access needed.' \
                '--archive-dir reads a local release archive and SHA256SUMS instead of downloading.'
            exit 0 ;;
        *) napkin_fail "unknown argument: $1 (use --help)" ;;
    esac
done
[ -n "$napkin_bin_dir" ] || napkin_fail 'installation directory must not be empty'
case "$napkin_version" in
    latest) ;;
    *[!a-zA-Z0-9.+-]*|'') napkin_fail 'invalid release version' ;;
    v[0-9]*) ;;
    [0-9]*) napkin_version=v$napkin_version ;;
    *) napkin_fail 'version must be a release tag such as v0.1.0' ;;
esac
case "$(uname -s)/$(uname -m)" in
    Linux/x86_64|Linux/amd64) napkin_target=x86_64-unknown-linux-musl ;;
    Linux/aarch64|Linux/arm64) napkin_target=aarch64-unknown-linux-musl ;;
    Darwin/x86_64) napkin_target=x86_64-apple-darwin ;;
    Darwin/arm64|Darwin/aarch64) napkin_target=aarch64-apple-darwin ;;
    *) napkin_fail 'unsupported platform; use a Windows release ZIP or install with Cargo' ;;
esac
napkin_archive=llm-napkin-$napkin_target.tar.gz
if command -v sha256sum >/dev/null 2>&1; then
    napkin_hash_tool=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    napkin_hash_tool=shasum
else
    napkin_fail 'sha256sum or shasum is required to verify the download'
fi
command -v tar >/dev/null 2>&1 || napkin_fail 'tar is required'
napkin_tmp=$(mktemp -d)
napkin_pending=
napkin_cleanup() {
    rm -rf "$napkin_tmp"
    if [ -n "$napkin_pending" ]; then rm -f "$napkin_pending"; fi
}
trap napkin_cleanup EXIT
trap 'exit 1' HUP INT TERM
if [ -n "$napkin_archive_dir" ]; then
    cp "$napkin_archive_dir/$napkin_archive" "$napkin_tmp/$napkin_archive"
    cp "$napkin_archive_dir/SHA256SUMS" "$napkin_tmp/SHA256SUMS"
else
    command -v curl >/dev/null 2>&1 || napkin_fail 'curl is required'
    napkin_base=https://github.com/pranavthombare/llm-napkin/releases
    if [ "$napkin_version" = latest ]; then
        napkin_base=$napkin_base/latest/download
    else
        napkin_base=$napkin_base/download/$napkin_version
    fi
    for napkin_file in "$napkin_archive" SHA256SUMS; do
        curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --connect-timeout 15 --max-time 120 --retry 3 \
            "$napkin_base/$napkin_file" -o "$napkin_tmp/$napkin_file" || \
            napkin_fail "could not download $napkin_file; choose a native CLI release (v0.1.0+) or install with Cargo"
    done
fi
napkin_expected=$(awk -v file="$napkin_archive" '$2 == file {print $1}' "$napkin_tmp/SHA256SUMS")
case "$napkin_expected" in
    *[!a-fA-F0-9]*|'') napkin_fail 'missing or invalid archive checksum' ;;
esac
[ "${#napkin_expected}" -eq 64 ] || napkin_fail 'missing or duplicate archive checksum'
if [ "$napkin_hash_tool" = sha256sum ]; then
    napkin_actual=$(sha256sum "$napkin_tmp/$napkin_archive" | awk '{print $1}')
else
    napkin_actual=$(shasum -a 256 "$napkin_tmp/$napkin_archive" | awk '{print $1}')
fi
[ "$napkin_actual" = "$napkin_expected" ] || napkin_fail 'checksum mismatch; installation aborted'
tar -xzf "$napkin_tmp/$napkin_archive" -C "$napkin_tmp" llm-napkin
[ -f "$napkin_tmp/llm-napkin" ] && [ ! -L "$napkin_tmp/llm-napkin" ] || napkin_fail 'archive has no regular llm-napkin executable'
chmod 755 "$napkin_tmp/llm-napkin"
napkin_installed_version=$("$napkin_tmp/llm-napkin" --version) || napkin_fail 'the downloaded executable cannot run on this system'
case "$napkin_installed_version" in
    'llm-napkin '*) ;;
    *) napkin_fail 'archive contains an unexpected executable' ;;
esac
if [ "$napkin_version" != latest ]; then
    [ "$napkin_installed_version" = "llm-napkin ${napkin_version#v}" ] || napkin_fail 'archive version does not match the requested release'
fi
mkdir -p "$napkin_bin_dir"
napkin_pending=$(mktemp "$napkin_bin_dir/.llm-napkin.XXXXXX")
cp "$napkin_tmp/llm-napkin" "$napkin_pending"
chmod 755 "$napkin_pending"
mv -f "$napkin_pending" "$napkin_bin_dir/llm-napkin"
napkin_pending=
printf 'Installed %s to %s/llm-napkin\n' "$napkin_installed_version" "$napkin_bin_dir"
case ":$PATH:" in
    *":$napkin_bin_dir:"*) ;;
    *) printf 'Add %s to your PATH to use the llm-napkin command.\n' "$napkin_bin_dir" ;;
esac
