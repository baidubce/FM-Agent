#!/usr/bin/env python3
"""Restore and verify the separately distributed FDE-Bench case inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return (
        bool(entry.get("sha256"))
        and path.is_file()
        and path.stat().st_size == entry.get("bytes")
        and digest(path) == entry["sha256"]
    )


def safe_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Unsafe release asset name: {name!r}")
    return name


def download(entry: dict, base_url: str, cache: Path) -> Path:
    name = safe_name(entry["name"])
    if not entry.get("sha256") or not isinstance(entry.get("bytes"), int):
        raise ValueError(f"Release checksums are not available for {name} yet.")
    target = cache / name
    if verified(target, entry):
        print(f"Using verified cache: {name}", flush=True)
        return target
    temporary = target.with_name(target.name + ".part")
    print(f"Downloading {name} ({entry['bytes'] / 1024**2:.1f} MiB)", flush=True)
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/" + name,
                                         headers={"User-Agent": "FDE-Bench-data-installer"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, CHUNK)
        if not verified(temporary, entry):
            raise ValueError(f"Downloaded size or SHA-256 mismatch: {name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def obtain_archive(entry: dict, base_url: str, cache: Path, local: dict[str, Path]) -> Path:
    name = safe_name(entry["name"])
    if name in local:
        archive = local[name]
        if not verified(archive, entry):
            raise ValueError(f"Local archive size or SHA-256 mismatch: {archive}")
        return archive
    destination = cache / name
    if verified(destination, entry):
        return destination
    parts = entry.get("parts", [])
    if not parts:
        return download(entry, base_url, cache)
    if not entry.get("sha256") or not isinstance(entry.get("bytes"), int):
        raise ValueError(f"Complete archive checksum is missing for {name}.")
    paths = []
    for part in parts:
        part_name = safe_name(part["name"])
        if part_name in local:
            path = local[part_name]
            if not verified(path, part):
                raise ValueError(f"Local part size or SHA-256 mismatch: {path}")
        else:
            path = download(part, base_url, cache)
        paths.append(path)
    temporary = destination.with_name(destination.name + ".part")
    print(f"Joining {len(paths)} verified parts into {name}", flush=True)
    try:
        with temporary.open("wb") as output:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, CHUNK)
        if not verified(temporary, entry):
            raise ValueError(f"Combined archive size or SHA-256 mismatch: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    # Preserve manually supplied files; remove downloaded parts once the verified
    # complete archive exists, to avoid retaining a duplicate multi-GB copy.
    for path in paths:
        if path.parent == cache and path.name not in local:
            path.unlink()
    return destination


def extract(archive: Path, asset: dict, files: list[dict], root: Path) -> None:
    expected = {entry["path"]: entry for entry in files if entry["asset"] == asset["name"]}
    seen: set[str] = set()
    root = root.resolve()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            name = member.name.removeprefix("./")
            parts = PurePosixPath(name).parts
            if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe archive path: {name!r}")
            if member.isdir():
                continue
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError(f"Unexpected or duplicate archive entry: {name}")
            record = expected[name]
            if member.size != record["bytes"]:
                raise ValueError(f"Unexpected file size for {name}")
            destination = root.joinpath(*parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Archive path escapes repository: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            try:
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive file: {name}")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
                if not verified(temporary, record):
                    raise ValueError(f"Restored file checksum mismatch: {name}")
                temporary.chmod(0o644)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            seen.add(name)
    missing = sorted(expected.keys() - seen)
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} expected files; first: {missing[0]}")
    print(f"Restored and verified {len(seen)} files from {asset['name']}", flush=True)


def check(files: list[dict], root: Path) -> bool:
    bad = [entry["path"] for entry in files if not verified(root / entry["path"], entry)]
    if bad:
        print(f"Missing or changed: {len(bad)} of {len(files)} external input files.")
        for path in bad[:10]:
            print(f"  {path}")
        return False
    print(f"All {len(files)} external input files passed size and SHA-256 checks.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify restored files without downloading.")
    parser.add_argument("--list", action="store_true", help="List archives.")
    parser.add_argument("--archive", action="append", type=Path, default=[],
                        help="Use a local archive or archive part; may be repeated.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "fde-bench-data")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data-manifest.json").read_text())
    for case in manifest.get("withheld", []):
        print(f"Withheld: {case['case']} — {case['reason']}")
    if args.list:
        for asset in manifest["assets"]:
            state = "ready" if asset.get("sha256") else "release checksums pending"
            print(f"{asset['name']}: {state}")
            for part in asset.get("parts", []):
                print(f"  part: {part['name']}")
        return
    if args.check:
        raise SystemExit(0 if check(manifest["files"], ROOT) else 1)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    local = {path.name: path.resolve() for path in args.archive}
    known = {asset["name"] for asset in manifest["assets"]}
    known.update(part["name"] for asset in manifest["assets"] for part in asset.get("parts", []))
    unknown = sorted(local.keys() - known)
    if unknown:
        parser.error("Unrecognized archive filenames: " + ", ".join(unknown))
    base = (f"https://github.com/{manifest['repository']}/releases/download/"
            f"{manifest['release_tag']}")
    for asset in manifest["assets"]:
        path = obtain_archive(asset, base, args.cache_dir.resolve(), local)
        extract(path, asset, manifest["files"], ROOT)
    print("Data restored. Case 040 inputs are included in the repository.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"Data installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
