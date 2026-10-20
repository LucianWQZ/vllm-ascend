#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory evidence packaging/receiver verification (no model imports)."""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def digest(stream):
    value, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        value.update(chunk)
        size += len(chunk)
    return size, value.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
        raise ValueError("unsafe ZIP path")
    return path


def pack(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("output must be outside input directory")
    members = []
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink not permitted")
            if path.is_file():
                name = path.relative_to(source).as_posix()
                if name in ("ARTIFACT_INDEX.json", "SHA256SUMS"):
                    raise ValueError("reserved package index filename")
                with path.open("rb") as stream:
                    size, sha = digest(stream)
                members.append(dict(path=name, bytes=size, sha256=sha))
                archive.write(path, name)
        archive.writestr("ARTIFACT_INDEX.json", json.dumps({"files": members}, indent=2) + "\n")
        archive.writestr("SHA256SUMS", "".join(f"{m['sha256']}  {m['path']}\n" for m in members))
    return verify(output)


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate ZIP members")
        for info in archive.infolist():
            safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("symlink ZIP member")
        if archive.getinfo("ARTIFACT_INDEX.json").file_size > 16 * 1024 * 1024:
            raise ValueError("oversize package index")
        index = json.loads(archive.read("ARTIFACT_INDEX.json"))
        expected = {m["path"] for m in index["files"]}
        if expected | {"ARTIFACT_INDEX.json", "SHA256SUMS"} != set(names) or len(expected) != len(index["files"]):
            raise ValueError("incomplete/duplicate index")
        for member in index["files"]:
            with archive.open(member["path"]) as stream:
                size, sha = digest(stream)  # zip reader checks CRC at EOF
            if size != member["bytes"] or sha != member["sha256"]:
                raise ValueError("member bytes/hash mismatch")
            if member["path"].endswith(".json") and size <= 16 * 1024 * 1024:
                json.loads(archive.read(member["path"]))
        expected_sums = "".join(f"{m['sha256']}  {m['path']}\n" for m in index["files"])
        if archive.read("SHA256SUMS").decode() != expected_sums:
            raise ValueError("checksum index mismatch")
    with Path(path).open("rb") as stream:
        size, sha = digest(stream)
    return dict(
        status="VERIFIED",
        path=str(path),
        bytes=size,
        sha256=sha,
        members=len(names),
        memory="streaming 4 MiB; JSON index <=16 MiB",
        device_verification="NOT_ASSESSED",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "verify"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "pack" and args.output is None:
        parser.error("pack requires --output outside source")
    print(json.dumps(pack(args.path, args.output) if args.command == "pack" else verify(args.path), indent=2))


if __name__ == "__main__":
    main()
