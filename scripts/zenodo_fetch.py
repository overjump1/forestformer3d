#!/usr/bin/env python3
"""Download files from a Zenodo record using only the Python standard library.

Queries https://zenodo.org/api/records/<id> for the file listing, downloads
files (optionally filtered by a regex), verifies their md5 checksums against
the values reported by the API, and can extract zip archives.

Run this on a machine WITH internet access (it is also used inside the
Docker image build to bake in the pretrained checkpoint).

Examples:
    # List files in the record without downloading
    python zenodo_fetch.py --record 16742708 --list

    # Download everything into ./zenodo_downloads
    python zenodo_fetch.py --record 16742708 --out zenodo_downloads

    # Download only files whose name matches a pattern, extracting zips
    python zenodo_fetch.py --record 16742708 --match '(?i)model|\\.pth$' \
        --out /tmp/ckpt --extract
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API_URL = "https://zenodo.org/api/records/{record}"
USER_AGENT = "forestformer3d-offline-fetch/1.0 (python-urllib)"
CHUNK = 1024 * 1024


def http_get(url, out_path=None, retries=4):
    last_err = None
    for attempt in range(retries):
        if attempt:
            wait = 2 ** attempt
            print(f"  retry {attempt}/{retries - 1} in {wait}s ({last_err})", flush=True)
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                if out_path is None:
                    return resp.read()
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(resp, f, CHUNK)
                return None
        except Exception as e:  # noqa: BLE001 - retry on any network error
            last_err = e
    raise SystemExit(f"ERROR: failed to fetch {url}: {last_err}")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def get_record_files(record):
    data = json.loads(http_get(API_URL.format(record=record)))
    files = []
    for f in data.get("files", []):
        name = f.get("key") or f.get("filename")
        checksum = (f.get("checksum") or "").removeprefix("md5:")
        link = (f.get("links") or {}).get("self") or (
            "https://zenodo.org/records/%s/files/%s?download=1"
            % (record, urllib.parse.quote(name))
        )
        files.append({"name": name, "size": f.get("size", 0),
                      "md5": checksum, "url": link})
    if not files:
        raise SystemExit(
            f"ERROR: Zenodo record {record} returned no files "
            "(restricted record or API change?)")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default="16742708", help="Zenodo record id")
    ap.add_argument("--out", default="zenodo_downloads", help="output directory")
    ap.add_argument("--match", default=None,
                    help="only download files whose name matches this regex")
    ap.add_argument("--extract", action="store_true",
                    help="extract downloaded .zip files into <out>/extracted/")
    ap.add_argument("--list", action="store_true",
                    help="list the record's files and exit")
    args = ap.parse_args()

    files = get_record_files(args.record)
    if args.match:
        pattern = re.compile(args.match)
        files = [f for f in files if pattern.search(f["name"])]
        if not files:
            raise SystemExit(
                f"ERROR: no file in record {args.record} matches {args.match!r}. "
                "Run with --list to see available files.")

    if args.list:
        for f in files:
            print(f"{f['name']}\t{f['size']} bytes\tmd5:{f['md5']}\t{f['url']}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = out / f["name"]
        if dest.exists() and f["md5"] and md5sum(dest) == f["md5"]:
            print(f"[skip] {f['name']} already downloaded and verified")
            continue
        print(f"[get ] {f['name']} ({f['size'] / 1e6:.1f} MB)", flush=True)
        part = dest.with_suffix(dest.suffix + ".part")
        http_get(f["url"], part)
        if f["md5"]:
            got = md5sum(part)
            if got != f["md5"]:
                part.unlink(missing_ok=True)
                raise SystemExit(
                    f"ERROR: md5 mismatch for {f['name']}: "
                    f"expected {f['md5']}, got {got}")
        part.rename(dest)
        print(f"[ ok ] {f['name']}")

    if args.extract:
        extract_dir = out / "extracted"
        extract_dir.mkdir(exist_ok=True)
        for f in files:
            dest = out / f["name"]
            if dest.suffix.lower() == ".zip" and zipfile.is_zipfile(dest):
                print(f"[unzip] {f['name']} -> {extract_dir}")
                with zipfile.ZipFile(dest) as z:
                    z.extractall(extract_dir)


if __name__ == "__main__":
    sys.exit(main())
