#!/usr/bin/env python3
"""Decode a Colab 'ARCHIVE_B64 <crc32-hex> <base64(gzip(json))>' line into a JSON file.

Usage: python scripts/decode_archive.py <out.json> < line.txt
Verifies the CRC32 of the decompressed bytes; exits non-zero on mismatch (so a mistyped copy is detected).
"""
import base64
import gzip
import sys
import zlib

line = sys.stdin.read().strip().split()
if len(line) != 3 or line[0] != "ARCHIVE_B64":
    sys.exit("expected: ARCHIVE_B64 <crc32> <b64>")
raw = gzip.decompress(base64.b64decode(line[2]))
crc = format(zlib.crc32(raw) & 0xFFFFFFFF, "08x")
if crc != line[1]:
    sys.exit(f"CRC mismatch: expected {line[1]} got {crc}")
with open(sys.argv[1], "wb") as f:
    f.write(raw)
print(f"wrote {sys.argv[1]} ({len(raw)} bytes, crc ok)")
