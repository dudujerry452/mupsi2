#!/usr/bin/env python3
"""Pixel-exact diff for PPM (P6 binary) images.

Usage: ppmdiff.py a.ppm b.ppm

Prints per-channel max abs diff and mean abs diff on the 0..255 scale, plus a
PASS/FAIL against a tolerance (default 2/255). Exit code 0 if within tolerance.

Used by TEST.md §7.2: for math-preserving optimizations, the same scene on the
same backend must be bit-identical (max diff == 0) or within 2/255 of floating
point reordering. Never used to compare CPU vs CUDA (known FMA/expf divergence).
"""
import sys


def read_ppm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise SystemExit(f"{path}: not a P6 binary PPM (got {magic!r})")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        maxval = int(f.readline().strip())
        if maxval != 255:
            raise SystemExit(f"{path}: maxval {maxval} != 255 unsupported")
        data = f.read(w * h * 3)
        if len(data) != w * h * 3:
            raise SystemExit(f"{path}: truncated ({len(data)} bytes, want {w*h*3})")
        return w, h, data


def main():
    if len(sys.argv) != 3:
        print("usage: ppmdiff.py a.ppm b.ppm", file=sys.stderr)
        return 2

    wa, ha, da = read_ppm(sys.argv[1])
    wb, hb, db = read_ppm(sys.argv[2])
    if (wa, ha) != (wb, hb):
        print(f"size mismatch: {wa}x{ha} vs {wb}x{hb}")
        return 1

    n = wa * ha * 3
    maxd = 0
    sumd = 0
    ndiff = 0
    for i in range(n):
        d = abs(da[i] - db[i])
        if d:
            ndiff += 1
            if d > maxd:
                maxd = d
        sumd += d

    tol = 2  # /255
    ok = maxd <= tol
    print(f"{wa}x{ha}  max_abs_diff={maxd}  mean_abs_diff={sumd / n:.3f}  "
          f"differing_bytes={ndiff}/{n}  {'OK (<=%d/255)' % tol if ok else 'FAIL (>%d/255)' % tol}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
