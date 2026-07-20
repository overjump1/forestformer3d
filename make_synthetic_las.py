"""Generate a small synthetic LAS for testing the treeiso pipeline.

Produces a flat ground plane (ASPRS class 2) plus a few conical "trees" whose
points are split into low/medium/high vegetation (classes 3/4/5) by height,
plus a handful of scattered noise points. Useful as an end-to-end smoke test
when no real LAS is available.
"""

import sys

import numpy as np
import laspy


def _tree(cx, cy, height=8.0, radius=1.5, n=4000, rng=None):
    """A rough cone of points centred at (cx, cy) rising from z=0."""
    rng = rng or np.random.default_rng()
    z = rng.uniform(0, height, n)
    r = radius * (1 - z / height) * np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return np.column_stack([x, y, z])


def main(out_path="synth.las", seed=0):
    rng = np.random.default_rng(seed)

    # Flat ground plane, class 2.
    gx, gy = np.meshgrid(np.linspace(0, 20, 120), np.linspace(0, 20, 120))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    # Three well-separated trees.
    centres = [(5, 5), (14, 6), (9, 14)]
    trees = [_tree(cx, cy, rng=rng) for cx, cy in centres]
    veg = np.vstack(trees)

    # Scattered noise well above / away from everything.
    noise = np.column_stack([
        rng.uniform(0, 20, 40),
        rng.uniform(0, 20, 40),
        rng.uniform(12, 18, 40),
    ])

    pts = np.vstack([ground, veg, noise])

    # Classification: ground=2; vegetation split by height 3/4/5; noise=1.
    cls = np.empty(len(pts), dtype=np.uint8)
    cls[:len(ground)] = 2
    z = pts[:, 2]
    veg_cls = np.where(z < 2, 3, np.where(z < 5, 4, 5)).astype(np.uint8)
    cls[len(ground):len(ground) + len(veg)] = veg_cls[len(ground):len(ground) + len(veg)]
    cls[-len(noise):] = 1

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = pts.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.classification = cls
    las.write(out_path)
    print(f"wrote {out_path}: {len(pts)} points "
          f"({len(ground)} ground, {len(veg)} veg, {len(noise)} noise)")


if __name__ == "__main__":
    main(*sys.argv[1:])
