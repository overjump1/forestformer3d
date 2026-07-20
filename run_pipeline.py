"""End-to-end individual-tree isolation pipeline.

Given a single LAS/LAZ whose ground is already classified (ASPRS class 2) and
whose vegetation is already split into low/medium/high vegetation by height
(classes 3/4/5), this script:

  1. removes ground points,
  2. removes noise with a statistical-outlier filter (SOR, NN=10, std=1),
  3. decimates the cloud to ~2 cm,
  4. isolates individual trees with treeiso (pure-Python), then
  5. writes an output LAS where every point belonging to an isolated tree gets a
     single ``Tree`` classification code plus a per-tree ``treeID`` extra
     dimension. Ground and any non-isolated vegetation keep their original
     classification (treeID = 0).

Steps 1-3 are the pre-processing listed in the treeiso README.

Usage:
    python run_pipeline.py input.las [-o output.laz] [options]
"""

import argparse
import os

import numpy as np
import laspy

from preprocessing import ground_mask, statistical_outlier_removal, voxel_downsample
from reclassify import carry_back_labels
from treeiso import isolate_trees

# Default classification code assigned to isolated-tree points. Sits in the
# LAS 1.4 reserved/user range so it does not collide with ground (2) or the
# low/medium/high vegetation classes (3/4/5).
DEFAULT_TREE_CLASS = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input LAS/LAZ file (ground already classified)")
    p.add_argument("-o", "--output", default=None,
                   help="output path (default: <input>_classified.laz)")
    p.add_argument("--voxel", type=float, default=0.02,
                   help="decimation voxel size in metres (default: 0.02)")
    p.add_argument("--sor-k", type=int, default=10,
                   help="SOR neighbour count NN (default: 10)")
    p.add_argument("--sor-std", type=float, default=1.0,
                   help="SOR standard-deviation multiplier (default: 1.0)")
    p.add_argument("--tree-class", type=int, default=DEFAULT_TREE_CLASS,
                   help=f"classification code for tree points (default: {DEFAULT_TREE_CLASS})")
    p.add_argument("--carryback-radius", type=float, default=0.1,
                   help="max distance (m) to carry tree labels back to full-res "
                        "points (default: 0.1)")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="also write the pre-processed cloud (<input>_preprocessed.laz)")
    return p.parse_args()


def main():
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    out_path = args.output or f"{stem}_classified.laz"

    print(f"Reading {args.input} ...")
    las = laspy.read(args.input)
    xyz = np.column_stack([las.x, las.y, las.z])
    n_total = len(xyz)

    # --- 1. Ground removal -------------------------------------------------
    veg = ground_mask(las.classification)
    veg_idx = np.where(veg)[0]
    print(f"  {n_total} points; {len(veg_idx)} non-ground (vegetation) points")
    if len(veg_idx) == 0:
        raise SystemExit("No non-ground points found -- nothing to isolate.")

    # --- 2. Noise removal (SOR) -------------------------------------------
    inliers = statistical_outlier_removal(xyz[veg_idx], k=args.sor_k,
                                          std_ratio=args.sor_std)
    kept_idx = veg_idx[inliers]
    print(f"  SOR (NN={args.sor_k}, std={args.sor_std}): "
          f"removed {len(veg_idx) - len(kept_idx)} noise points")

    # --- 3. Decimation to ~voxel ------------------------------------------
    rep_local = voxel_downsample(xyz[kept_idx], voxel=args.voxel)
    rep_idx = kept_idx[rep_local]
    print(f"  Decimation ({args.voxel} m): {len(kept_idx)} -> {len(rep_idx)} points")

    if args.keep_intermediate:
        pp_path = f"{stem}_preprocessed.laz"
        pp = laspy.LasData(las.header)
        pp.points = las.points[rep_idx].copy()
        pp.write(pp_path)
        print(f"  Wrote pre-processed cloud -> {pp_path}")

    # --- 4. Tree isolation (treeiso) --------------------------------------
    print("  Running treeiso ...")
    final_labels, _, _ = isolate_trees(xyz[rep_idx])
    n_trees = len(np.unique(final_labels))
    print(f"  treeiso found {n_trees} tree(s)")

    # --- 5. Carry labels back to the full-resolution vegetation -----------
    # Only original vegetation points can become trees; ground is untouched.
    veg_labels = carry_back_labels(xyz[veg_idx], xyz[rep_idx], final_labels,
                                   radius=args.carryback_radius)

    tree_id = np.zeros(n_total, dtype=np.uint32)
    # treeID is 1-based so that 0 cleanly means "not part of any isolated tree".
    matched = veg_labels >= 0
    tree_id[veg_idx[matched]] = veg_labels[matched] + 1

    # --- Write output ------------------------------------------------------
    out = laspy.LasData(las.header)
    out.points = las.points.copy()
    out.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type="uint32",
        description="tree id (0 = not a tree)"))
    out.treeID = tree_id

    classification = np.asarray(out.classification).copy()
    is_tree = tree_id > 0
    classification[is_tree] = args.tree_class
    out.classification = classification

    out.write(out_path)
    print(f"Wrote {out_path}: {int(is_tree.sum())} tree points "
          f"reclassified to class {args.tree_class} across {n_trees} trees")


if __name__ == "__main__":
    main()
