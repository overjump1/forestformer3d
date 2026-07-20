"""Pre-processing helpers for the treeiso pipeline.

Implements the three pre-processing steps listed in the artemis_treeiso README,
in pure NumPy/SciPy so no external point-cloud tools (e.g. CloudCompare) are
required:

* ground removal      -- drop points already classified as ground,
* noise removal       -- statistical outlier removal (SOR), NN=10 / std=1,
* decimation          -- voxel down-sampling to ~2 cm.

All functions operate on plain arrays so they can be reused independently of
laspy.
"""

import numpy as np
from scipy.spatial import cKDTree

# ASPRS standard classification code for ground.
GROUND_CLASS = 2


def ground_mask(classification):
    """Boolean mask selecting non-ground (vegetation) points.

    The input LAS is expected to already have its ground classified (ASPRS
    class 2); treeiso requires ground to be removed before isolation.
    """
    classification = np.asarray(classification)
    return classification != GROUND_CLASS


def statistical_outlier_removal(xyz, k=10, std_ratio=1.0):
    """Statistical Outlier Removal, matching CloudCompare's SOR filter.

    For every point the mean distance to its ``k`` nearest neighbours is
    computed. Points whose mean distance exceeds ``global_mean + std_ratio *
    global_std`` are considered noise and rejected. Defaults (k=10, std=1)
    mirror the README recommendation.

    Args:
        xyz: (N, 3) point coordinates.
        k: number of nearest neighbours (NN).
        std_ratio: standard-deviation multiplier.

    Returns:
        Boolean mask of length N; True marks kept (inlier) points.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n <= k + 1:
        # Too few points to estimate neighbourhood statistics -- keep them all.
        return np.ones(n, dtype=bool)

    tree = cKDTree(xyz)
    # k + 1 because the first neighbour returned is the point itself.
    dist, _ = tree.query(xyz, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)

    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= threshold


def voxel_downsample(xyz, voxel=0.02):
    """Voxel-grid decimation keeping one representative point per voxel.

    Returns the indices (into the original array) of the kept representative
    points, so that labels computed on the decimated cloud can be mapped back
    onto the original points later. This mirrors treeiso's own ``decimate_pcd``.

    Args:
        xyz: (N, 3) point coordinates.
        voxel: voxel edge length in the same units as ``xyz`` (metres).

    Returns:
        Integer array of representative indices into the original array.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int64)
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx.sort()
    return keep_idx
