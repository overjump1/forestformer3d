"""Map treeiso instance labels back onto the original point cloud.

treeiso runs on the decimated / denoised vegetation cloud. To produce a useful
output we propagate its per-tree labels back onto the full-resolution original
points, then encode the result as:

* a single ``Tree`` classification code for every isolated-tree point, and
* a per-tree ``treeID`` extra dimension (0 = not part of any isolated tree).

Non-tree points (ground, and any vegetation not absorbed into a tree) keep their
original classification.
"""

import numpy as np
from scipy.spatial import cKDTree


def carry_back_labels(orig_xyz, ref_xyz, ref_labels, radius):
    """Propagate ``ref_labels`` onto ``orig_xyz`` by nearest neighbour.

    Each original point takes the label of the closest reference (treeiso'd)
    point, provided that point lies within ``radius``. Original points with no
    reference neighbour inside ``radius`` (e.g. removed noise) get label -1.

    Args:
        orig_xyz: (N, 3) coordinates of the points to label.
        ref_xyz: (M, 3) coordinates of the labelled reference points.
        ref_labels: (M,) integer labels for the reference points.
        radius: maximum matching distance (metres).

    Returns:
        (N,) integer array of propagated labels (-1 where unmatched).
    """
    orig_xyz = np.asarray(orig_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    ref_labels = np.asarray(ref_labels)

    out = np.full(len(orig_xyz), -1, dtype=np.int64)
    if len(ref_xyz) == 0 or len(orig_xyz) == 0:
        return out

    tree = cKDTree(ref_xyz)
    dist, idx = tree.query(orig_xyz, k=1, distance_upper_bound=radius)
    matched = np.isfinite(dist)
    out[matched] = ref_labels[idx[matched]]
    return out
