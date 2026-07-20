# Individual-tree isolation pipeline (treeiso)

Given a single LAS/LAZ whose **ground is already classified** (ASPRS class 2)
and whose vegetation is already split into low / medium / high vegetation by
height (classes 3 / 4 / 5), this pipeline runs the pre-processing recommended by
[artemis_treeiso](https://github.com/truebelief/artemis_treeiso), isolates the
individual trees, and writes an output cloud where **every isolated-tree point
gets its own `Tree` class plus a per-tree `treeID`**.

The [pure-Python treeiso](https://github.com/truebelief/artemis_treeiso)
implementation (`treeiso.py`, `cut_pursuit_L0.py`,
`cut_pursuit_L0_replica_cpp.py`) is vendored here unchanged except for one bug
fix (see [Notes](#notes)).

## What it does

Running `run_pipeline.py input.las` performs, in order:

1. **Ground removal** — drops points classified as ground (class 2). treeiso
   requires an above-ground cloud.
2. **Noise removal** — statistical outlier removal (SOR, NN=10, std=1), matching
   the README's CloudCompare recommendation.
3. **Decimation** — voxel down-sampling to ~2 cm.
4. **Tree isolation** — treeiso's three-stage cut-pursuit segmentation.
5. **Reclassification** — the per-tree labels are carried back onto the
   full-resolution vegetation and encoded as:
   - `classification` = a single **`Tree`** code (default **40**) for every
     isolated-tree point, and
   - a new **`treeID`** extra dimension (`uint32`, 1-based; `0` = not a tree).

Ground points and any vegetation not absorbed into a tree keep their original
classification (`treeID = 0`), so the existing low/mid/high-veg classes are
preserved for everything that is not an isolated tree.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.x with `numpy < 2.0` (see `requirements.txt`).

## Usage

```bash
python run_pipeline.py input.las
# -> input_classified.laz
```

Options:

| flag | default | meaning |
|------|---------|---------|
| `-o, --output` | `<input>_classified.laz` | output path |
| `--voxel` | `0.02` | decimation voxel size (m) |
| `--sor-k` | `10` | SOR neighbour count (NN) |
| `--sor-std` | `1.0` | SOR standard-deviation multiplier |
| `--tree-class` | `40` | classification code for tree points |
| `--carryback-radius` | `0.1` | max distance (m) to propagate tree labels back to full-res points |
| `--keep-intermediate` | off | also write `<input>_preprocessed.laz` |

### Output schema

| field | meaning |
|-------|---------|
| `classification` | unchanged except isolated-tree points set to `--tree-class` (40) |
| `treeID` | per-tree instance id, `1..N`; `0` for non-tree points |

## Running treeiso on its own

The vendored tool still works standalone on an already-preprocessed,
ground-removed folder of LAS/LAZ files:

```bash
python treeiso.py     # prompts for a directory; writes *_treeiso.laz
```

## Testing

`make_synthetic_las.py` generates a small synthetic cloud (flat ground + three
conical trees split into veg classes + noise) for an end-to-end smoke test:

```bash
python make_synthetic_las.py synth.las
python run_pipeline.py synth.las
```

## Notes

- treeiso assigns **every** above-ground point it is given to some tree segment;
  after ground and noise removal, the remaining vegetation is therefore all
  grouped into trees.
- **Bug fix vs upstream:** the upstream `treeiso.main` mapped the intermediate
  (2D) segmentation labels back with the wrong decimation index (RES1 instead of
  RES2), which raised an `IndexError`. The per-file logic is refactored into
  `treeiso.isolate_trees()` with the corrected mapping; both `run_pipeline.py`
  and standalone `treeiso.py` use it.

## Attribution & license

Built on **artemis_treeiso** by Zhouxin Xi and Chris Hopkinson, University of
Lethbridge — Artemis Lab. Please cite:

> Xi, Z.; Hopkinson, C. 3D Graph-Based Individual-Tree Isolation (treeiso) from
> terrestrial laser scanning point clouds.

The isolation relies on the cut-pursuit algorithm of Landrieu and Obozinski. See
[`LICENSE`](LICENSE) for the vendored treeiso / cut-pursuit license terms.
