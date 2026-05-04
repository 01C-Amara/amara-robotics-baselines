# amara-robotics-baselines

Physics stability and grasping benchmarks for robotics object datasets.

Two independent simulation backends are supported — pick one or both:

| Backend | Simulator | Physics engine | Python |
|---|---|---|---|
| ManiSkill / SAPIEN | SAPIEN 3 | PhysX 5 | 3.10 |
| Habitat-sim | Bullet | Bullet 3 | 3.9 |

---

## Supported datasets

- **Amara Spatial 10K** — large-scale 3D object dataset (household subset used for benchmarking)
- **YCB** — standard manipulation benchmark objects
- **Objaverse** — large-scale 3D asset dataset (raw render meshes used as collision)

---

## Installation

### ManiSkill / SAPIEN backend (recommended)

```bash
conda create -n maniskill python=3.10 -y
conda activate maniskill

pip install mani_skill==3.0.1
pip install pandas pyarrow tqdm numpy scipy trimesh imageio Pillow pyrender

pip install -e .
```

> **GPU note:** ManiSkill requires a CUDA-capable GPU for rendering.
> If running headless (e.g. a server), set `PYOPENGL_PLATFORM=egl` before running scripts.

### Habitat-sim / Bullet backend

```bash
conda create -n habitat python=3.9 -y
conda activate habitat

# Install habitat-sim from conda (includes Bullet physics)
conda install -y -c aihabitat -c conda-forge habitat-sim=0.3.3 withbullet

pip install pandas pyarrow tqdm numpy scipy trimesh imageio Pillow pyrender

pip install -e .
```

---

## Data setup

### 1. Download raw assets

**Amara Spatial 10K** is hosted on HuggingFace at [ZeroOneCreative/amara-spatial-10k](https://huggingface.co/datasets/ZeroOneCreative/amara-spatial-10k).

```bash
pip install huggingface_hub

python -m amara_robotics_baselines.scripts.download_data --dataset amara
# downloads to data/datasets/amara-spatial-10k/
```

Or manually with the HuggingFace CLI:

```bash
huggingface-cli download ZeroOneCreative/amara-spatial-10k \
  --repo-type dataset \
  --local-dir data/datasets/amara-spatial-10k
```

After download, Amara Spatial 10K assets are at:
```
data/datasets/amara-spatial-10k/
  extracted/    per-asset subfolders with *.glb meshes
  metadata/     per-asset metadata JSON files
```

### 2. Filter household assets (Amara only)

Amara Spatial 10K spans many categories. The benchmark uses only the household subset
(kitchen, furniture, food, toys, etc.) filtered by category whitelist and bounding-box size.
This step produces the `filtered_manifest.parquet` used by all downstream scripts:

```bash
python -m amara_robotics_baselines.scripts.filter_assets \
  --metadata-dir data/datasets/amara-spatial-10k/metadata \
  --out          data/datasets/amara-spatial-10k/filtered_manifest.parquet
```

Assets are classified into `manipulation` (< 0.5 m), `obstacle` (0.5–2.0 m), and `excluded`.

### 3. Generate object config files (Amara only)

The `.object_config.json` files required by the benchmark are generated from the filtered manifest:

```bash
python -m amara_robotics_baselines.scripts.generate_object_configs \
  --manifest      data/datasets/amara-spatial-10k/filtered_manifest.parquet \
  --extracted-dir data/datasets/amara-spatial-10k/extracted \
  --out-dir       data/datasets/amara-spatial-10k/configs
```

### 4. Generate VHACD collision meshes (optional, Amara only)

Required only for `--collision-mode vhacd` or `--collision-mode all`.
Runs CoACD on every render mesh — use `--workers` to parallelize:

```bash
pip install coacd

python -m amara_robotics_baselines.scripts.generate_vhacd_collisions \
  --manifest      data/datasets/amara-spatial-10k/filtered_manifest.parquet \
  --extracted-dir data/datasets/amara-spatial-10k/extracted \
  --workers 8
```

Output `.vhacd.glb` files are written alongside the existing meshes in `extracted/`.

### Expected layout after setup

```
data/
  datasets/
    amara-spatial-10k/
      extracted/    *.glb, *.vhacd.glb
      metadata/
      configs/      *.object_config.json
      filtered_manifest.parquet
    objaverse/
      configs/
      meshes/
  versioned_data/
    ycb/
      configs/
  robots/
    hab_fetch/
      robots/hab_fetch.urdf
```

---

## Quick start

### Physics stability check

**ManiSkill / SAPIEN backend:**
```bash
python -m amara_robotics_baselines.scripts.run_physics_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir results/amara_physics \
  --collision-mode all \
  --workers 4
```

**Habitat-sim / Bullet backend:**
```bash
python -m amara_robotics_baselines.scripts.run_physics_check \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir results/amara_physics
```

### Graspability check

**ManiSkill / SAPIEN backend:**
```bash
python -m amara_robotics_baselines.scripts.run_graspability_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir results/amara_grasp
```

**Habitat-sim / Bullet backend:**
```bash
python -m amara_robotics_baselines.scripts.run_graspability_check \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir results/amara_grasp
```

### Collision modes (physics check only)

| Mode | Description |
|---|---|
| `convex_hull` | Single convex hull computed from the collision asset |
| `vhacd` | Approximate convex decomposition (requires pre-generated `.vhacd.glb`) |
| `raw` | Render mesh passed directly to PhysX cooking (same pipeline as Objaverse) |
| `both` | Runs `convex_hull` + `vhacd` |
| `all` | Runs `convex_hull` + `vhacd` + `raw` |

### Generating GIFs (optional)

Add `--save-images` to any physics check command to save a `.gif` per asset showing the drop simulation:

```bash
python -m amara_robotics_baselines.scripts.run_physics_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir results/amara_physics \
  --collision-mode convex_hull \
  --save-images
```

GIFs are saved to `results/amara_physics/physics/images/<mode>/`.

---

## Results inspector

Build the aggregated results JSON and launch the browser inspector:

```bash
python -m amara_robotics_baselines.scripts.build_results_json \
  --results-dir results/amara_physics

python -m amara_robotics_baselines.scripts.serve_inspector
# Open http://localhost:8000/inspector.html
```

---

## Metrics reference

### Physics check (`physics_results.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `collision_mode` | str | `convex_hull`, `vhacd`, or `raw` |
| `physics_settles` | bool | Object reached near-zero velocity before timeout |
| `physics_stable` | bool | Settles + no fly-away + no floor penetration |
| `displacement_m` | float | Final XY displacement from spawn position (m) |
| `flies_away` | bool | XY displacement > 1.5 m |
| `penetration_y_m` | float | Minimum Z of object AABB (negative = below floor) |
| `floor_penetration` | bool | Penetration below −0.05 m |
| `settle_time_s` | float | Time to settle (s), null if never settled |
| `wall_time_s` | float | Elapsed wall-clock time per asset (s) |
| `contact_points_at_rest` | int | Number of floor contact points at rest |
| `error` | str | Error message if the check failed, null otherwise |

### Graspability check (`graspability_results.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `graspable` | bool | At least one valid antipodal grasp found |
| `grasp_score` | float | Geometric grasp quality score [0, 1] |
| `num_grasps` | int | Number of valid antipodal grasp candidates |
| `error` | str | Error message if the check failed, null otherwise |

---

## Citation

```bibtex
@misc{amara-robotics-baselines,
  title  = {amara-robotics-baselines},
  author = {TODO},
  year   = {2025},
  url    = {TODO}
}
```
