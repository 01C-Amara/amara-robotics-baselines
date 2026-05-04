# amara-robotics-baselines

Physics stability and grasping benchmarks for robotics object datasets.

Supports two simulation backends:

| Backend | Simulator | Physics engine |
|---|---|---|
| Habitat-sim | Bullet | PhysX (via Bullet) |
| ManiSkill / SAPIEN | SAPIEN 3 | PhysX 5 |

---

## Supported datasets

- **Amara Spatial 10K** — large-scale household object dataset with pre-computed collision meshes
- **YCB** — standard manipulation benchmark objects
- **Objaverse** — large-scale 3D asset dataset (raw render meshes used as collision)

---

## Installation

```bash
pip install -e .
```

### Backend dependencies

**Habitat-sim (Bullet) backend:**
```bash
pip install habitat-sim
pip install habitat
```

**ManiSkill / SAPIEN backend:**
```bash
pip install mani_skill
# or: pip install sapien
```

---

## Data download

See `amara_robotics_baselines/scripts/download_data.py` for instructions on fetching:

- `data/datasets/amara-spatial-10k/` — Amara Spatial 10K object configs and meshes
- `data/datasets/objaverse/` — Objaverse GLB assets
- `data/versioned_data/ycb/` — YCB object configs
- `data/robots/hab_fetch/` — Fetch robot URDF (for graspability checks)

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

### Collision modes (physics check)

| Mode | Description |
|---|---|
| `convex_hull` | Single convex hull from collision asset |
| `vhacd` | Approximate convex decomposition (.vhacd.glb) |
| `raw` | Render mesh used directly as collision geometry |
| `both` | Run convex_hull + vhacd |
| `all` | Run convex_hull + vhacd + raw |

---

## Results inspector

After running checks, build the results JSON and serve the inspector:

```bash
python -m amara_robotics_baselines.scripts.build_results_json \
  --results-dir results/amara_physics

python -m amara_robotics_baselines.scripts.serve_inspector
# opens http://localhost:8000/inspector.html
```

---

## Metrics reference

### Physics check (physics_results.csv)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `collision_mode` | str | convex_hull, vhacd, or raw |
| `physics_settles` | bool | Object reached near-zero velocity before timeout |
| `physics_stable` | bool | Settles + no fly-away + no floor penetration |
| `displacement_m` | float | XY displacement from spawn position (m) |
| `flies_away` | bool | XY displacement > 1.5 m |
| `penetration_y_m` | float | Minimum Z of object AABB (negative = below floor) |
| `floor_penetration` | bool | Penetration below -0.05 m |
| `settle_time_s` | float | Time to settle (s), null if never settled |
| `wall_time_s` | float | Elapsed wall-clock time (s) |
| `contact_points_at_rest` | int | Number of floor contact points at rest |
| `error` | str | Error message if check failed, null otherwise |

### Graspability check (graspability_results.csv)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `graspable` | bool | At least one valid antipodal grasp found |
| `grasp_score` | float | Geometric grasp quality score [0, 1] |
| `num_grasps` | int | Number of valid antipodal grasp candidates |
| `error` | str | Error message if check failed, null otherwise |

---

## Citation

```bibtex
@misc{amara-robotics-baselines,
  title  = {amara-robotics-baselines},
  author = {TODO},
  year   = {2024},
  url    = {TODO}
}
```
