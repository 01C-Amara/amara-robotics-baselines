# ManiSkill / SAPIEN backend

Physics engine: PhysX 5 via SAPIEN 3. Requires a CUDA-capable GPU.

## Installation

```bash
conda create -n maniskill python=3.10 -y
conda activate maniskill

pip install mani_skill==3.0.1
pip install pandas pyarrow tqdm numpy scipy trimesh imageio Pillow pyrender

pip install -e .
```

> If running headless (server without display), set `PYOPENGL_PLATFORM=egl` before running scripts.

## Physics stability check

```bash
python -m amara_robotics_baselines.scripts.run_physics_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir    results/amara_physics \
  --collision-mode all \
  --workers 4
```

### Collision modes

| Mode | Description |
|---|---|
| `convex_hull` | Single convex hull from the collision asset |
| `vhacd` | Approximate convex decomposition (requires pre-generated `.vhacd.glb`) |
| `raw` | Render mesh passed directly to PhysX cooking |
| `both` | `convex_hull` + `vhacd` |
| `all` | `convex_hull` + `vhacd` + `raw` |

### Save GIFs

Add `--save-images` to capture a per-asset drop animation:

```bash
python -m amara_robotics_baselines.scripts.run_physics_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir    results/amara_physics \
  --collision-mode convex_hull \
  --save-images
```

GIFs are saved to `results/amara_physics/physics/images/<mode>/`.

## Graspability check

```bash
python -m amara_robotics_baselines.scripts.run_graspability_check_ms \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir    results/amara_grasp
```
