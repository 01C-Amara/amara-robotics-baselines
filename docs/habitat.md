# Habitat-sim / Bullet backend

Physics engine: Bullet 3 via habitat-sim.

## Installation

```bash
conda create -n habitat python=3.9 -y
conda activate habitat

# habitat-sim from conda — the withbullet flag is required for physics
conda install -y -c aihabitat -c conda-forge habitat-sim=0.3.3 withbullet

pip install pandas pyarrow tqdm numpy scipy trimesh imageio Pillow pyrender

pip install -e .
```

## Physics stability check

```bash
python -m amara_robotics_baselines.scripts.run_physics_check \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir    results/amara_physics
```

## Graspability check

```bash
python -m amara_robotics_baselines.scripts.run_graspability_check \
  --config-dir data/datasets/amara-spatial-10k/configs \
  --out-dir    results/amara_grasp
```
