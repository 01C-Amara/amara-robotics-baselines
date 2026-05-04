# Data setup

## Amara Spatial 10K

### 1. Download from HuggingFace

```bash
pip install huggingface_hub

python -m amara_robotics_baselines.scripts.download_data --dataset amara
# downloads to data/datasets/amara-spatial-10k/
```

Or with the HuggingFace CLI:

```bash
huggingface-cli download ZeroOneCreative/amara-spatial-10k \
  --repo-type dataset \
  --local-dir data/datasets/amara-spatial-10k
```

### 2. Extract mesh shards

The 3D meshes are stored as chunked tar archives under `meshes/`. Extract them:

```bash
python -m amara_robotics_baselines.scripts.extract_shards \
  --shard-dir data/datasets/amara-spatial-10k/meshes \
  --out-dir   data/datasets/amara-spatial-10k/extracted
```

Each `shard-*.tar` is deleted after extraction.

### 3. Filter household assets

Amara Spatial 10K spans many categories. The benchmark uses only the household subset
(kitchen, furniture, food, toys, etc.) filtered by category whitelist and bounding-box size.

```bash
python -m amara_robotics_baselines.scripts.filter_assets \
  --metadata-dir data/datasets/amara-spatial-10k/metadata \
  --out          data/datasets/amara-spatial-10k/filtered_manifest.parquet
```

Assets are classified into `manipulation` (< 0.5 m), `obstacle` (0.5–2.0 m), and `excluded`.

### 4. Generate object config files

```bash
python -m amara_robotics_baselines.scripts.generate_object_configs \
  --manifest      data/datasets/amara-spatial-10k/filtered_manifest.parquet \
  --extracted-dir data/datasets/amara-spatial-10k/extracted \
  --out-dir       data/datasets/amara-spatial-10k/configs
```

### 5. Generate VHACD collision meshes (optional)

Required only for `--collision-mode vhacd` or `--collision-mode all`.

```bash
pip install coacd

python -m amara_robotics_baselines.scripts.generate_vhacd_collisions \
  --manifest      data/datasets/amara-spatial-10k/filtered_manifest.parquet \
  --extracted-dir data/datasets/amara-spatial-10k/extracted \
  --workers 8
```

Output `.vhacd.glb` files are written alongside the existing meshes in `extracted/`.

### Final layout

```
data/datasets/amara-spatial-10k/
  extracted/                  *.glb, *.vhacd.glb
  metadata/
  configs/                    *.object_config.json
  filtered_manifest.parquet
```

---

## YCB

TODO: add download instructions.

---

## Objaverse

TODO: add download instructions.

---

## Fetch robot URDF (for graspability checks)

Requires habitat-sim to be installed (see [habitat.md](habitat.md)).

```bash
python -m amara_robotics_baselines.scripts.download_data --dataset fetch
```
