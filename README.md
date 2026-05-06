# amara-robotics-baselines

Physics stability and grasping benchmarks for robotics object datasets.

<!-- Replace the path below with the actual GIF once generated:
     e.g. docs/assets/grasp_demo.gif or a direct URL -->
![Robot grasping demo](docs/assets/grasp_demo.gif)

Two independent simulation backends are supported — pick one or both:

| Backend | Simulator | Physics engine | Python |
| --- | --- | --- | --- |
| ManiSkill / SAPIEN | SAPIEN 3 | PhysX 5 | 3.10 |
| Habitat-sim | Bullet | Bullet 3 | 3.9 |

## Supported datasets

- **[Amara Spatial 10K](https://huggingface.co/datasets/ZeroOneCreative/amara-spatial-10k)** — large-scale 3D object dataset (household subset used for benchmarking)
- **YCB** — standard manipulation benchmark objects
- **Objaverse** — large-scale 3D asset dataset

## Getting started

1. **[Data setup](docs/data_setup.md)** — download, extract, filter, and generate configs for each dataset
2. **[Habitat-sim / Bullet backend](docs/habitat.md)** — installation and usage (recommended)
3. **[ManiSkill / SAPIEN backend](docs/maniskill.md)** — installation and usage

## Results inspector

After running checks, build the aggregated results JSON and launch the browser inspector:

```bash
python -m amara_robotics_baselines.scripts.build_results_json \
  --results-dir results/amara_physics

python -m amara_robotics_baselines.scripts.serve_inspector
# Open http://localhost:8000/inspector.html
```

## Metrics reference

See [docs/metrics.md](docs/metrics.md) for a full description of all CSV output columns.

## License

[CC BY 4.0](LICENSE) — Attribution 4.0 International.

## Citation

```bibtex
@misc{salehi2026amaraspatial10kspatiallysemanticallyaligned,
      title={AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI}, 
      author={Mohammad Sadegh Salehi and Alex Perkins and Igor Maurell and Ashkan Dabbagh and Raymond Wong},
      year={2026},
      eprint={2604.23018},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.23018}, 
}
```
