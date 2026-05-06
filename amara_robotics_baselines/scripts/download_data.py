#!/usr/bin/env python3
"""Download datasets for amara-robotics-baselines.

Usage
-----
  # Amara Spatial 10K (from HuggingFace)
  python -m amara_robotics_baselines.scripts.download_data --dataset amara

  # Fetch robot URDF (requires habitat-sim)
  python -m amara_robotics_baselines.scripts.download_data --dataset fetch

  # All
  python -m amara_robotics_baselines.scripts.download_data --dataset all
"""

import argparse
from pathlib import Path


def download_amara(data_root: Path) -> None:
    """Download Amara Spatial 10K from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    out_dir = data_root / "datasets" / "amara-spatial-10k"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ZeroOneCreative/amara-spatial-10k → {out_dir}")
    snapshot_download(
        repo_id="ZeroOneCreative/amara-spatial-10k",
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    print("Done. Next steps:")
    print("  1. python -m amara_robotics_baselines.scripts.filter_assets ...")
    print("  2. python -m amara_robotics_baselines.scripts.generate_object_configs ...")


def download_fetch(data_root: Path) -> None:
    """Download Fetch robot URDF via habitat-sim."""
    try:
        import habitat_sim  # noqa: F401
    except ImportError:
        raise SystemExit("Fetch URDF requires habitat-sim. Install it first (see README).")

    import subprocess
    subprocess.run(
        [
            "python", "-m", "habitat_sim.utils.datasets_download",
            "--uids", "hab_fetch",
            "--data-path", str(data_root),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["amara", "fetch", "all"],
        default="all",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )
    args = parser.parse_args()

    if args.dataset in ("amara", "all"):
        download_amara(args.data_root)
    if args.dataset in ("fetch", "all"):
        download_fetch(args.data_root)


if __name__ == "__main__":
    main()
