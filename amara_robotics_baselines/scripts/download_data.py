#!/usr/bin/env python3
"""Data download instructions for amara-robotics-baselines.

TODO: Replace placeholder URLs/commands with actual download links once hosting is finalized.

Expected directory layout after download:
  data/
    datasets/
      amara-spatial-10k/
        configs/          *.object_config.json
        meshes/           *.glb
      objaverse/
        configs/
        meshes/
    versioned_data/
      ycb/
        configs/
    robots/
      hab_fetch/
        robots/hab_fetch.urdf
        ...
"""

# Amara Spatial 10K
# -----------------
# TODO: add download command / URL

# Objaverse
# ---------
# pip install objaverse
# python -c "import objaverse; objaverse.load_uids()"

# YCB
# ---
# TODO: add download command / URL

# Fetch robot URDF
# ----------------
# Included with habitat-sim data:
#   python -m habitat_sim.utils.datasets_download --uids hab_fetch --data-path data/

if __name__ == "__main__":
    print(__doc__)
