#!/usr/bin/env python3
"""Interactive SAPIEN viewer to inspect collision vs visual mesh alignment.

Opens a live 3D viewer with the object snapped on the table.
In the viewer UI: Rendering → Show Collision to overlay collision shapes.

Usage:
    python -m amara_robotics_baselines.scripts.debug_collision_viewer \
        --config path/to/asset.object_config.json \
        [--collision-mode convex_hull|vhacd|raw]
"""

import argparse
from pathlib import Path

import sapien

from amara_robotics_baselines.utils.maniskill_factory import (
    TABLE_TOP_Z,
    add_table,
    load_object,
    make_scene,
    snap_down,
)


def main():
    parser = argparse.ArgumentParser(description="SAPIEN collision viewer")
    parser.add_argument("--config",         required=True, type=Path)
    parser.add_argument("--collision-mode", default="convex_hull",
                        choices=["convex_hull", "vhacd", "raw"])
    args = parser.parse_args()

    scene = make_scene(with_renderer=True)
    table_entity, _ = add_table(scene)

    obj = load_object(scene, str(args.config), collision_mode=args.collision_mode)
    obj.set_pose(sapien.Pose(p=[0.0, 0.0, TABLE_TOP_Z + 0.5]))

    print("Snapping object to table...")
    if snap_down(scene, obj, [table_entity]):
        pose = obj.get_pose()
        print(f"Settled at z={pose.p[2]:.4f}")
    else:
        print("Warning: snap did not settle — object placed at drop height")

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [1, 1, 1])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(1.5, -0.5, 1.2)
    viewer.set_camera_rpy(0, -0.5, 2.8)

    print("\nViewer open. In the UI: Rendering → Show Collision")
    print("Press Q or close the window to exit.\n")

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
