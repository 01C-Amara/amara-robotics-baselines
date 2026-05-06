#!/usr/bin/env python3
"""SAPIEN 3 / ManiSkill3 scene factory for the object benchmark.

Coordinate convention: SAPIEN uses Z-up, -Z gravity.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from scipy.spatial.transform import Rotation

FETCH_URDF = "data/robots/hab_fetch/robots/hab_fetch.urdf"

FLOOR_Z = 0.0
TABLE_HALF_SIZE = [0.6, 0.5, 0.35]  # X, Y, Z half-extents → top at Z = 0.35
TABLE_TOP_Z = TABLE_HALF_SIZE[2] * 2  # 0.70 m


def make_scene(with_renderer: bool = False, timestep: float = 1 / 60,
               use_gpu: bool = False) -> sapien.Scene:
    if use_gpu:
        physx.enable_gpu()
        physx.set_scene_config(gravity=np.array([0, 0, -9.81], dtype=np.float32))
        physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
        physx.set_body_config(solver_position_iterations=15, solver_velocity_iterations=1,
                              sleep_threshold=0.005)
        gpu_system    = physx.PhysxGpuSystem()
        render_system = sapien.render.RenderSystem()
        scene = sapien.Scene(systems=[gpu_system, render_system])
        gpu_system.set_timestep(timestep)
        if with_renderer:
            scene.set_ambient_light([0.4, 0.4, 0.4])
            scene.add_directional_light([1,  1, -1], [1.0, 1.0, 1.0], shadow=True)
            scene.add_directional_light([-1, 0, -1], [0.5, 0.5, 0.6])
            scene.add_directional_light([0, -1, -0.5], [0.4, 0.4, 0.4])
        # In GPU mode, place the ground well below the robot base (z=0) so the
        # robot's collision shapes don't contact it and cause unwanted lateral forces.
        scene.add_ground(altitude=-1.0)
    else:
        scene = sapien.Scene()
        scene.set_timestep(timestep)
        if with_renderer:
            scene.set_ambient_light([0.4, 0.4, 0.4])
            scene.add_directional_light([1,  1, -1], [1.0, 1.0, 1.0], shadow=True)
            scene.add_directional_light([-1, 0, -1], [0.5, 0.5, 0.6])
            scene.add_directional_light([0, -1, -0.5], [0.4, 0.4, 0.4])
        scene.add_ground(altitude=FLOOR_Z)
    return scene


def get_rb(entity: sapien.Entity) -> physx.PhysxRigidDynamicComponent:
    """Return the dynamic rigid-body component of an entity."""
    return entity.find_component_by_type(physx.PhysxRigidDynamicComponent)


def _up_to_sapien_pose(up: list) -> sapien.Pose:
    """Return a Pose whose rotation aligns the mesh's semantic 'up' vector with SAPIEN's world +Z.

    SAPIEN world is Z-up.  Asset configs declare the mesh-local 'up' direction.
    If that vector is already [0,0,1] no rotation is applied.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    asset_up = np.array(up, dtype=float)
    asset_up /= np.linalg.norm(asset_up)

    if np.allclose(asset_up, world_up):
        return sapien.Pose()

    # align_vectors finds the rotation mapping asset_up → world_up
    rot, _ = Rotation.align_vectors([world_up], [asset_up])
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    return sapien.Pose(q=[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _glb_full_scale(glb_path: str) -> float:
    """Return the full composed uniform scale of the first mesh-holding node.

    trimesh stores a node's own local scale as a self-edge (node, node) in the
    scene graph, separate from the parent→child edge.  For the two-level pattern
    (parent 'world' holds scale, child 'geometry_0' holds the mesh) the self-edge
    on 'world' carries the 0.4001 scale while the world→geometry_0 edge is 1.0.
    We find the first self-edge with a non-unity scale as the full composed scale.
    """
    try:
        import trimesh
        scene = trimesh.load(glb_path, process=False)
        if isinstance(scene, trimesh.Scene):
            for (src, dst), data in scene.graph.transforms.edge_data.items():
                if src == dst:
                    T = data.get("matrix")
                    if T is not None:
                        s = float(np.linalg.det(np.array(T)[:3, :3]) ** (1.0 / 3.0))
                        if abs(s - 1.0) > 1e-4:
                            return s
            # Fallback: geometry node's direct transform (handles one-level GLBs)
            for node in scene.graph.nodes_geometry:
                T, _ = scene.graph[node]
                s = float(np.linalg.det(T[:3, :3]) ** (1.0 / 3.0))
                if s > 0:
                    return s
    except Exception:
        pass
    return 1.0


def _glb_leaf_scale(glb_path: str) -> float:
    """Return the scale on the node that DIRECTLY holds the mesh (leaf-level only).

    SAPIEN's collision loader applies only the leaf node's own scale, not parent
    scales.  For two-level GLBs (scale on 'world', mesh on child), returns 1.0.
    For one-level GLBs (scale and mesh on the same node), returns that scale.
    """
    try:
        import trimesh
        scene = trimesh.load(glb_path, process=False)
        if isinstance(scene, trimesh.Scene):
            for node in scene.graph.nodes_geometry:
                T, _ = scene.graph[node]
                s = float(np.linalg.det(T[:3, :3]) ** (1.0 / 3.0))
                return s if s > 0 else 1.0
    except Exception:
        pass
    return 1.0


def load_object(
    scene: sapien.Scene,
    config_json_path: str,
    collision_mode: str = "convex_hull",
    scale: float = 1.0,
    name: Optional[str] = None,
) -> sapien.Entity:
    """Load a rigid dynamic object from a .object_config.json into a SAPIEN scene.

    Applies the config's 'up' orientation and 'friction_coefficient' so behaviour
    matches the Habitat/Bullet baseline.
    """
    cfg = json.loads(Path(config_json_path).read_text())
    config_dir = Path(config_json_path).parent
    render_path = str((config_dir / cfg["render_asset"]).resolve())
    collision_path = str((config_dir / cfg["collision_asset"]).resolve())

    if collision_mode == "raw":
        collision_path = render_path
    elif collision_mode == "vhacd":
        vhacd = render_path.replace(".glb", ".vhacd.glb")
        if not os.path.exists(vhacd):
            raise FileNotFoundError(f"VHACD not found: {vhacd}")
        collision_path = vhacd

    # Orientation: rotate mesh so its semantic 'up' aligns with SAPIEN world +Z
    up = cfg.get("up", [0.0, 0.0, 1.0])
    orient_pose = _up_to_sapien_pose(up)

    # Friction from config (static = dynamic = friction_coefficient)
    friction = float(cfg.get("friction_coefficient", 0.5))
    material = physx.PhysxMaterial(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )

    # Scale handling:
    # add_visual_from_file traverses the full GLB scene graph and auto-applies
    # the composed node scale (including parent 'world' scales). We pass 1.0
    # unless render and collision full scales differ (rare).
    #
    # add_multiple_convex_collisions_from_file applies ONLY the scale on the leaf
    # node that directly holds the mesh, ignoring parent scales.
    #   - collision.glb (one-level): leaf has scale=0.4001 → SAPIEN applies it,
    #     we pass 1.0 to avoid double-scaling. (full=0.4001, leaf=0.4001 → explicit=1.0)
    #   - vhacd.glb: vertices are pre-scaled (scene.dump() was used at generation time,
    #     which applies the full composed transform). Pass scale=1.0 to avoid re-scaling.
    #   - render.glb (two-level): leaf has scale=1.0, parent has 0.4001
    #     → SAPIEN applies 1.0, we must pass full_scale explicitly.
    #     (full=0.4001, leaf=1.0 → explicit=0.4001)
    collision_full_scale = _glb_full_scale(collision_path)
    collision_leaf_scale = _glb_leaf_scale(collision_path)
    render_full_scale    = _glb_full_scale(render_path)
    # If leaf_scale ≈ 1.0 the GLB is two-level: SAPIEN traverses the full graph and
    # applies full_scale automatically. Passing it again would double-scale the mesh.
    # collision.glb is one-level (leaf_scale = full_scale) so explicit = 1.0 via formula.
    if abs(collision_leaf_scale - 1.0) < 0.01:
        collision_explicit = scale
    else:
        collision_explicit = scale * collision_full_scale / max(collision_leaf_scale, 1e-9)
    if abs(collision_full_scale - render_full_scale) > 1e-4:
        visual_scale = scale * (collision_full_scale / render_full_scale)
    else:
        visual_scale = scale

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(render_path, pose=orient_pose, scale=[visual_scale] * 3)
    builder.add_multiple_convex_collisions_from_file(
        collision_path, pose=orient_pose, scale=[collision_explicit] * 3, material=material
    )
    entity_name = name or os.path.basename(config_json_path)
    return builder.build(name=entity_name)


def add_table(scene: sapien.Scene) -> Tuple[sapien.Entity, float]:
    """Add a static box table. Returns (entity, table_top_z)."""
    hx, hy, hz = TABLE_HALF_SIZE
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=[0.6, 0.4, 0.2, 1.0])
    builder.add_box_collision(half_size=[hx, hy, hz])
    builder.add_box_visual(half_size=[hx, hy, hz], material=mat)
    table = builder.build_static(name="__table__")
    table.set_pose(sapien.Pose(p=[0, 0, hz]))  # bottom at Z=0, top at Z=2*hz
    return table, TABLE_TOP_Z


def load_fetch_robot(
    scene: sapien.Scene,
    urdf_path: str = FETCH_URDF,
    setup_drives: bool = False,
) -> sapien.physx.PhysxArticulation:
    """Load Fetch URDF into SAPIEN scene. Returns the Articulation."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_path)
    # Disable gravity on all robot links — matches ManiSkill's balance_passive_force
    # behaviour, which works around the lack of gravity compensation in PhysX.
    for link in robot.get_links():
        link.disable_gravity = True
    if setup_drives:
        setup_robot_drives(robot)
    return robot


def setup_robot_drives(
    robot: sapien.physx.PhysxArticulation,
    stiffness: float = 1e4,
    damping: float = 5e2,
    finger_stiffness: float = 5e3,
    finger_damping: float = 2e2,
) -> None:
    """Set PD drive properties on all active joints so set_qpos targets are held."""
    finger_names = {"r_gripper_finger_joint", "l_gripper_finger_joint"}
    for j in robot.get_active_joints():
        if j.name in finger_names:
            j.set_drive_properties(stiffness=finger_stiffness, damping=finger_damping,
                                   force_limit=200.0)
        else:
            j.set_drive_properties(stiffness=stiffness, damping=damping)


def snap_down(
    scene: sapien.Scene,
    entity: sapien.Entity,
    support_entities: list,
    max_steps: int = 300,
    vel_threshold: float = 0.005,
) -> bool:
    """Drop entity under gravity until it settles on a support surface or times out."""
    rb = get_rb(entity)
    rb.set_kinematic(False)
    for _ in range(max_steps):
        scene.step()
        v = np.linalg.norm(rb.get_linear_velocity())
        if v < vel_threshold:
            return True
    return False


# ── GPU simulation helpers ─────────────────────────────────────────────────────
# These functions work with PhysxGpuSystem scenes where state must be read/written
# through CUDA buffers rather than the normal SAPIEN entity API.

def fetch_all_gpu(px: physx.PhysxGpuSystem) -> None:
    """Pull all simulation state from GPU buffers to CPU-accessible tensors."""
    px.gpu_fetch_rigid_dynamic_data()
    px.gpu_fetch_articulation_link_pose()
    px.gpu_fetch_articulation_qpos()
    px.gpu_fetch_articulation_qvel()


def apply_all_gpu(px: physx.PhysxGpuSystem, apply_root_pose: bool = False) -> None:
    """Push all CPU-side state changes to GPU buffers and update kinematics.

    apply_root_pose should be False for fix_root_link articulations (Fetch) since
    the root is fixed to the world and overriding its pose causes instability.
    """
    px.gpu_apply_rigid_dynamic_data()
    if apply_root_pose:
        px.gpu_apply_articulation_root_pose()
    px.gpu_apply_articulation_qpos()
    px.gpu_apply_articulation_qvel()
    px.gpu_apply_articulation_target_position()
    px.gpu_update_articulation_kinematics()


def set_rb_pose_gpu(px: physx.PhysxGpuSystem,
                    rb: physx.PhysxRigidDynamicComponent,
                    pose: sapien.Pose, zero_vel: bool = True) -> None:
    """Write rigid body pose (and optionally zero velocities) to CUDA buffer."""
    import torch
    buf = px.cuda_rigid_dynamic_data.torch()
    idx = rb.gpu_pose_index
    p, q = pose.p, pose.q  # q = [w, x, y, z]
    buf[idx, :7] = torch.tensor([p[0], p[1], p[2], q[0], q[1], q[2], q[3]],
                                 dtype=torch.float32)
    if zero_vel:
        buf[idx, 7:] = 0.0


def get_rb_pose_gpu(px: physx.PhysxGpuSystem,
                    rb: physx.PhysxRigidDynamicComponent) -> sapien.Pose:
    """Read rigid body pose from the CUDA buffer (call after gpu_fetch_rigid_dynamic_data)."""
    buf = px.cuda_rigid_dynamic_data.torch()
    data = buf[rb.gpu_pose_index, :7].cpu().numpy()
    return sapien.Pose(p=data[:3].tolist(), q=data[3:7].tolist())


def get_rb_lin_vel_magnitude_gpu(px: physx.PhysxGpuSystem,
                                  rb: physx.PhysxRigidDynamicComponent) -> float:
    """Read linear speed of a rigid body from CUDA buffer (after gpu_fetch_rigid_dynamic_data)."""
    buf = px.cuda_rigid_dynamic_data.torch()
    return float(buf[rb.gpu_pose_index, 7:10].norm().item())


def set_robot_state_gpu(px: physx.PhysxGpuSystem,
                        robot: physx.PhysxArticulation,
                        qpos: np.ndarray,
                        root_pose: Optional[sapien.Pose] = None,
                        zero_vel: bool = True) -> None:
    """Write full robot state (qpos + optional root pose) to CUDA buffers.

    Call apply_all_gpu(px) afterwards to push the state into the simulation.
    """
    import torch
    art_idx = robot.gpu_index
    qpos_t = torch.tensor(qpos, dtype=torch.float32)
    n = len(qpos)
    px.cuda_articulation_qpos.torch()[art_idx, :n] = qpos_t
    px.cuda_articulation_target_qpos.torch()[art_idx, :n] = qpos_t
    if zero_vel:
        px.cuda_articulation_qvel.torch()[art_idx, :] = 0.0

    if root_pose is not None:
        p, q = root_pose.p, root_pose.q
        buf = px.cuda_articulation_link_data.torch()
        buf[art_idx, 0, :7] = torch.tensor([p[0], p[1], p[2], q[0], q[1], q[2], q[3]],
                                            dtype=torch.float32)


def get_link_pose_gpu(px: physx.PhysxGpuSystem,
                      robot: physx.PhysxArticulation,
                      link_name: str) -> sapien.Pose:
    """Read a named link's world pose from CUDA buffer (call after gpu_fetch_articulation_link_pose).

    Uses the link's PhysxArticulationLinkComponent.gpu_pose_index as the link dimension index.
    """
    art_idx = robot.gpu_index
    buf = px.cuda_articulation_link_data.torch()
    for link in robot.get_links():
        if link.name == link_name:
            lc = link.entity.find_component_by_type(physx.PhysxArticulationLinkComponent)
            li = lc.gpu_pose_index
            data = buf[art_idx, li, :7].cpu().numpy()
            return sapien.Pose(p=data[:3].tolist(), q=data[3:7].tolist())
    raise RuntimeError(f"Link '{link_name}' not found in robot")


def set_drive_targets_gpu(px: physx.PhysxGpuSystem,
                          robot: physx.PhysxArticulation,
                          qpos_target: np.ndarray) -> None:
    """Write drive targets to CUDA buffer. Call px.gpu_apply_articulation_target_position() after."""
    import torch
    art_idx = robot.gpu_index
    n = len(qpos_target)
    px.cuda_articulation_target_qpos.torch()[art_idx, :n] = \
        torch.tensor(qpos_target, dtype=torch.float32)


def set_qpos_kinematic_gpu(px: physx.PhysxGpuSystem,
                           robot: physx.PhysxArticulation,
                           qpos: np.ndarray) -> None:
    """Teleport robot joints to qpos (kinematic, no PD physics).

    Writes qpos directly and zeroes velocities so the arm moves without PD
    dynamics. The GPU physics step still runs, so dynamic objects receive
    contact forces from the arm's collision shapes.
    Call this before scene.step().
    """
    import torch
    art_idx = robot.gpu_index
    n = len(qpos)
    qpos_t = torch.tensor(qpos, dtype=torch.float32)
    px.cuda_articulation_qpos.torch()[art_idx, :n] = qpos_t
    px.cuda_articulation_target_qpos.torch()[art_idx, :n] = qpos_t  # zero PD error
    px.cuda_articulation_qvel.torch()[art_idx, :] = 0.0
    px.gpu_apply_articulation_qpos()
    px.gpu_apply_articulation_target_position()
    px.gpu_apply_articulation_qvel()
    px.gpu_update_articulation_kinematics()


def snap_down_gpu(scene: sapien.Scene,
                  px: physx.PhysxGpuSystem,
                  rb: physx.PhysxRigidDynamicComponent,
                  max_steps: int = 300,
                  vel_threshold: float = 0.005) -> bool:
    """GPU version of snap_down: step until object velocity falls below threshold."""
    for _ in range(max_steps):
        scene.step()
        px.gpu_fetch_rigid_dynamic_data()
        if get_rb_lin_vel_magnitude_gpu(px, rb) < vel_threshold:
            return True
    return False


def init_gpu_scene(scene: sapien.Scene,
                   robot: physx.PhysxArticulation,
                   init_qpos: np.ndarray,
                   root_pose: Optional[sapien.Pose] = None,
                   dynamic_entities: Optional[list] = None,
                   entity_poses: Optional[list] = None) -> physx.PhysxGpuSystem:
    """Call gpu_init and set initial state for a GPU scene.

    Must be called after ALL actors/articulations are added to the scene.
    Returns the PhysxGpuSystem for subsequent buffer access.
    """
    px = scene.physx_system
    px.gpu_init()

    if root_pose is None:
        root_pose = sapien.Pose()

    set_robot_state_gpu(px, robot, init_qpos, root_pose=root_pose, zero_vel=True)

    if dynamic_entities and entity_poses:
        for entity, pose in zip(dynamic_entities, entity_poses):
            rb = get_rb(entity)
            set_rb_pose_gpu(px, rb, pose, zero_vel=True)

    apply_all_gpu(px)
    return px
