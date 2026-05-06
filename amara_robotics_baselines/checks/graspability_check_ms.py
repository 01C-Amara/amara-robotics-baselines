#!/usr/bin/env python3
"""Snap-based graspability check using SAPIEN 3 / ManiSkill3 (PhysX + pytorch_kinematics IK).

Coordinate convention: SAPIEN uses Z-up, gravity = -Z.
Object geometry helpers (_load_glb_mesh, _sample_candidates) are reused from
graspability_check.py unchanged — they work in mesh-local space regardless of
world convention.

Output dict schema is identical to graspability_check.run_snap() so build_results_json.py
and the inspector work unchanged.
"""

import math
import os
import json as _json_mod
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh
from scipy.spatial.transform import Rotation as _Rotation

# Constants (copied from graspability_check.py — no habitat_sim import needed)
ARM_INIT        = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005], dtype=np.float32)
GRIPPER_Z_OFFSET = -0.04  # EE offset from object top face (negative = below top)
APPROACH_DIST   = 0.80
FALL_THRESHOLD  = 0.05
GRASP_CANDIDATES = 8
GIF_FPS         = 30
GRIPPER_CLOSED  = 0.0
GRIPPER_OPEN    = 0.04
HOLD_STEPS      = 60
LIFT_HEIGHT     = 0.20
LIFT_THRESHOLD  = 0.10
MAX_DRIVE_STEPS   = 500   # safety cap per phase
DRIVE_TOL         = 0.01  # joint convergence tolerance (rad / m)
DRIVE_SPEED       = 0.02  # max joint delta per sim step (rad / m) — controls arm speed
SNAP_HOLD_STEPS = 40
SNAP_LIFT_STEPS = 80
SNAP_RELEASE_STEPS = 30
SNAP_THRESHOLD  = 0.15
TORSO_HEIGHT    = 0.15
TORSO_MAX       = 0.38
TORSO_SEARCH    = [0.15, 0.10, 0.20, 0.05, 0.25, 0.0, 0.30, 0.38]
_ANTIPODAL_THRESH = 0.7


def _load_glb_mesh(glb_path):
    import json as _json, struct as _struct
    gltf_scale = 1.0
    scale_on_leaf = False  # True when the scale node directly holds a mesh
    try:
        with open(glb_path, "rb") as fh:
            fh.read(12)
            chunk_len = _struct.unpack("<I", fh.read(4))[0]
            fh.read(4)
            gltf = _json.loads(fh.read(chunk_len))
        for node in gltf.get("nodes", []):
            s = node.get("scale")
            if s and len(s) == 3 and not all(abs(v - 1.0) < 1e-6 for v in s):
                gltf_scale = float(s[0])
                scale_on_leaf = node.get("mesh") is not None
                break
    except Exception:
        pass
    loaded = trimesh.load(glb_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    # trimesh applies scale only when it is on the leaf node that directly holds the mesh.
    # For two-level GLBs (scale on parent 'world', mesh on child), trimesh misses it.
    # Apply the missing scale manually in that case.
    if abs(gltf_scale - 1.0) > 1e-6 and not scale_on_leaf:
        loaded = loaded.copy()
        loaded.vertices *= gltf_scale
    return loaded


def _sample_candidates(mesh, n_samples=500, filter_width=True):
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]
    nudge = mesh.scale * 5e-3
    locs, ray_ids, tri_ids = mesh.ray.intersects_location(
        points + normals * nudge, -normals, multiple_hits=True)
    hits = defaultdict(list)
    for loc, rid, tid in zip(locs, ray_ids, tri_ids):
        hits[rid].append((loc, tid))
    candidates = []
    for rid, hit_list in hits.items():
        p1 = points[rid]
        n1 = normals[rid]
        dists = [np.linalg.norm(h - p1) for h, _ in hit_list]
        best_idx = int(np.argmax(dists))
        p2, tid2 = hit_list[best_idx]
        n2 = mesh.face_normals[tid2]
        width = float(np.linalg.norm(p2 - p1))
        if width <= nudge * 2:
            continue
        if filter_width and width > GRIPPER_OPEN * 2:
            continue
        axis = (p2 - p1) / width
        score1 = float(np.dot(-n1, axis))
        score2 = float(np.dot( n2, axis))
        if score1 < _ANTIPODAL_THRESH or score2 < _ANTIPODAL_THRESH:
            continue
        candidates.append(((score1 + score2) / 2.0, p1, p2, width))
    candidates.sort(key=lambda x: -x[0])
    return candidates, points, normals


def _pick_best_trial(results):
    with_frames = [r for r in results if r.get("frames")]
    if not with_frames:
        return results[0]
    successes = [r for r in with_frames if r.get("success")]
    if successes:
        return max(successes, key=lambda r: r.get("obj_y_rise", 0.0))
    return max(with_frames, key=lambda r: r.get("obj_y_rise") or 0.0)
from amara_robotics_baselines.utils.maniskill_factory import (
    TABLE_TOP_Z,
    add_table,
    get_rb,
    load_object,
    snap_down,
    make_scene,
    load_fetch_robot,
    setup_robot_drives,
    fetch_all_gpu,
    apply_all_gpu,
    set_rb_pose_gpu,
    get_rb_pose_gpu,
    get_rb_lin_vel_magnitude_gpu,
    set_robot_state_gpu,
    get_link_pose_gpu,
    set_drive_targets_gpu,
    set_qpos_kinematic_gpu,
    snap_down_gpu,
    init_gpu_scene,
)

import pytorch_kinematics as pk

# ── ManiSkill Fetch constants ─────────────────────────────────────────────────

MS_FETCH_URDF = str(Path(__file__).parent.parent.parent /
                    "data/robots/hab_fetch/robots/hab_fetch.urdf")
# Fall back to the bundled ManiSkill Fetch if hab_fetch not loadable
try:
    import importlib.util
    _ms_pkg = importlib.util.find_spec("mani_skill")
    if _ms_pkg:
        _ms_dir = Path(_ms_pkg.origin).parent
        MS_FETCH_URDF_BUNDLED = str(_ms_dir / "assets/robots/fetch/fetch.urdf")
        MS_FETCH_URDF_ARM_IK  = str(_ms_dir / "assets/robots/fetch/fetch_torso_up.urdf")
except Exception:
    MS_FETCH_URDF_BUNDLED = MS_FETCH_URDF
    MS_FETCH_URDF_ARM_IK  = MS_FETCH_URDF

# qpos indices for ManiSkill Fetch (from active_joints inspection):
#   [0] root_x, [1] root_y, [2] root_z_rotation
#   [3] torso_lift_joint
#   [4] head_pan, [5] shoulder_pan, [6] head_tilt,
#   [7] shoulder_lift, [8] upperarm_roll, [9] elbow_flex,
#   [10] forearm_roll, [11] wrist_flex, [12] wrist_roll
#   [13] r_gripper_finger, [14] l_gripper_finger
QPOS_BASE_X   = 0
QPOS_BASE_Y   = 1
QPOS_BASE_YAW = 2
QPOS_TORSO    = 3
QPOS_ARM      = [5, 7, 8, 9, 10, 11, 12]   # 7-DOF arm: shoulder_pan → wrist_roll
QPOS_FINGERS  = [13, 14]
# Indices within the 7-DOF arm array that are continuous (no joint limits, can wrap):
# upperarm_roll(2), forearm_roll(4), wrist_roll(6)
_CONTINUOUS_ARM = [2, 4, 6]

FLOOR_Z = 0.0


def _make_ik_solver(urdf_arm_path: str):
    """Build a pytorch_kinematics serial chain + PseudoInverseIK for the Fetch arm.

    Returns (chain, ik_solver, lo_limits, hi_limits).
    lo/hi are only used for FK validation tolerance; joint limits are NOT hard-enforced
    since this check only cares about EE reachability, not arm configuration validity.
    """
    chain = pk.build_serial_chain_from_urdf(
        open(urdf_arm_path).read(), "gripper_link",
        root_link_name="torso_lift_link"
    )
    lo, hi = chain.get_joint_limits()
    lo, hi = np.array(lo), np.array(hi)
    # Continuous joints (type="continuous") get [0, 0] — override to [-π, π]
    continuous = (lo == 0) & (hi == 0)
    lo[continuous] = -math.pi
    hi[continuous] = math.pi
    limits = torch.tensor(np.stack([lo, hi], axis=1), dtype=torch.float32)
    ik_solver = pk.PseudoInverseIKWithSVD(chain, max_iterations=200, num_retries=20,
                                          joint_limits=limits)
    return chain, ik_solver, lo, hi


def _fk(chain, arm_angles: np.ndarray) -> np.ndarray:
    """FK: arm angles → EE position in torso frame."""
    q = torch.tensor(arm_angles, dtype=torch.float32).unsqueeze(0)
    mat = chain.forward_kinematics(q).get_matrix()
    return mat[0, :3, 3].numpy()


def _unwrap_to_reference(angles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shift continuous arm joints by multiples of 2π to minimize distance to reference.

    Prevents the PD drive interpolation from sweeping a full rotation on joints like
    forearm_roll and wrist_roll that have no angular limits.
    """
    result = angles.copy()
    for i in _CONTINUOUS_ARM:
        diff = reference[i] - result[i]
        result[i] += round(diff / (2 * math.pi)) * 2 * math.pi
    return result


def _solve_ik(chain, ik_solver, lo_limits, hi_limits,
              target_in_torso: np.ndarray,
              rot: np.ndarray = None,
              seed: np.ndarray = None,
              fk_tol: float = 0.05) -> Optional[np.ndarray]:
    """Solve IK for EE target in torso frame.

    rot:  optional 3×3 rotation matrix (in torso frame) for orientation-constrained IK.
    seed: optional 7-DOF arm angles to initialize the solver (keeps solution close to
          current config, avoiding wild joint trajectories).
    Returns 7 arm angles, or None if IK does not converge within fk_tol.
    """
    pos = torch.tensor(target_in_torso, dtype=torch.float32).unsqueeze(0)
    if rot is not None:
        rot_t = torch.tensor(rot, dtype=torch.float32).unsqueeze(0)
        goal = pk.Transform3d(pos=pos, rot=rot_t)
    else:
        goal = pk.Transform3d(pos=pos)
    if seed is not None:
        # Build a one-shot solver seeded from the given config so the solution
        # stays close to the current arm pose (avoids wild joint trajectories).
        limits = torch.tensor(np.stack([lo_limits, hi_limits], axis=1), dtype=torch.float32)
        seed_t = torch.tensor(seed, dtype=torch.float32).unsqueeze(0)  # (1, DOF)
        seeded_solver = pk.PseudoInverseIKWithSVD(
            chain, max_iterations=200, num_retries=1,
            joint_limits=limits, retry_configs=seed_t,
        )
        sol = seeded_solver.solve(goal)
    else:
        sol = ik_solver.solve(goal)
    if not sol.converged.any():
        return None
    best_idx = int(sol.converged[0].float().argmax())
    angles = sol.solutions[0, best_idx].numpy()
    if np.linalg.norm(_fk(chain, angles) - target_in_torso) > fk_tol:
        return None
    return angles


def _world_to_torso(torso_pose: sapien.Pose, world_pos: np.ndarray) -> np.ndarray:
    """Transform a world-frame position into the torso_lift_link frame."""
    p = sapien.Pose(p=world_pos.tolist())
    in_torso = torso_pose.inv() * p
    return np.array(in_torso.p)


def _set_robot_pose(robot, base_x: float, base_y: float, yaw: float,
                    torso_height: float, arm_angles: np.ndarray,
                    gripper_open: float = GRIPPER_OPEN):
    """Set full robot configuration via qpos and drive targets."""
    qpos = robot.get_qpos().copy()
    qpos[QPOS_BASE_X]   = base_x
    qpos[QPOS_BASE_Y]   = base_y
    qpos[QPOS_BASE_YAW] = yaw
    qpos[QPOS_TORSO]    = torso_height
    for i, ai in enumerate(QPOS_ARM):
        qpos[ai] = arm_angles[i]
    for fi in QPOS_FINGERS:
        qpos[fi] = gripper_open
    robot.set_qpos(qpos)
    # Set drive targets so PD controllers hold the pose during scene.step()
    active = robot.get_active_joints()
    for i, j in enumerate(active):
        j.set_drive_target(float(qpos[i]))


def _get_ee_pose(robot) -> sapien.Pose:
    """Return the EE (gripper_link) world pose."""
    for l in robot.get_links():
        if l.name == "gripper_link":
            return l.entity_pose
    raise RuntimeError("gripper_link not found")


def _get_torso_pose(robot) -> sapien.Pose:
    """Return the torso_lift_link world pose."""
    for l in robot.get_links():
        if l.name == "torso_lift_link":
            return l.entity_pose
    raise RuntimeError("torso_lift_link not found")


def _save_gif(frames: list, path: str) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0)


def _sapien_lookat_pose(eye: np.ndarray, target: np.ndarray) -> sapien.Pose:
    """Build a SAPIEN Pose for a camera at `eye` looking at `target`.

    SAPIEN camera convention: +X forward, +Y right (image right), +Z up (image top).
    R columns: [fwd, img_right, img_up] where img_right = cross(img_up, fwd).
    """
    _Rot = _Rotation

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    img_up = world_up - np.dot(world_up, fwd) * fwd
    img_up /= np.linalg.norm(img_up)
    img_right = np.cross(img_up, fwd)
    img_right /= np.linalg.norm(img_right)
    R = np.stack([fwd, img_right, img_up], axis=1)
    q_xyzw = _Rotation.from_matrix(R).as_quat()
    return sapien.Pose(
        p=eye.tolist(),
        q=[float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])],
    )


def _make_sapien_camera(scene: sapien.Scene, obj_pos: np.ndarray,
                        width: int = 1280, height: int = 960,
                        robot_dir: np.ndarray = None):
    """Add a RenderCameraComponent looking at the grasp scene.

    Camera is placed 45° azimuth from the direction opposite to the robot,
    at 30° elevation, looking at the object.
    Returns (cam_entity, cam_component). Caller must remove cam_entity when done.
    """
    import sapien.render as sr

    if robot_dir is None:
        robot_dir = np.array([0.0, 1.0, 0.0])
    robot_dir_xy = robot_dir.copy(); robot_dir_xy[2] = 0.0
    norm = np.linalg.norm(robot_dir_xy)
    if norm > 1e-6:
        robot_dir_xy /= norm

    # Rotate the opposite-robot direction by 45° around Z to get a diagonal view
    opp = -robot_dir_xy
    c45, s45 = math.cos(math.radians(45)), math.sin(math.radians(45))
    cam_dir = np.array([c45 * opp[0] - s45 * opp[1],
                        s45 * opp[0] + c45 * opp[1], 0.0])

    # Spherical position: 30° elevation, 1.5 m radial distance
    total_dist = 1.5
    elev = math.radians(30)
    horiz = total_dist * math.cos(elev)
    vert  = total_dist * math.sin(elev)
    eye    = obj_pos + cam_dir * horiz + np.array([0.0, 0.0, vert])
    target = obj_pos.copy()
    cam_pose = _sapien_lookat_pose(eye, target)

    cam_entity = sapien.Entity()
    cam = sr.RenderCameraComponent(width, height)
    cam.set_fovy(np.deg2rad(60))
    cam.set_near(0.01)
    cam.set_far(20.0)
    cam_entity.add_component(cam)
    scene.add_entity(cam_entity)
    cam_entity.set_pose(cam_pose)
    return cam_entity, cam


def _capture_sapien_frame(scene: sapien.Scene, cam) -> "Image":
    from PIL import Image
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb  = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _grasp_frame_zup(p1, p2):
    """Return (M, x_grip, y_grip, z_grip) for antipodal pair in Z-up convention.

    x_grip = squeeze axis (unit vector along finger contact line)
    z_grip = upward direction (perpendicular to squeeze axis, roughly +Z)
    y_grip = x_grip × z_grip
    """
    M = (np.array(p1) + np.array(p2)) / 2
    x_grip = np.array(p2) - np.array(p1)
    x_grip /= np.linalg.norm(x_grip)
    world_up = np.array([0.0, 0.0, 1.0])   # Z-up
    z_grip = world_up - np.dot(world_up, x_grip) * x_grip
    z_n = np.linalg.norm(z_grip)
    z_grip = z_grip / z_n if z_n > 1e-6 else np.array([0.0, 0.0, 1.0])
    y_grip = np.cross(z_grip, x_grip)
    y_grip /= np.linalg.norm(y_grip)
    return M, x_grip, y_grip, z_grip


def _settle(robot, scene: sapien.Scene,
            max_steps: int = 500, vel_threshold: float = 0.01,
            cam=None, frames: list = None, capture_every: int = 4) -> int:
    """Step until arm joint velocities drop below threshold or max_steps is reached.

    Returns the number of steps taken.
    """
    arm_indices = QPOS_ARM
    for step in range(max_steps):
        scene.step()
        if cam is not None and frames is not None and step % capture_every == 0:
            frames.append(_capture_sapien_frame(scene, cam))
        qvel = robot.get_qvel()
        if np.max(np.abs(qvel[arm_indices])) < vel_threshold:
            return step + 1
    return max_steps


def _drive_to(robot, scene: sapien.Scene, base_x, base_y, base_yaw,
              torso_height, arm_angles, gripper_val,
              n_steps: int = 120, settle: bool = True,
              max_settle_steps: int = 500, vel_threshold: float = 0.01,
              cam=None, frames: list = None, capture_every: int = 4,
              kinematic: bool = False):
    """Ramp drive targets from current qpos to target over n_steps, then settle.

    If settle=True, continues stepping after the ramp until arm joint velocities
    fall below vel_threshold (or max_settle_steps is exhausted).
    """
    q0 = robot.get_qpos().copy()
    q_target = q0.copy()
    q_target[QPOS_BASE_X]   = base_x
    q_target[QPOS_BASE_Y]   = base_y
    q_target[QPOS_BASE_YAW] = base_yaw
    q_target[QPOS_TORSO]    = torso_height
    for i, ai in enumerate(QPOS_ARM):
        q_target[ai] = arm_angles[i]
    for fi in QPOS_FINGERS:
        q_target[fi] = gripper_val

    active = robot.get_active_joints()
    for step in range(n_steps):
        t = (step + 1) / n_steps
        q_interp = q0 + t * (q_target - q0)
        if kinematic:
            robot.set_qpos(q_interp)
            for i, j in enumerate(active):
                j.set_drive_target(float(q_interp[i]))
        else:
            for i, j in enumerate(active):
                j.set_drive_target(float(q_interp[i]))
        scene.step()
        if cam is not None and frames is not None and step % capture_every == 0:
            frames.append(_capture_sapien_frame(scene, cam))

    if settle:
        _settle(robot, scene, max_steps=max_settle_steps,
                vel_threshold=vel_threshold, cam=cam, frames=frames,
                capture_every=capture_every)


def _run_snap_trial_ms(
    scene: sapien.Scene,
    robot,
    chain,
    ik_solver,
    config_json_path: str,
    collision_mode: str,
    candidate,
    table_top_z: float,
    capture: bool = False,
) -> dict:
    """Single physics-based grasp trial in SAPIEN Z-up convention.

    Pipeline: snap object to table → arm approaches (kinematic IK) →
    gripper closes (physics) → lift (physics) → release → check z_rise.
    """
    cand, M, x_grip, z_grip = candidate
    fail = {"success": False, "ee_dist_m": None, "ik_error": None,
            "frames": [], "obj_y_rise": 0.0, "grasp_close_idx": None}

    table_entity = None
    obj_entity   = None
    cam_entity   = None

    try:
        table_entity, _ = add_table(scene)

        _pipe_r, _pipe_w = os.pipe()
        _old_stderr = os.dup(2)
        os.dup2(_pipe_w, 2)
        os.close(_pipe_w)
        try:
            obj_entity = load_object(scene, config_json_path, collision_mode=collision_mode,
                                     name="grasp_obj")
        finally:
            os.dup2(_old_stderr, 2)
            os.close(_old_stderr)
            _sapien_log = os.read(_pipe_r, 65536).decode(errors="replace")
            os.close(_pipe_r)

        if "failed to load a component" in _sapien_log:
            raise RuntimeError(f"SAPIEN could not load collision mesh: {_sapien_log[:300].strip()}")

        rb = get_rb(obj_entity)

        # ── Orient object so gripper squeeze axis aligns with finger-opening direction ──
        # The Fetch gripper opens along gripper_link Y-axis (world +X when yaw=-π/2).
        # To bring x_grip into alignment with +X we rotate the object by R_z(-theta_z),
        # which maps (cos θ, sin θ) → (1, 0) in the XY plane.
        theta_z  = math.atan2(float(x_grip[1]), float(x_grip[0]))
        R_z      = np.array([[ math.cos(theta_z), math.sin(theta_z), 0],
                              [-math.sin(theta_z), math.cos(theta_z), 0],
                              [0, 0, 1]], dtype=float)
        rot_quat = sapien.Pose(q=_euler_z_to_quat(-theta_z)).q

        # ── Snap object onto table ────────────────────────────────────────────
        _set_robot_pose(robot, 50.0, 0.0, 0.0, 0.0, ARM_INIT)
        obj_entity.set_pose(sapien.Pose(p=[0, 0, table_top_z + 0.5], q=rot_quat))
        if not snap_down(scene, obj_entity, [table_entity]):
            return fail

        settled_t = np.array(obj_entity.get_pose().p)
        world_M   = settled_t + R_z @ np.array(M)
        rb.set_kinematic(True)
        obj_entity.set_pose(sapien.Pose(p=settled_t.tolist(), q=rot_quat))

        wp_grasp    = world_M.copy()
        wp_pregrasp = wp_grasp + np.array([0.0, 0.0, 0.20])

        # ── Robot base placement ──────────────────────────────────────────────
        base_x   = float(settled_t[0]) + APPROACH_DIST * math.sin(0.0)
        base_y   = float(settled_t[1]) + APPROACH_DIST * math.cos(0.0)
        base_yaw = float(math.atan2(float(settled_t[1]) - base_y,
                                    float(settled_t[0]) - base_x))

        # ── IK: search torso heights for pregrasp and grasp ──────────────────
        def _solve_at_torso(target_world, torso_height):
            _set_robot_pose(robot, base_x, base_y, base_yaw, torso_height,
                            ARM_INIT, GRIPPER_OPEN)
            t_pose = _get_torso_pose(robot)
            return _solve_ik(chain, ik_solver, None, None,
                             _world_to_torso(t_pose, target_world)), t_pose

        angles_pre = None
        best_torso = TORSO_HEIGHT
        for th in TORSO_SEARCH:
            a, _ = _solve_at_torso(wp_pregrasp, th)
            if a is not None:
                angles_pre = a
                best_torso = th
                break
        if angles_pre is None:
            angles_pre = ARM_INIT.copy()

        angles_grasp = None
        ik_err       = float("inf")
        for th in TORSO_SEARCH:
            a, t_pose = _solve_at_torso(wp_grasp, th)
            if a is not None:
                angles_grasp = a
                best_torso   = th
                ee_world = np.array((t_pose * sapien.Pose(
                    p=_fk(chain, a).tolist())).p)
                ik_err = float(np.linalg.norm(ee_world - wp_grasp))
                break

        if angles_grasp is None:
            return {**fail, "ik_error": round(ik_err, 4)}

        # ── Lift IK: raise torso first, then arm IK for the remainder ─────────
        torso_raise   = min(LIFT_HEIGHT, TORSO_MAX - best_torso)
        arm_lift_rem  = LIFT_HEIGHT - torso_raise
        lift_torso    = best_torso + torso_raise
        wp_lifted     = wp_grasp + np.array([0.0, 0.0, arm_lift_rem])
        angles_lifted = angles_grasp.copy()
        if arm_lift_rem > 0.01:
            a, _ = _solve_at_torso(wp_lifted, lift_torso)
            if a is not None:
                angles_lifted = a

        # ── Kinematic EE-distance check ───────────────────────────────────────
        _set_robot_pose(robot, base_x, base_y, base_yaw, best_torso,
                        angles_grasp, GRIPPER_OPEN)
        ee_pos  = np.array(_get_ee_pose(robot).p)
        ee_dist = float(np.linalg.norm(ee_pos - settled_t))

        # ── Camera setup ──────────────────────────────────────────────────────
        frames = []
        grasp_close_idx = 0
        cam = None
        if capture:
            robot_dir = np.array([base_x - float(settled_t[0]),
                                  base_y - float(settled_t[1]), 0.0])
            cam_entity, cam = _make_sapien_camera(scene, wp_grasp, robot_dir=robot_dir)

        def _maybe_capture(step, every=3):
            if cam is not None and step % every == 0:
                frames.append(_capture_sapien_frame(scene, cam))

        # ── Phase 0: teleport arm to pregrasp (kinematic) ─────────────────────
        _set_robot_pose(robot, base_x, base_y, base_yaw, best_torso,
                        angles_pre, GRIPPER_OPEN)
        for step in range(APPROACH_STEPS):
            scene.step()
            _maybe_capture(step)

        # ── Phase 1: kinematic descent to grasp pose ──────────────────────────
        # Object stays kinematic; arm is teleported directly to grasp config
        # each step, bypassing PhysX push-back.  After this phase the arm is
        # at world_M with fingers positioned at the contact points.
        _drive_to(robot, scene, base_x, base_y, base_yaw, best_torso,
                  angles_grasp, GRIPPER_OPEN, APPROACH_STEPS,
                  cam=cam, frames=frames, kinematic=True)

        # ── Phase 2: kinematic gripper close ─────────────────────────────────
        # Close fingers kinematically so they are at the contact points before
        # any physics release.  A few simulation steps let the renderer settle.
        grasp_close_idx = len(frames)
        _drive_to(robot, scene, base_x, base_y, base_yaw, best_torso,
                  angles_grasp, GRIPPER_CLOSED, SNAP_HOLD_STEPS,
                  cam=cam, frames=frames, kinematic=True)

        # ── Phase 3: kinematic lift with object tracking ──────────────────────
        # Object remains kinematic and is moved to track EE displacement so the
        # GIF shows the object being lifted.  This tests IK reachability for the
        # lift config, which is the key graspability signal.
        ee_at_grasp  = np.array(_get_ee_pose(robot).p)
        obj_at_grasp = np.array(obj_entity.get_pose().p)

        _drive_to(robot, scene, base_x, base_y, base_yaw, lift_torso,
                  angles_lifted, GRIPPER_CLOSED, SNAP_LIFT_STEPS,
                  cam=cam, frames=frames, kinematic=True)

        ee_at_lift   = np.array(_get_ee_pose(robot).p)
        ee_delta     = ee_at_lift - ee_at_grasp
        obj_at_lift  = obj_at_grasp + ee_delta
        obj_entity.set_pose(sapien.Pose(p=obj_at_lift.tolist(), q=rot_quat))

        for step in range(SNAP_HOLD_STEPS):
            scene.step()
            _maybe_capture(step)

        obj_z_rise = float(ee_delta[2])

        # ── Phase 4: release (kinematic open + lower object back) ────────────
        _drive_to(robot, scene, base_x, base_y, base_yaw, lift_torso,
                  angles_lifted, GRIPPER_OPEN, SNAP_RELEASE_STEPS,
                  cam=cam, frames=frames, kinematic=True)

        success = obj_z_rise >= LIFT_THRESHOLD

        return {"success": success, "ee_dist_m": round(ee_dist, 4),
                "ik_error": round(ik_err, 4), "frames": frames,
                "obj_y_rise": round(obj_z_rise, 4),
                "grasp_close_idx": grasp_close_idx}

    finally:
        if cam_entity is not None:
            try:
                scene.remove_entity(cam_entity)
            except Exception:
                pass
        if obj_entity is not None:
            try:
                scene.remove_actor(obj_entity)
            except Exception:
                pass
        if table_entity is not None:
            try:
                scene.remove_actor(table_entity)
            except Exception:
                pass


def _euler_z_to_quat(theta: float):
    """Convert Z-axis rotation angle to quaternion [w, x, y, z] (SAPIEN convention)."""
    half = theta / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


# Offset from base_link to torso_lift_joint origin (from fetch.urdf torso_lift_joint)
_TORSO_ORIGIN_IN_BASE = np.array([-0.086875, 0.0, 0.37743], dtype=np.float64)


def _compute_torso_pose(base_x: float, base_y: float, base_yaw: float,
                        torso_height: float) -> sapien.Pose:
    """Compute torso_lift_link world pose analytically from joint values.

    Avoids GPU link buffer reads (which are unreliable mid-simulation).
    Torso prismatic joint moves along global Z (base yaw doesn't tilt z-axis).
    """
    cy, sy = math.cos(base_yaw), math.sin(base_yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    base_pos = np.array([base_x, base_y, 0.0], dtype=np.float64)
    torso_pos = base_pos + Rz @ _TORSO_ORIGIN_IN_BASE + np.array([0.0, 0.0, torso_height])
    q = _euler_z_to_quat(base_yaw)
    return sapien.Pose(p=torso_pos.tolist(), q=q)


def _save_all_candidates_html(path, mesh, cand_list, settled_t, R_z, R_orient,
                               rejected_candidates=None):
    """Save one HTML showing selected and rejected antipodal candidate pairs.

    Selected pairs are each shown in a distinct color.
    Rejected pairs are shown as a single togglable group (hidden by default).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for HTML saving: pip install plotly")

    R_world = R_z.T @ R_orient
    verts_world = (R_world @ np.array(mesh.vertices).T).T + settled_t
    faces = np.array(mesh.faces, dtype=int)

    COLORS = [
        "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
        "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
        "#dcbeff","#9A6324","#800000","#aaffc3","#808000",
        "#ffd8b1","#000075","#a9a9a9","#e6beff","#fffac8",
    ]

    traces = []
    traces.append(go.Mesh3d(
        x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightblue', opacity=0.3, name='Object mesh',
        legendgroup='mesh', showlegend=True,
    ))

    axis_len = 0.04
    for idx, (cand, M_a, x_a, z_a) in enumerate(cand_list):
        color = COLORS[idx % len(COLORS)]
        p1_local, p2_local = np.array(cand[1]), np.array(cand[2])
        if np.linalg.norm(p1_local) < 1e-6 and np.linalg.norm(p2_local) < 1e-6:
            continue
        p1_w = settled_t + R_world @ p1_local
        p2_w = settled_t + R_world @ p2_local
        M_w  = (p1_w + p2_w) / 2
        grp  = f"selected_{idx:02d}"
        traces.append(go.Scatter3d(
            x=[p1_w[0], p2_w[0]], y=[p1_w[1], p2_w[1]], z=[p1_w[2], p2_w[2]],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            name=f"cand {idx:02d}",
            legendgroup=grp, legendgrouptitle=dict(text=f"cand {idx:02d}"),
            showlegend=True,
        ))
        tip = M_w + z_a * axis_len
        traces.append(go.Scatter3d(
            x=[M_w[0], tip[0]], y=[M_w[1], tip[1]], z=[M_w[2], tip[2]],
            mode='lines', line=dict(color=color, width=2, dash='dash'),
            legendgroup=grp, showlegend=False,
        ))

    if rejected_candidates:
        first_rejected = True
        for raw_cand in rejected_candidates:
            _, p1_local, p2_local, _ = raw_cand
            p1_local = R_orient @ np.array(p1_local)
            p2_local = R_orient @ np.array(p2_local)
            p1_w = settled_t + R_z.T @ p1_local
            p2_w = settled_t + R_z.T @ p2_local
            traces.append(go.Scatter3d(
                x=[p1_w[0], p2_w[0]], y=[p1_w[1], p2_w[1]], z=[p1_w[2], p2_w[2]],
                mode='lines+markers',
                line=dict(color='gray', width=1),
                marker=dict(size=3, color='gray'),
                name='rejected' if first_rejected else None,
                legendgroup='rejected',
                legendgrouptitle=dict(text='Rejected') if first_rejected else None,
                showlegend=first_rejected,
                visible='legendonly',
            ))
            first_rejected = False

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=os.path.basename(path),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(groupclick='toggleitem'),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs='cdn')


def _save_grasp_html_gpu(path, mesh, p1_local, p2_local,
                         world_M, x_grip_world, z_grip_world,
                         wp_grasp, wp_pregrasp, settled_t, R_z, R_orient):
    """Save an interactive Plotly HTML showing the grasp geometry for a GPU trial."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for HTML saving: pip install plotly")

    # Full object world rotation: first apply up-axis orient, then -theta_z (= R_z.T)
    R_world = R_z.T @ R_orient

    p1_w = settled_t + R_world @ np.array(p1_local)
    p2_w = settled_t + R_world @ np.array(p2_local)

    verts = np.array(mesh.vertices)
    verts_world = (R_world @ verts.T).T + settled_t
    faces = np.array(mesh.faces, dtype=int)

    traces = []
    traces.append(go.Mesh3d(
        x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightblue', opacity=0.4, name='Object mesh', showlegend=True,
    ))
    hw = 0.3
    tx, ty, tz = float(settled_t[0]), float(settled_t[1]), float(verts_world[:, 2].min())
    tv = np.array([[tx-hw, ty-hw, tz], [tx+hw, ty-hw, tz],
                   [tx+hw, ty+hw, tz], [tx-hw, ty+hw, tz]])
    traces.append(go.Mesh3d(
        x=tv[:, 0], y=tv[:, 1], z=tv[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='tan', opacity=0.3, name='Table top', showlegend=True,
    ))
    traces.append(go.Scatter3d(x=[p1_w[0]], y=[p1_w[1]], z=[p1_w[2]],
        mode='markers', marker=dict(size=8, color='red'), name='p1', showlegend=True))
    traces.append(go.Scatter3d(x=[p2_w[0]], y=[p2_w[1]], z=[p2_w[2]],
        mode='markers', marker=dict(size=8, color='darkred'), name='p2', showlegend=True))
    traces.append(go.Scatter3d(
        x=[p1_w[0], p2_w[0]], y=[p1_w[1], p2_w[1]], z=[p1_w[2], p2_w[2]],
        mode='lines', line=dict(color='red', width=3), name='squeeze axis', showlegend=True))
    traces.append(go.Scatter3d(x=[world_M[0]], y=[world_M[1]], z=[world_M[2]],
        mode='markers', marker=dict(size=10, color='gold'), name='M (midpoint)', showlegend=True))
    axis_len = 0.06
    for aname, avec, acolor in [('x_grip', x_grip_world, 'firebrick'),
                                  ('z_grip (approach)', z_grip_world, 'royalblue')]:
        tip = world_M + avec * axis_len
        traces.append(go.Scatter3d(
            x=[world_M[0], tip[0]], y=[world_M[1], tip[1]], z=[world_M[2], tip[2]],
            mode='lines+markers', line=dict(color=acolor, width=4),
            marker=dict(size=[0, 6], color=acolor), name=aname, showlegend=True))
    traces.append(go.Scatter3d(x=[wp_grasp[0]], y=[wp_grasp[1]], z=[wp_grasp[2]],
        mode='markers', marker=dict(size=10, color='purple', symbol='circle'),
        name='EE grasp target', showlegend=True))
    traces.append(go.Scatter3d(x=[wp_pregrasp[0]], y=[wp_pregrasp[1]], z=[wp_pregrasp[2]],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle-open'),
        name='EE pregrasp', showlegend=True))
    traces.append(go.Scatter3d(
        x=[wp_pregrasp[0], wp_grasp[0]], y=[wp_pregrasp[1], wp_grasp[1]],
        z=[wp_pregrasp[2], wp_grasp[2]],
        mode='lines', line=dict(color='purple', width=2, dash='dash'),
        name='approach', showlegend=True))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=os.path.basename(path),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs='cdn')


def _run_physics_trial_gpu(
    config_json_path: str,
    collision_mode: str,
    candidate,
    chain,
    ik_solver,
    lo,
    hi,
    capture: bool = False,
    ray_tracing: bool = False,
    save_html_path: Optional[str] = None,
) -> dict:
    """GPU-physics grasp trial: creates a fresh GPU scene, runs real physics grasping.

    Robot arm physically closes on the dynamic object; success is measured by z-rise.
    """
    import sapien.physx as physx
    import torch

    cand, M, x_grip, z_grip = candidate
    fail = {"success": False, "ee_dist_m": None, "ik_error": None,
            "frames": [], "obj_y_rise": 0.0, "grasp_close_idx": None}

    theta_z  = math.atan2(float(x_grip[1]), float(x_grip[0]))
    R_z      = np.array([[ math.cos(theta_z), math.sin(theta_z), 0],
                          [-math.sin(theta_z), math.cos(theta_z), 0],
                          [0, 0, 1]], dtype=float)

    # load_object already bakes cfg["up"] orient_pose into the shape-local poses,
    # so the entity world pose only needs the gripper-alignment rotation (-theta_z).
    rot_quat = _euler_z_to_quat(-theta_z)

    if capture and ray_tracing:
        import sapien.render as _sr
        _sr.set_camera_shader_dir("rt")
        _sr.set_ray_tracing_samples_per_pixel(256)
        _sr.set_ray_tracing_path_depth(16)
        _sr.set_ray_tracing_denoiser("none")

    scene = make_scene(use_gpu=True, with_renderer=capture)
    px    = scene.physx_system

    robot        = load_fetch_robot(scene, MS_FETCH_URDF_BUNDLED, setup_drives=True)
    # High-friction rubber-like material on gripper fingers to improve grasp holding.
    _finger_mat = physx.PhysxMaterial(static_friction=5.0, dynamic_friction=5.0,
                                      restitution=0.0)
    _finger_link_names = {"r_gripper_finger_link", "l_gripper_finger_link"}
    for link in robot.get_links():
        if link.name in _finger_link_names:
            for shape in link.get_collision_shapes():
                shape.set_physical_material(_finger_mat)

    table_entity, _ = add_table(scene)

    _pipe_r, _pipe_w = os.pipe()
    _old_stderr = os.dup(2)
    os.dup2(_pipe_w, 2)
    os.close(_pipe_w)
    try:
        obj_entity = load_object(scene, config_json_path, collision_mode=collision_mode,
                                 name="grasp_obj")
    finally:
        os.dup2(_old_stderr, 2)
        os.close(_old_stderr)
        _sapien_log = os.read(_pipe_r, 65536).decode(errors="replace")
        os.close(_pipe_r)

    if "failed to load a component" in _sapien_log:
        raise RuntimeError(f"SAPIEN could not load collision mesh: {_sapien_log[:300].strip()}")

    rb = get_rb(obj_entity)
    rb.set_mass(0.5)  # realistic tabletop object mass; default PhysX density yields unrealistic values
    n_dof = len(robot.get_active_joints())

    def _make_qpos(bx, by, byaw, torso, arm, gripper=GRIPPER_OPEN):
        q = np.zeros(n_dof, dtype=np.float32)
        q[QPOS_BASE_X] = bx;  q[QPOS_BASE_Y] = by;  q[QPOS_BASE_YAW] = byaw
        q[QPOS_TORSO]  = torso
        for i, ai in enumerate(QPOS_ARM): q[ai] = arm[i]
        for fi in QPOS_FINGERS:           q[fi] = gripper
        return q

    q_init = _make_qpos(50.0, 0.0, 0.0, TORSO_HEIGHT, ARM_INIT, GRIPPER_OPEN)
    obj_start_pose = sapien.Pose(p=[0.0, 0.0, TABLE_TOP_Z + 0.5], q=rot_quat)

    px = init_gpu_scene(scene, robot, q_init,
                        root_pose=sapien.Pose(),
                        dynamic_entities=[obj_entity],
                        entity_poses=[obj_start_pose])

    if not snap_down_gpu(scene, px, rb):
        return fail

    px.gpu_fetch_rigid_dynamic_data()
    settled_t = np.array(get_rb_pose_gpu(px, rb).p)
    world_M   = settled_t + R_z @ np.array(M)

    BASE_X   = float(settled_t[0]) + APPROACH_DIST * math.sin(0.0)
    BASE_Y   = float(settled_t[1]) + APPROACH_DIST * math.cos(0.0)
    BASE_YAW = float(math.atan2(settled_t[1] - BASE_Y, settled_t[0] - BASE_X))

    # ── Candidate-based grasp target ─────────────────────────────────────────
    # M is the antipodal midpoint in mesh-local space (already rotated by R_orient
    # in run_snap_gpu before being passed in). world_M is its world position after
    # adding the settled object translation.
    # z_grip is the grasp approach axis (up direction of grasp frame, ~world +Z for
    # top-down, but tilted for side/angled grasps). The EE clearance is applied along
    # this axis so the arm approaches from the correct direction regardless of orientation.
    _, x_grip_world, y_grip_world, z_grip_world = _grasp_frame_zup(
        settled_t + R_z @ np.array(cand[1]),
        settled_t + R_z @ np.array(cand[2]),
    ) if (np.linalg.norm(np.array(cand[1])) > 1e-6 and
          np.linalg.norm(np.array(cand[2])) > 1e-6) else (
        world_M, np.array([1.0,0,0]), np.array([0,1.0,0]), np.array([0,0,1.0]))

    px.sync_poses_gpu_to_cpu()
    aabb = rb.get_global_aabb_fast()   # shape [2,3]: [min_xyz, max_xyz]

    # Grasp target: candidate midpoint, with EE clearance along approach axis.
    # For degenerate (fallback) candidates M==0: use AABB top-center instead.
    if np.linalg.norm(world_M - settled_t) < 1e-4:
        top_z    = float(aabb[1, 2])
        world_M  = np.array([settled_t[0], settled_t[1], top_z])
        z_grip_world = np.array([0.0, 0.0, 1.0])

    wp_grasp    = world_M + z_grip_world * GRIPPER_Z_OFFSET
    wp_pregrasp = world_M + z_grip_world * 0.25
    print(f"[targets] settled=({settled_t[0]:.3f},{settled_t[1]:.3f},{settled_t[2]:.3f}) "
          f"aabb_z=[{aabb[0,2]:.3f},{aabb[1,2]:.3f}] "
          f"world_M=({world_M[0]:.3f},{world_M[1]:.3f},{world_M[2]:.3f}) "
          f"wp_grasp=({wp_grasp[0]:.3f},{wp_grasp[1]:.3f},{wp_grasp[2]:.3f}) "
          f"BASE_Y={BASE_Y:.3f}")

    # Gripper orientation: X-axis (approach) = -z_grip_world, Y-axis = x_grip_world (squeeze).
    # Build rotation matrix columns [X_gripper, Y_gripper, Z_gripper] in torso frame.
    def _R_grasp(base_yaw):
        t_pose = _compute_torso_pose(BASE_X, BASE_Y, base_yaw, 0.0)
        R_torso_inv = np.array(t_pose.inv().to_transformation_matrix()[:3, :3])
        x_col = R_torso_inv @ (-z_grip_world)          # gripper approach axis
        y_col = R_torso_inv @ x_grip_world             # finger squeeze axis
        z_col = np.cross(x_col, y_col)
        z_col /= max(np.linalg.norm(z_col), 1e-9)
        y_col = np.cross(z_col, x_col)
        y_col /= max(np.linalg.norm(y_col), 1e-9)
        return np.column_stack([x_col, y_col, z_col])

    def _solve_at_torso_gpu(target_world, torso_height, seed=None):
        t_pose = _compute_torso_pose(BASE_X, BASE_Y, BASE_YAW, torso_height)
        return _solve_ik(chain, ik_solver, lo, hi,
                         _world_to_torso(t_pose, target_world),
                         rot=_R_grasp(BASE_YAW), seed=seed), t_pose

    # ── IK: find best torso height for grasp first, then solve pregrasp at same torso ──
    # Torso is set to best_torso before arm moves; pregrasp and grasp share that height.
    angles_grasp = None
    best_torso   = TORSO_HEIGHT
    ik_err       = float("inf")
    for th in TORSO_SEARCH:
        a, t_pose = _solve_at_torso_gpu(wp_grasp, th, seed=ARM_INIT)
        if a is not None:
            angles_grasp = a
            best_torso   = th
            ee_world = np.array((t_pose * sapien.Pose(p=_fk(chain, a).tolist())).p)
            ik_err   = float(np.linalg.norm(ee_world - wp_grasp))
            break

    if angles_grasp is None:
        return {**fail, "ik_error": round(ik_err, 4)}

    # Rotate wrist roll 90° on the grasp configuration (arm index 6 = wrist_roll)
    angles_grasp[6] += math.pi / 2.0

    # Pregrasp: search torso heights (pregrasp is higher so may need lower torso)
    angles_pre  = None
    best_torso_pre = best_torso
    for th in TORSO_SEARCH:
        a = _solve_ik(chain, ik_solver, lo, hi,
                      _world_to_torso(_compute_torso_pose(BASE_X, BASE_Y, BASE_YAW, th),
                                      wp_pregrasp),
                      rot=_R_grasp(BASE_YAW), seed=angles_grasp)
        if a is not None:
            angles_pre     = a
            best_torso_pre = th
            break
    if angles_pre is None:
        print(f"[WARN] pregrasp IK failed for all torso heights — using grasp angles fallback")
        angles_pre     = angles_grasp.copy()
        best_torso_pre = best_torso
    angles_pre = _unwrap_to_reference(angles_pre, angles_grasp)

    torso_raise  = min(LIFT_HEIGHT, TORSO_MAX - best_torso)
    arm_lift_rem = LIFT_HEIGHT - torso_raise
    lift_torso   = best_torso + torso_raise
    wp_lifted    = wp_grasp + np.array([0.0, 0.0, arm_lift_rem])  # always lift along world Z
    angles_lifted = angles_grasp.copy()
    if arm_lift_rem > 0.01:
        a, _ = _solve_at_torso_gpu(wp_lifted, lift_torso, seed=angles_grasp)
        if a is not None:
            angles_lifted = _unwrap_to_reference(a, angles_grasp)

    # q_setup: torso at best_torso_pre (pregrasp height), arm in ARM_INIT
    q_setup    = _make_qpos(BASE_X, BASE_Y, BASE_YAW, best_torso_pre, ARM_INIT,     GRIPPER_OPEN)
    q_pregrasp = _make_qpos(BASE_X, BASE_Y, BASE_YAW, best_torso_pre, angles_pre,   GRIPPER_OPEN)
    q_grasp    = _make_qpos(BASE_X, BASE_Y, BASE_YAW, best_torso,     angles_grasp, GRIPPER_OPEN)
    q_closed   = _make_qpos(BASE_X, BASE_Y, BASE_YAW, best_torso,  angles_grasp, GRIPPER_CLOSED)
    q_lifted   = _make_qpos(BASE_X, BASE_Y, BASE_YAW, lift_torso,  angles_lifted, GRIPPER_CLOSED)

    t_pose_grasp = _compute_torso_pose(BASE_X, BASE_Y, BASE_YAW, best_torso)
    ee_pos  = np.array((t_pose_grasp * sapien.Pose(p=_fk(chain, angles_grasp).tolist())).p)
    ee_dist = float(np.linalg.norm(ee_pos - settled_t))

    # ── Camera setup ──────────────────────────────────────────────────────────
    frames = []
    cam_entity = None
    cam = None
    if capture:
        robot_dir = np.array([BASE_X - float(settled_t[0]),
                              BASE_Y - float(settled_t[1]), 0.0])
        cam_entity, cam = _make_sapien_camera(scene, wp_grasp, robot_dir=robot_dir)

    _capture_counter = [0]

    def _maybe_capture():
        if cam is not None:
            _capture_counter[0] += 1
            if _capture_counter[0] % 2 != 0:  # capture every 2 sim steps → 30 fps at 60 Hz
                return
            try:
                px.sync_poses_gpu_to_cpu()
                frames.append(_capture_sapien_frame(scene, cam))
            except Exception as _ce:
                print(f"[capture error at frame {len(frames)}] {type(_ce).__name__}: {_ce}")

    art_idx = robot.gpu_index

    def _current_qpos():
        px.gpu_fetch_articulation_qpos()
        return px.cuda_articulation_qpos.torch()[art_idx].cpu().numpy()

    def _ee_world_pos(arm_angles, torso_h):
        t_pose = _compute_torso_pose(BASE_X, BASE_Y, BASE_YAW, torso_h)
        return np.array((t_pose * sapien.Pose(p=_fk(chain, arm_angles).tolist())).p)

    def _debug_phase(label, q_target):
        q_actual = _current_qpos()
        arm_actual = q_actual[np.array(QPOS_ARM)]
        torso_actual = float(q_actual[QPOS_TORSO])
        ee_actual = _ee_world_pos(arm_actual, torso_actual)
        arm_target = q_target[np.array(QPOS_ARM)]
        torso_target = float(q_target[QPOS_TORSO])
        ee_target = _ee_world_pos(arm_target, torso_target)
        err = np.max(np.abs(q_actual[:len(q_target)] - q_target))
        print(f"[{label}] "
              f"setpoint xyz=({ee_target[0]:.3f},{ee_target[1]:.3f},{ee_target[2]:.3f}) "
              f"actual xyz=({ee_actual[0]:.3f},{ee_actual[1]:.3f},{ee_actual[2]:.3f}) "
              f"joint_err={err:.4f}  torso: target={torso_target:.3f} actual={torso_actual:.3f}")

    # Indices of finger joints within the qpos vector — excluded from convergence checks
    # because a grasped object physically prevents them from reaching their targets.
    _finger_idx = set(QPOS_FINGERS)

    def _drive_to(q_target, label="", tol=DRIVE_TOL, max_steps=MAX_DRIVE_STEPS):
        """Slowly ramp drive target toward q_target, then wait for actual joints to converge."""
        n = len(q_target)
        non_finger = [i for i in range(n) if i not in _finger_idx]
        q_ramp = _current_qpos().copy()
        # Phase A: ramp the drive target slowly toward q_target
        for _ in range(max_steps):
            delta = q_target - q_ramp[:n]
            if np.max(np.abs(delta)) < tol:
                break
            step_frac = min(1.0, DRIVE_SPEED / np.max(np.abs(delta)))
            q_ramp[:n] += step_frac * delta
            set_drive_targets_gpu(px, robot, q_ramp)
            px.gpu_apply_articulation_target_position()
            scene.step()
            _maybe_capture()
        # Phase B: hold final target until non-finger joints converge
        set_drive_targets_gpu(px, robot, q_target)
        px.gpu_apply_articulation_target_position()
        for _ in range(max_steps):
            scene.step()
            _maybe_capture()
            q_now = _current_qpos()
            if np.max(np.abs(q_now[non_finger] - q_target[non_finger])) < tol:
                break
        if label:
            _debug_phase(label, q_target)

    # ── Phase 0a: drive torso to best_torso, arm stays at ARM_INIT ──────────────
    q_init_at_base = _make_qpos(BASE_X, BASE_Y, BASE_YAW, TORSO_HEIGHT, ARM_INIT, GRIPPER_OPEN)
    set_robot_state_gpu(px, robot, q_init_at_base, root_pose=sapien.Pose(), zero_vel=True)
    apply_all_gpu(px)
    _drive_to(q_setup, label="setup")

    # ── Phase 0b: teleport arm to pregrasp (torso already at best_torso) ─────
    set_robot_state_gpu(px, robot, q_pregrasp, root_pose=sapien.Pose(), zero_vel=True)
    apply_all_gpu(px)
    _debug_phase("pregrasp(teleport)", q_pregrasp)
    _maybe_capture()

    # ── Phase 1: PD drive from pregrasp to grasp ──────────────────────────────
    _drive_to(q_grasp, label="grasp")

    # ── Phase 2: PD drive gripper close ──────────────────────────────────────
    # Fingers stop when blocked by the object — don't wait for impossible convergence.
    grasp_close_idx = len(frames)
    set_drive_targets_gpu(px, robot, q_closed)
    px.gpu_apply_articulation_target_position()
    for _ in range(SNAP_HOLD_STEPS):
        scene.step()
        _maybe_capture()
    _debug_phase("closed", q_closed)

    # ── Phase 3: PD drive lift ────────────────────────────────────────────────
    _drive_to(q_lifted, label="lifted")

    # ── Hold lifted position for 2 seconds ───────────────────────────────────
    for _ in range(120):
        scene.step()
        _maybe_capture()

    # ── Measure z rise while still grasped ───────────────────────────────────
    px.gpu_fetch_rigid_dynamic_data()
    obj_z_rise = float(get_rb_pose_gpu(px, rb).p[2] - settled_t[2])

    # ── Phase 4: release — open gripper and let object drop ──────────────────
    q_released = _make_qpos(BASE_X, BASE_Y, BASE_YAW, lift_torso, angles_lifted, GRIPPER_OPEN)
    _drive_to(q_released, label="released")
    for _ in range(60):
        scene.step()
        _maybe_capture()

    # ── Success: object was lifted and then released ───────────────────────────
    success    = obj_z_rise >= LIFT_THRESHOLD

    if save_html_path is not None:
        try:
            cfg_html = _json_mod.loads(Path(config_json_path).read_text())
            config_dir_html = Path(config_json_path).parent
            coll_path_html = str((config_dir_html / cfg_html["collision_asset"]).resolve())
            if collision_mode == "vhacd":
                vhacd_html = coll_path_html.replace(".glb", ".vhacd.glb")
                if os.path.exists(vhacd_html):
                    coll_path_html = vhacd_html
            elif collision_mode == "raw":
                coll_path_html = str((config_dir_html / cfg_html["render_asset"]).resolve())
            mesh_html = _load_glb_mesh(coll_path_html)
            up_vec_html = np.array(cfg_html.get("up", [0.0, 0.0, 1.0]), dtype=float)
            up_vec_html /= np.linalg.norm(up_vec_html)
            if np.allclose(up_vec_html, np.array([0.0, 0.0, 1.0])):
                R_orient_html = np.eye(3)
            else:
                from scipy.spatial.transform import Rotation as _R2
                rot_o, _ = _R2.align_vectors([[0, 0, 1]], [up_vec_html])
                R_orient_html = rot_o.as_matrix()
            cand_orig = cand  # (score, p1_local, p2_local, width)
            _save_grasp_html_gpu(
                save_html_path, mesh_html,
                cand_orig[1], cand_orig[2],
                world_M, x_grip_world, z_grip_world,
                wp_grasp, wp_pregrasp,
                settled_t, R_z, R_orient_html,
            )
        except Exception as _he:
            import traceback; traceback.print_exc()
            print(f"[html] failed: {_he}")
        else:
            print(f"[html] saved: {save_html_path}")

    if cam_entity is not None:
        try:
            scene.remove_entity(cam_entity)
        except Exception:
            pass

    return {"success": success, "ee_dist_m": round(ee_dist, 4),
            "ik_error": round(ik_err, 4), "frames": frames,
            "obj_y_rise": round(obj_z_rise, 4),
            "grasp_close_idx": grasp_close_idx}


def run_snap_gpu(
    config_json_path: str,
    collision_mode: str = "convex_hull",
    save_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    ray_tracing: bool = False,
    save_all_gifs: bool = False,
    save_html: bool = False,
) -> dict:
    """GPU-physics snap-based graspability check.

    Each candidate trial creates a fresh GPU scene for real contact-based grasping.
    Same output schema as run_snap().
    """
    result = {
        "collision_mode":     collision_mode,
        "grasp_success_rate": None,
        "grasp_successes":    None,
        "grasp_trials":       GRASP_CANDIDATES,
        "mean_grasp_width_m": None,
        "snap_rate":          None,
        "mean_ee_dist_m":     None,
        "error":              None,
    }

    try:
        cfg = _json_mod.loads(Path(config_json_path).read_text())
        config_dir  = Path(config_json_path).parent
        render_path = str((config_dir / cfg["render_asset"]).resolve())
        collision_path = str((config_dir / cfg["collision_asset"]).resolve())

        if collision_mode == "vhacd":
            vhacd = render_path.replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd):
                result["error"] = f"vhacd not found: {vhacd}"
                return result
            collision_path = vhacd
        elif collision_mode == "raw":
            collision_path = render_path

        up_vec = np.array(cfg.get("up", [0.0, 0.0, 1.0]), dtype=float)
        up_vec /= np.linalg.norm(up_vec)
        world_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(up_vec, world_up):
            R_orient = np.eye(3)
        else:
            rot_orient, _ = _Rotation.align_vectors([world_up], [up_vec])
            R_orient = rot_orient.as_matrix()

        # ── Sample antipodal candidates from collision mesh ───────────────────
        mesh = None
        try:
            loaded = _load_glb_mesh(collision_path)
            mesh = loaded
            candidates, _, _ = _sample_candidates(mesh, n_samples=500)
        except Exception:
            candidates = []

        if not candidates:
            M_fallback = np.zeros(3)
            fallback   = (1.0, M_fallback, M_fallback, 0.04)
            x_ang      = np.array([1.0, 0.0, 0.0])
            cand_list  = [(fallback, M_fallback, x_ang, M_fallback)]
        else:
            cand_list = []
            for c in candidates[:GRASP_CANDIDATES]:
                _, p1, p2, _ = c
                p1_w = R_orient @ np.array(p1)
                p2_w = R_orient @ np.array(p2)
                M_a, x_a, y_a, z_a = _grasp_frame_zup(p1_w, p2_w)
                cand_list.append((c, np.array(M_a), x_a, z_a))

        chain, ik_solver, lo, hi = _make_ik_solver(MS_FETCH_URDF_ARM_IK)

        all_results = []
        successes   = 0
        ee_dists    = []
        z_rises     = []
        n_trials    = len(cand_list)
        do_capture  = save_dir is not None and asset_id is not None

        for trial_idx, cand_tuple in enumerate(cand_list):
            html_path = None
            if save_html and asset_id is not None:
                asset_dir = os.path.join(save_dir, asset_id) if save_dir else asset_id
                os.makedirs(asset_dir, exist_ok=True)
                html_path = os.path.join(asset_dir, f"cand{trial_idx:02d}.html")
            r = _run_physics_trial_gpu(
                config_json_path, collision_mode, cand_tuple,
                chain, ik_solver, lo, hi,
                capture=do_capture,
                ray_tracing=ray_tracing,
                save_html_path=html_path,
            )
            if save_all_gifs and do_capture and r.get("frames"):
                os.makedirs(os.path.join(save_dir, asset_id), exist_ok=True)
                _save_gif(r["frames"],
                          os.path.join(save_dir, asset_id,
                                       f"cand{trial_idx:02d}.gif"))
            all_results.append(r)
            if r["success"]:
                successes += 1
            if r["ee_dist_m"] is not None:
                ee_dists.append(r["ee_dist_m"])
            if r["obj_y_rise"] is not None:
                z_rises.append(r["obj_y_rise"])

        if save_dir is not None and asset_id is not None and all_results:
            best = _pick_best_trial(all_results)
            if best.get("frames"):
                os.makedirs(save_dir, exist_ok=True)
                _save_gif(best["frames"], os.path.join(save_dir, f"{asset_id}.gif"))

        if save_html and mesh is not None and asset_id is not None:
            asset_dir = os.path.join(save_dir, asset_id) if save_dir else asset_id
            os.makedirs(asset_dir, exist_ok=True)
            overview_path = os.path.join(asset_dir, "candidates_overview.html")
            try:
                _save_all_candidates_html(
                    overview_path, mesh, cand_list,
                    settled_t=np.zeros(3),
                    R_z=np.eye(3),
                    R_orient=R_orient,
                    rejected_candidates=candidates[GRASP_CANDIDATES:] if candidates else None,
                )
                print(f"[html] overview saved: {overview_path}")
            except Exception as _he:
                import traceback; traceback.print_exc()
                print(f"[html] overview failed: {_he}")

        result["grasp_successes"]    = successes
        result["grasp_trials"]       = n_trials
        result["grasp_success_rate"] = round(successes / n_trials, 4) if n_trials else 0.0
        result["snap_rate"]          = round(successes / n_trials, 4) if n_trials else 0.0
        result["mean_ee_dist_m"]     = round(float(np.mean(ee_dists)), 4) if ee_dists else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def run_snap(
    scene: sapien.Scene,
    robot,
    chain,
    ik_solver,
    config_json_path: str,
    collision_mode: str = "convex_hull",
    save_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
) -> dict:
    """Snap-based graspability check (ManiSkill/SAPIEN).

    Same interface and output dict as graspability_check.run_snap().
    """
    result = {
        "collision_mode":     collision_mode,
        "grasp_success_rate": None,
        "grasp_successes":    None,
        "grasp_trials":       GRASP_CANDIDATES,
        "mean_grasp_width_m": None,
        "snap_rate":          None,
        "mean_ee_dist_m":     None,
        "error":              None,
    }

    try:
        cfg = _json_mod.loads(Path(config_json_path).read_text())
        config_dir = Path(config_json_path).parent
        render_path = str((config_dir / cfg["render_asset"]).resolve())
        collision_path = str((config_dir / cfg["collision_asset"]).resolve())


        if collision_mode == "vhacd":
            vhacd = render_path.replace(".glb", ".vhacd.glb")
            if not os.path.exists(vhacd):
                result["error"] = f"vhacd not found: {vhacd}"
                return result
            collision_path = vhacd
        elif collision_mode == "raw":
            collision_path = render_path

        # ── Mesh-space → SAPIEN-world orientation transform ───────────────────
        # _load_glb_mesh returns vertices in GLB/trimesh space (asset-local coords).
        # SAPIEN applies orient_pose (from cfg["up"]) to rotate the mesh so its
        # semantic up axis aligns with world +Z. Grasp points must be transformed
        # by the same rotation before being used as world-space targets.
        up_vec = np.array(cfg.get("up", [0.0, 0.0, 1.0]), dtype=float)
        up_vec /= np.linalg.norm(up_vec)
        world_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(up_vec, world_up):
            R_orient = np.eye(3)
        else:
            rot_orient, _ = _Rotation.align_vectors([world_up], [up_vec])
            R_orient = rot_orient.as_matrix()

        # ── Sample antipodal candidates from collision mesh ───────────────────
        mesh = None
        try:
            loaded = _load_glb_mesh(collision_path)
            mesh = loaded
            candidates, _, _ = _sample_candidates(mesh, n_samples=500)
        except Exception:
            candidates = []

        if not candidates:
            # Fallback: compass-angle approach candidates
            M_fallback    = np.zeros(3)
            x_fallback    = np.array([1.0, 0.0, 0.0])
            fallback_cand = (1.0, M_fallback, M_fallback, 0.04)
            cand_list = []
            for ang in np.linspace(0, 2 * np.pi, GRASP_CANDIDATES, endpoint=False):
                z_ang = np.array([float(np.sin(ang)), float(np.cos(ang)), 0.0])
                cand_list.append((fallback_cand, M_fallback, x_fallback, z_ang))
            cand_list = cand_list[:GRASP_CANDIDATES]
        else:
            cand_list = []
            for c in candidates[:GRASP_CANDIDATES]:
                _, p1, p2, _ = c
                # Transform contact points from mesh-local space to SAPIEN world orientation.
                # Note: in SAPIEN entity.get_pose().p is the actor frame origin = mesh origin,
                # so no CoM offset subtraction is needed (unlike Habitat where translation=CoM).
                p1_w = R_orient @ np.array(p1)
                p2_w = R_orient @ np.array(p2)
                M_a, x_a, y_a, z_a = _grasp_frame_zup(p1_w, p2_w)
                cand_list.append((c, np.array(M_a), x_a, z_a))

        all_results = []
        successes   = 0
        ee_dists    = []
        z_rises     = []
        n_trials    = len(cand_list)

        do_capture = save_dir is not None and asset_id is not None
        for cand, M_a, x_a, z_a in cand_list:
            r = _run_snap_trial_ms(
                scene, robot, chain, ik_solver,
                config_json_path, collision_mode,
                (cand, M_a, x_a, z_a),
                TABLE_TOP_Z,
                capture=do_capture,
            )
            all_results.append(r)
            if r["success"]:
                successes += 1
            if r["ee_dist_m"] is not None:
                ee_dists.append(r["ee_dist_m"])
            if r["obj_y_rise"] is not None:
                z_rises.append(r["obj_y_rise"])

        if save_dir is not None and asset_id is not None and all_results:
            best = _pick_best_trial(all_results)
            if best.get("frames"):
                os.makedirs(save_dir, exist_ok=True)
                _save_gif(best["frames"], os.path.join(save_dir, f"{asset_id}.gif"))

        result["grasp_successes"]    = successes
        result["grasp_trials"]       = n_trials
        result["grasp_success_rate"] = round(successes / n_trials, 4) if n_trials else 0.0
        result["mean_grasp_width_m"] = None
        result["snap_rate"]          = round(successes / n_trials, 4) if n_trials else 0.0
        result["mean_ee_dist_m"]     = round(float(np.mean(ee_dists)), 4) if ee_dists else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"

    return result


if __name__ == "__main__":
    """GPU physics grasp smoke test: uses run_snap_gpu for real contact-based grasping."""
    import glob

    cfg_paths = sorted(glob.glob("data/datasets/amara-spatial-10k/configs/*.object_config.json"))
    cfg_path  = next(p for p in cfg_paths if "CeramicEssential" in p)
    print(f"Object: {cfg_path.split('/')[-1]}")

    result = run_snap_gpu(
        cfg_path,
        collision_mode="vhacd",
        save_dir="/tmp",
        asset_id="grasp_test_gpu",
    )

    print(f"\nResult:")
    print(f"  grasp_success_rate : {result['grasp_success_rate']}")
    print(f"  grasp_successes    : {result['grasp_successes']} / {result['grasp_trials']}")
    print(f"  mean_ee_dist_m     : {result['mean_ee_dist_m']}")
    print(f"  error              : {result['error']}")
    if result.get("error") is None:
        print(f"\nGIF saved to /tmp/grasp_test_gpu.gif")
