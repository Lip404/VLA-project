"""
v0_system1_scripted_rules.py

System-1 execution wrapper for RoboCasa heuristic atomic skills.

Interface (unchanged):
    system1_execute(skill_config, env, max_sim_steps) -> (subtask_done_flag: bool, rgb_image: np.ndarray)

Observation keys follow the verified RoboCasa 1.0.1 schema: camera keys end with
`_image`, their values are plain numpy arrays (never nested dicts), and MuJoCo
renders them vertically flipped, so rows are reversed before use.

Action layout (verified on PandaOmron, env.action_dim == 12, after env.reset()):
    [0:6]  arm OSC_POSE delta [dx, dy, dz, dax, day, daz]
    [6]    gripper (GRIP, -1 open / +1 closed)
    [7:9]  mobile base (LegacyMobileBaseJointVelocityController)
    [10:]  torso (JointPositionController, already at its lower limit at reset)
`OSC_POSE` takes a normalized command in [-1, 1] where 1.0 means `output_max` = 0.05 m,
so metre deltas must be divided by OSC_DELTA_MAX before they are sent.
The OSC delta is expressed in the robot base frame, which on this robot is rotated
relative to the world, so `_action_frame()` measures that rotation at runtime instead
of assuming identity.

A grasp needs the wrist turned a quarter turn (`action[5]` = daz): the fingers only
span ~7.2 cm while the task object is 5.03 cm wide and 13.58 cm long, so the closing
axis has to line up with the short side, and the closed gripper settling at
`robot0_gripper_qpos` ~= +/-0.032 is the sign that the object is actually held.

Registry layout (verified on PickPlaceCounterToCabinet):
    env.objects  -> {"obj", "dist_counter", "dist_cab"}; the single manipulable object
                    is registered under the fixed key "obj". Category names from the
                    task config ("tongs") never appear in the MuJoCo model.
    env.fixtures -> 114 scene fixtures keyed like "cab_1_main_group".
    env.model.mujoco_objects -> the same fixture list, NOT the task objects.
    Entity.body_id is None for everything, so body ids are resolved via root_body.
    The robot has no free joint; `base_pos` and `robot0_base` are stale placeholders,
    so base motion must go through the velocity dims [7:9], never through base_pos.

Door skills resolve the handle position and the door angle by duck-typing several
known RoboCasa / MuJoCo APIs. The first working source is logged; if none works,
`_door_diagnostics()` prints every candidate name so the accessor can be fixed in
one line instead of a refactor.
"""

from __future__ import annotations

import numpy as np

# --- verified obs keys ------------------------------------------------------
CAMERA_LEFT = "robot0_agentview_left_image"
CAMERA_RIGHT = "robot0_agentview_right_image"
CAMERA_WRIST = "robot0_eye_in_hand_image"
EEF_POS = "robot0_eef_pos"

# robosuite gripper convention: -1 fully open, +1 fully closed
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# OSC_POSE output_max from robosuite/controllers/config/robots/default_pandaomron.json
OSC_DELTA_MAX = 0.05

MAX_EEF_DELTA = 0.05    # metres per control step
APPROACH_OFFSET = 0.05  # metres above the object centre before closing; the fingers sit at eef height
CLEAR_HEIGHT = 0.25     # metres above the object centre used while travelling over counter fixtures
LIFT_HEIGHT = 0.15      # metres lifted after a successful grasp
POS_TOL = 0.01          # metres, "target reached" tolerance
APPROACH_ACCEPT = 0.08  # metres, "close enough to grasp" after the fine approach
SETTLE_STEPS = 8        # extra steps holding the gripper closed
CLOSE_STEPS = 25        # closed steps needed before the fingers settle onto the object
MIN_GRASP_RISE = 0.05   # metres the object must rise for a grasp to count

# (daz per step, yaw steps, approach offset). The two fingers only span 7.2 cm, so
# the wrist has to be turned until they straddle the 5 cm width of the object instead
# of its 13.6 cm length. A failed plan is retried with the wrist flipped another
# 180 degrees (2 x 10 steps) and a lower hover height.
GRASP_PLANS = ((0.5, 10, APPROACH_OFFSET), (0.5, 20, 0.02))

NAV_APPROACH_HEIGHT = 0.25  # metres above the target fixture base
NAV_TOL = 0.10              # metres, looser than a grasp target

DOOR_GRASP_OFFSET = 0.02    # metres above the handle when grasping
DOOR_PULL_DIST = 0.20       # metres pulled outward to swing the door
DOOR_OPEN_ANGLE = 0.25      # radians (~14 deg) considered "open"
DOOR_CLOSED_ANGLE = 0.10    # radians considered "closed"
DOOR_HANDLE_MOVE = 0.05     # metres, fallback criterion when no angle is readable
DOOR_DEFAULT_PULL_DIR = np.array([0.0, -1.0, 0.0])  # door faces are usually -y

PLACE_ENTRY_BACK = 0.35     # metres held back from the interior centre before pushing in
PLACE_RETREAT = 0.30        # metres withdrawn after release; env success needs 0.25
PLACE_YAW = 0.5             # daz per step used to line the object up with the opening
PLACE_YAW_STEPS = 10        # steps of PLACE_YAW before the push, about a quarter turn
PLACE_DANGLE_SLACK = 0.0    # metres added to the measured hang when compensating the push height
PLACE_RELEASE_LIFT = 0.04   # metres lifted vertically after release, before retreating
PLACE_DEPTH_FRAC = 0.5      # fraction of the interior depth used as the release point (0.5 = centre)

SUPPORTED_SKILLS = ("PickObject", "PlaceObject", "NavigateTo", "OpenSingleDoor", "CloseSingleDoor")

# RoboCasa registers one manipulable object per task under this fixed key.
TASK_OBJECT_KEY = "obj"

# Planner vocabulary -> RoboCasa fixture registry keys.
_LOC_ALIASES = {
    "cabinet": "cab_1_main_group",
    "counter": "counter_main_group",
    "sink": "sink_1_main_group",
    "fridge": "fridge_1_main_group",
    "dishwasher": "dishwasher_main_group",
}


def read_rgb(obs: dict, camera: str = CAMERA_RIGHT) -> np.ndarray:
    """Return the RGB frame for `camera`, flipped back to display orientation."""
    img = np.asarray(obs[camera])
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"{camera} is not HxWx3: {img.shape}")
    return img[::-1]


def assert_action_layout(env, pose_dims: int = 7) -> int:
    """Validate that the env action layout matches the heuristic controller."""
    if env.action_dim is None:
        raise RuntimeError("env.action_dim is None; call env.reset() before executing skills")
    dim = int(env.action_dim)
    if dim < pose_dims:
        raise RuntimeError(
            f"env.action_dim={dim} < {pose_dims}; expected the OSC delta layout "
            "[dx, dy, dz, dax, day, daz, gripper]"
        )
    return dim


def _build_action(dim: int, delta_pos, gripper: float) -> np.ndarray:
    """Compose a normalized action; OSC_POSE expects [-1, 1] = +/- OSC_DELTA_MAX metres."""
    if dim is None:
        raise RuntimeError("_build_action got dim=None; call env.reset() first")
    action = np.zeros(int(dim), dtype=np.float64)
    vec = np.asarray(delta_pos, dtype=np.float64).reshape(-1)
    vec = np.resize(vec, 3) if vec.size else np.zeros(3, dtype=np.float64)
    action[0:3] = np.clip(vec / OSC_DELTA_MAX, -1.0, 1.0)
    action[6] = float(gripper)
    return action


def _step(env, action) -> tuple:
    """Single env.step() that tolerates both 4- and 5-tuple returns."""
    out = env.step(action)
    obs = out[0]
    done = bool(out[2]) if len(out) > 2 else False
    return obs, done


def _eef_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs[EEF_POS], dtype=np.float64).ravel()[:3]


# ============================== entity lookup ===============================


def _iter_entities(env):
    """Yield (registry_name, entity) for task objects first, then scene fixtures."""
    pool = getattr(env, "objects", None)
    if isinstance(pool, dict):
        for key, item in pool.items():
            yield str(key), item
    elif isinstance(pool, (list, tuple)):
        for item in pool:
            yield str(getattr(item, "name", "") or "object"), item

    fixtures = getattr(env, "fixtures", None)
    if isinstance(fixtures, dict):
        for key, item in fixtures.items():
            yield str(key), item
    elif isinstance(fixtures, (list, tuple)):
        for item in fixtures:
            yield str(getattr(item, "name", "") or "fixture"), item
    else:
        # Fallback for envs that expose fixtures only through the MuJoCo model.
        for item in getattr(getattr(env, "model", None), "mujoco_objects", []) or []:
            yield str(getattr(item, "name", "") or "fixture"), item


def _entity_body_id(env, entity):
    """MuJoCo body id for an entity; RoboCasa leaves `body_id` unset, so use root_body."""
    body_id = getattr(entity, "body_id", None)
    if body_id is not None:
        return int(body_id)
    root_body = getattr(entity, "root_body", None)
    if not root_body:
        return None
    try:
        return int(env.sim.model.body_name2id(root_body))
    except Exception:
        return None


def _body_pos(env, entity) -> np.ndarray | None:
    """World position for an object / fixture, preferring the MuJoCo body pose."""
    if entity is None:
        return None
    body_id = _entity_body_id(env, entity)
    if body_id is not None:
        return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
    pos = getattr(entity, "pos", None)
    if pos is not None:
        return np.asarray(pos, dtype=np.float64).ravel()[:3]
    return None


def _iter_fixtures(env):
    """Return [(registry_name, fixture_object)] for env.fixtures."""
    pool = getattr(env, "fixtures", None)
    if isinstance(pool, dict):
        return [(str(name), fixture) for name, fixture in pool.items()]
    if isinstance(pool, (list, tuple)):
        return [(str(getattr(f, "name", index)), f) for index, f in enumerate(pool)]
    return []


def _find_fixture(env, name):
    """Resolve a fixture by exact -> prefix -> substring match on its registry name."""
    if not name:
        return None, None
    needle = str(name).lower().strip()
    entries = _iter_fixtures(env)
    if not entries:
        print("[system1] env.fixtures is empty; is the fixture registry exposed?")
        return None, None

    alias = _LOC_ALIASES.get(needle)
    matchers = (
        lambda n: n.lower() == needle,
        lambda n: n.lower().startswith(needle),
        lambda n: needle in n.lower(),
    )
    if alias:
        matchers = (lambda n: n.lower() == alias.lower(),) + matchers

    for matcher in matchers:
        group = [(n, f) for n, f in entries if matcher(n)]
        if group:
            if len(group) > 1:
                print(
                    f"[system1] fixture '{name}' matched {len(group)} candidates "
                    f"{[n for n, _ in group]}, using {group[0][0]!r}"
                )
            return group[0][1], group[0][0]

    print(f"[system1] fixture '{name}' not found, available: {[n for n, _ in entries]}")
    return None, None


def _entity_pos_by_name(env, name):
    """(world position, registry name) of the first entity matching `name`."""
    needle = str(name).lower().strip()
    if not needle:
        return None, None

    matched = []
    for label, entity in _iter_entities(env):
        alt = str(getattr(entity, "name", "") or "")
        if needle in label.lower() or needle in alt.lower():
            matched.append((label, entity))

    if not matched:
        return None, None
    if len(matched) > 1:
        print(
            f"[system1] '{name}' matched {len(matched)} entities "
            f"{[label for label, _ in matched]}, using {matched[0][0]!r}"
        )
    label, entity = matched[0]
    return _body_pos(env, entity), label


def _task_object_pos(obs, fallback_env=None, fallback_name=None):
    """World position of the manipulable task object, preferring the observation."""
    if isinstance(obs, dict) and "obj_pos" in obs:
        pos = np.asarray(obs["obj_pos"], dtype=np.float64).ravel()[:3]
        if np.all(np.isfinite(pos)):
            return pos
    if fallback_env is not None and fallback_name:
        return _find_entity_pos(fallback_env, fallback_name)
    return None


def _find_entity_pos(env, name):
    """World position of the object / fixture a subtask refers to."""
    if not name:
        return None

    pos, label = _entity_pos_by_name(env, name)
    if pos is not None:
        return pos
    if label is not None:
        print(f"[system1] entity {label!r} matched but its body pose is unresolved")

    alias = _LOC_ALIASES.get(str(name).lower().strip())
    if alias:
        pos, label = _entity_pos_by_name(env, alias)
        if pos is not None:
            print(f"[system1] '{name}' resolved via alias -> {label!r}")
            return pos

    task_objects = getattr(env, "objects", None)
    if isinstance(task_objects, dict) and TASK_OBJECT_KEY in task_objects:
        print(
            f"[system1] '{name}' is not a registry name; falling back to the task "
            f"object {TASK_OBJECT_KEY!r}"
        )
        return _body_pos(env, task_objects[TASK_OBJECT_KEY])

    available = [label for label, _ in _iter_entities(env)]
    print(f"[system1] entity '{name}' not found, available: {available[:40]}")
    return None


# ============================== motion helpers ==============================


def _action_frame(env, obs, steps=3, probe=0.05, gripper=GRIPPER_OPEN):
    """Measure how a unit eef command maps into world motion (the OSC frame is rotated).

    Returns a 2x2 matrix whose columns are the world xy motion per commanded metre,
    so `pinv(frame) @ world_delta` yields the command to send. `None` means the
    measurement was degenerate and commands should be treated as world deltas.
    `gripper` must stay closed while an object is held, or probing drops it.
    """
    dim = assert_action_layout(env)
    cols = []
    for axis in (0, 1):
        start = _eef_pos(obs)
        cmd = np.zeros(3, dtype=np.float64)
        cmd[axis] = probe
        for _ in range(max(1, steps)):
            obs, _ = _step(env, _build_action(dim, cmd, gripper))
        cols.append((_eef_pos(obs) - start)[:2] / probe)

    frame = np.stack(cols, axis=1)
    if abs(float(np.linalg.det(frame))) < 1e-4:
        print(f"[system1] action frame degenerate: {np.round(frame, 4).tolist()}; using world deltas")
        return None, obs
    print(f"[system1] action frame (world m per command m): {np.round(frame, 3).tolist()}")
    return frame, obs


def _move_to(env, obs, target, gripper, max_steps, tol=POS_TOL, frame=None):
    """Proportional eef move towards `target`; returns (reached, obs)."""
    dim = assert_action_layout(env)
    target = np.asarray(target, dtype=np.float64).ravel()[:3]
    inv = None if frame is None else np.linalg.pinv(np.asarray(frame, dtype=np.float64))

    start = _eef_pos(obs)
    start_dist = float(np.linalg.norm(target - start))
    steps = 0
    for _ in range(max(1, max_steps)):
        err = target - _eef_pos(obs)
        dist = float(np.linalg.norm(err))
        if dist < tol:
            return True, obs
        step_world = err / max(dist, 1e-6) * min(dist, MAX_EEF_DELTA)
        cmd = step_world if inv is None else np.array([*(inv @ step_world[:2]), step_world[2]])
        obs, _ = _step(env, _build_action(dim, cmd, gripper))
        steps += 1

    eef = _eef_pos(obs)
    remaining = float(np.linalg.norm(target - eef))
    travelled = float(np.linalg.norm(eef - start))
    print(
        f"[system1] move timeout: remaining={remaining:.3f} of {start_dist:.3f} m, "
        f"travelled={travelled:.3f} m in {steps} steps "
        f"(needs about {int(np.ceil(start_dist / MAX_EEF_DELTA))} steps)"
    )
    if travelled < 2e-3:
        print(
            "[system1] WARNING: eef did not move; the action vector is not driving "
            f"the controller (action_dim={dim})"
        )
    return False, obs


def _hold_gripper(env, obs, gripper, steps=SETTLE_STEPS):
    dim = assert_action_layout(env)
    for _ in range(steps):
        obs, _ = _step(env, _build_action(dim, np.zeros(3), gripper))
    return obs


def _close_gripper(env, obs, steps=SETTLE_STEPS):
    return _hold_gripper(env, obs, GRIPPER_CLOSE, steps)


def _open_gripper(env, obs, steps=SETTLE_STEPS):
    return _hold_gripper(env, obs, GRIPPER_OPEN, steps)


def _yaw_gripper(env, obs, daz, steps, gripper=GRIPPER_OPEN):
    """Rotate the wrist in place (action[5] = daz) while keeping the translation at zero."""
    dim = assert_action_layout(env)
    for _ in range(max(0, int(steps))):
        action = _build_action(dim, np.zeros(3), gripper)
        action[5] = float(daz)
        obs, _ = _step(env, action)
    return obs


# ============================== door accessors ==============================


def _site_id(sim, name):
    try:
        return int(sim.model.site_name2id(name))
    except Exception:
        return None


def _handle_pos(env, fixture):
    """Handle position: get_handle_pos() -> handle_pos attr -> MuJoCo site."""
    fn = getattr(fixture, "get_handle_pos", None)
    if callable(fn):
        for kwargs in ({}, {"site_name": "handle"}, {"joint_name": "handle"}):
            try:
                out = fn(**kwargs)
            except TypeError:
                continue
            if out is not None:
                return np.asarray(out, dtype=np.float64).ravel()[:3], f"get_handle_pos({kwargs})"

    for attr in ("handle_pos", "door_handle_pos", "get_handle_xpos"):
        value = getattr(fixture, attr, None)
        if value is None:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        return np.asarray(value, dtype=np.float64).ravel()[:3], f"{attr}"

    sim = getattr(env, "sim", None)
    fname = str(getattr(fixture, "name", "") or "")
    if sim is not None and fname:
        for candidate in (f"{fname}_handle", f"handle_{fname}", "handle"):
            sid = _site_id(sim, candidate)
            if sid is not None:
                return np.asarray(sim.data.site_xpos[sid], dtype=np.float64), f"site:{candidate}"
    return None, None


def _door_angle(env, fixture):
    """Door angle in radians: door_angle attr -> get_door_angle() -> hinge joint qpos."""
    for attr in ("door_angle", "get_door_angle"):
        value = getattr(fixture, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                return float(value()), f"{attr}()"
            except Exception:
                continue
        return float(value), attr

    sim = getattr(env, "sim", None)
    fname = str(getattr(fixture, "name", "") or "").lower()
    if sim is None or not fname:
        return None, None

    tokens = [t for t in fname.split("_") if t]
    try:
        for joint_id in range(int(sim.model.njnt)):
            joint_name = sim.model.joint_id2name(joint_id) or ""
            low = joint_name.lower()
            if "door" not in low:
                continue
            if fname in low or any(token in low for token in tokens):
                qpos_addr = int(sim.model.jnt_qposadr[joint_id])
                return float(sim.data.qpos[qpos_addr]), f"joint:{joint_name}"
    except Exception:
        return None, None
    return None, None


def _door_diagnostics(env, fixture) -> str:
    """Human-readable dump used when no handle accessor works."""
    fname = str(getattr(fixture, "name", "") or "?")
    attrs = [a for a in dir(fixture) if any(k in a.lower() for k in ("handle", "door", "pos", "joint"))]
    joints = []
    sim = getattr(env, "sim", None)
    if sim is not None:
        try:
            joints = [
                sim.model.joint_id2name(j) or ""
                for j in range(int(sim.model.njnt))
                if "door" in (sim.model.joint_id2name(j) or "").lower()
            ]
        except Exception:
            joints = []
    return (
        f"[system1] cannot resolve the door handle of fixture {fname!r}. "
        f"candidate attributes: {attrs}. door joints in the model: {joints}. "
        "Add the working accessor in _handle_pos()."
    )


def _outward_dir(point, body):
    """Horizontal unit vector pointing from the fixture body towards the handle."""
    point = np.asarray(point, dtype=np.float64).ravel()[:3]
    if body is None:
        return DOOR_DEFAULT_PULL_DIR.copy()
    offset = point[:2] - np.asarray(body, dtype=np.float64).ravel()[:2]
    norm = float(np.linalg.norm(offset))
    if norm < 1e-6:
        return DOOR_DEFAULT_PULL_DIR.copy()
    return np.array([offset[0] / norm, offset[1] / norm, 0.0])


# ================================ skills ====================================


def _skill_pick(env, obs, skill_config, budget):
    """Try the fixed yaw/height grasp plans in order and keep the first plan that lifts."""
    name = skill_config.get("target_obj")
    obj = _task_object_pos(obs, env, name)
    if obj is None:
        print(f"[system1] PickObject failed: unknown object {name}")
        return False, obs

    start_z = float(obj[2])
    # Hover above the centre by a fraction of the object's own half height: a fixed 5 cm
    # offset leaves the fingers closing in mid-air above a short object.
    half_z = _object_half_z(env)
    cap = None if half_z is None else max(0.01, half_z - 0.01)
    if cap is not None:
        print(f"[system1] PickObject: object half height {half_z:.3f} m, hover cap {cap:.3f} m")

    for daz, yaw_steps, offset in GRASP_PLANS:
        hover = offset if cap is None else min(offset, cap)
        obs = _yaw_gripper(env, obs, daz, yaw_steps)
        frame, obs = _action_frame(env, obs)
        obj = _task_object_pos(obs, env, name)
        if obj is None:
            print("[system1] PickObject failed: lost the object mid attempt")
            return False, obs
        target = obj + np.array([0.0, 0.0, hover])

        # Counters carry fixtures (toaster ovens, microwaves) taller than the object,
        # so travel first at a clearance height, then come straight down onto it.
        clear = obj + np.array([0.0, 0.0, CLEAR_HEIGHT])
        _, obs = _move_to(env, obs, clear, GRIPPER_OPEN, budget // 4, frame=frame)
        reached, obs = _move_to(env, obs, target, GRIPPER_OPEN, budget // 4, frame=frame)
        gap = float(np.linalg.norm(target - _eef_pos(obs)))
        if not reached and gap > APPROACH_ACCEPT:
            print(
                f"[system1] PickObject attempt failed: yaw {daz:+.1f}x{yaw_steps} "
                f"offset {hover:.2f} reach timeout at {gap:.3f} m"
            )
            obs = _open_gripper(env, obs, CLOSE_STEPS)
            continue

        obs = _close_gripper(env, obs, CLOSE_STEPS)
        _, obs = _move_to(
            env, obs, target + np.array([0.0, 0.0, LIFT_HEIGHT]), GRIPPER_CLOSE,
            budget // 4, frame=frame,
        )

        now = _task_object_pos(obs, env, name)
        rise = 0.0 if now is None else float(now[2]) - start_z
        qpos = np.round(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64), 4)
        if rise > MIN_GRASP_RISE:
            print(
                f"[system1] PickObject ok: yaw {daz:+.1f}x{yaw_steps} offset {hover:.2f} "
                f"gap {gap:.3f} rise {rise:+.3f} m qpos {qpos}"
            )
            return True, obs
        print(
            f"[system1] PickObject attempt failed: yaw {daz:+.1f}x{yaw_steps} "
            f"offset {hover:.2f} gap {gap:.3f} rise {rise:+.3f} m qpos {qpos}"
        )
        obs = _open_gripper(env, obs, CLOSE_STEPS)

    print("[system1] PickObject failed: no grasp plan produced a lift")
    return False, obs


def _place_target(env, eef):
    """Interior point of the task cabinet closest to the arm that still clears every face."""
    cab = getattr(env, "cab", None)
    getter = getattr(cab, "get_int_sites", None)
    if cab is None or not callable(getter):
        return None

    half = np.array([0.05, 0.05, 0.05])
    bbox_half = None
    pool = getattr(env, "objects", None)
    obj = pool.get(TASK_OBJECT_KEY) if isinstance(pool, dict) else None
    if obj is not None:
        # obj.size holds the full extents, while get_bbox_points() yields half-extents;
        # halve `size` so the object is placed with its bottom on the shelf.
        size = np.asarray(getattr(obj, "size", []), dtype=np.float64).ravel()
        if size.size >= 3:
            half = np.abs(size[:3]) / 2.0
        try:
            pts = np.asarray(obj.get_bbox_points(), dtype=np.float64).reshape(-1, 3)
            if pts.shape[0] >= 8:
                bbox_half = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
        except Exception:
            bbox_half = None
        if bbox_half is not None and np.all(size.size < 3):
            half = bbox_half

    eef = np.asarray(eef, dtype=np.float64).ravel()[:3]
    for corners in getter(relative=False).values():
        pts = np.stack([np.asarray(c, dtype=np.float64).ravel()[:3] for c in corners])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        margin = half + 0.01
        low, high = lo + margin, hi - margin
        if np.any(high < low):
            margin = (hi - lo) * 0.25
            low, high = lo + margin, hi - margin
            print(f"[system1] PlaceObject: tight region, margin relaxed to {np.round(margin, 3)}")
        print(
            f"[system1] PlaceObject: interior lo={np.round(lo, 3).tolist()} "
            f"hi={np.round(hi, 3).tolist()} half={np.round(half, 3).tolist()} "
            f"bbox_half={None if bbox_half is None else np.round(bbox_half, 3).tolist()}"
        )
        # Aim at the middle of the interior instead of the point nearest the arm: a release
        # point at the front lip leaves the object balanced on the frame edge, and it rolls
        # out and falls once the fingers open.
        xy = low[:2] + (high[:2] - low[:2]) * PLACE_DEPTH_FRAC
        return np.array(
            [
                float(xy[0]),
                float(xy[1]),
                np.clip(lo[2] + float(half[2]) + 0.02, low[2], high[2]),
            ]
        )
    return None


def _held_gap(env, obs, point):
    """Distance from the held object to `point`, or None when the pose is unavailable."""
    pos = _task_object_pos(obs)
    return None if pos is None else float(np.linalg.norm(np.asarray(point, dtype=np.float64) - pos))


def _held_dangle(obs):
    """How far the held object hangs below the eef (positive when the object is lower)."""
    pos = _task_object_pos(obs)
    return None if pos is None else float(_eef_pos(obs)[2] - pos[2])


def _object_half_z(env):
    """Half height of the task object, from `size` (full extents) or the bbox fallback."""
    pool = getattr(env, "objects", None)
    obj = pool.get(TASK_OBJECT_KEY) if isinstance(pool, dict) else None
    if obj is None:
        return None
    size = np.asarray(getattr(obj, "size", []), dtype=np.float64).ravel()
    if size.size >= 3:
        return float(abs(size[2]) / 2.0)
    try:
        pts = np.asarray(obj.get_bbox_points(), dtype=np.float64).reshape(-1, 3)
        return float((pts[:, 2].max() - pts[:, 2].min()) / 2.0)
    except Exception:
        return None


def _skill_place(env, obs, skill_config, budget):
    """Lift the held object to shelf height, slide it in, release it, then retreat."""
    frame, obs = _action_frame(env, obs, gripper=GRIPPER_CLOSE)
    cab = getattr(env, "cab", None)
    if cab is not None:
        try:
            door_state = cab.get_door_state(env)
        except Exception as exc:
            door_state = f"no door api ({type(exc).__name__})"
        print(
            f"[system1] PlaceObject: cab={getattr(cab, 'name', None)} "
            f"type={type(cab).__name__} door={door_state} "
            f"joints={getattr(cab, 'door_joint_names', None)}"
        )

    target = _place_target(env, _eef_pos(obs))
    if target is None:
        print(f"[system1] PlaceObject failed: no usable interior region for {skill_config.get('target_loc')}")
        return False, obs

    eef = _eef_pos(obs)
    dangle = _held_dangle(obs)
    dangle = 0.0 if dangle is None or dangle <= 0.02 else dangle
    push_z = float(target[2]) + dangle + PLACE_DANGLE_SLACK
    print(
        f"[system1] PlaceObject: object hangs {dangle:.3f} m below the eef, "
        f"push height set to {push_z:.3f} m for an object goal of {target[2]:.3f} m"
    )

    lift = np.array([eef[0], eef[1], push_z + 0.06])
    _, obs = _move_to(env, obs, lift, GRIPPER_CLOSE, budget // 5, frame=frame)
    print(
        f"[system1] PlaceObject: after lift eef_z={_eef_pos(obs)[2]:.3f} "
        f"object_gap={_held_gap(env, obs, target)}"
    )

    # Turn the held object a quarter turn so its long side points into the opening
    # instead of across it, which is what knocks it out of the gripper on the frame.
    obs = _yaw_gripper(env, obs, PLACE_YAW, PLACE_YAW_STEPS, GRIPPER_CLOSE)

    back = target[:2] - _eef_pos(obs)[:2]
    norm = float(np.linalg.norm(back))
    back = back / norm if norm > 1e-6 else np.array([0.0, 1.0])
    entry = target.copy()
    entry[:2] = target[:2] - back * PLACE_ENTRY_BACK
    entry[2] = push_z

    _, obs = _move_to(env, obs, entry, GRIPPER_CLOSE, budget // 5, frame=frame)
    print(f"[system1] PlaceObject: after entry object_gap={_held_gap(env, obs, target)}")

    push = target.copy()
    push[2] = push_z
    reached, obs = _move_to(env, obs, push, GRIPPER_CLOSE, budget // 5, frame=frame)
    gap = float(np.linalg.norm(push - _eef_pos(obs)))
    if not reached:
        print(
            f"[system1] PlaceObject failed: cabinet entry timeout at {gap:.3f} m "
            f"eef_z={_eef_pos(obs)[2]:.3f} push_z={push_z:.3f}"
        )
    print(f"[system1] PlaceObject: after push object_gap={_held_gap(env, obs, target)}")

    obs = _open_gripper(env, obs, CLOSE_STEPS)
    # Lift straight up before retreating: a horizontal withdrawal drags the released
    # object along the shelf and out through the open door.
    lift_off = _eef_pos(obs) + np.array([0.0, 0.0, PLACE_RELEASE_LIFT])
    _, obs = _move_to(env, obs, lift_off, GRIPPER_OPEN, budget // 8, frame=frame)
    retreat = target.copy()
    retreat[:2] = target[:2] - back * PLACE_RETREAT
    retreat[2] = push_z + PLACE_RELEASE_LIFT
    _, obs = _move_to(env, obs, retreat, GRIPPER_OPEN, budget // 4, frame=frame)

    inside = bool(env._check_success())
    obj_pos = _task_object_pos(obs, env, skill_config.get("target_obj"))
    obj_gap = None if obj_pos is None else float(np.linalg.norm(obj_pos - target))
    print(
        f"[system1] PlaceObject: reached={reached} gap={gap:.3f} "
        f"object_to_target={obj_gap} env_success={inside} "
        f"object_at={None if obj_pos is None else np.round(obj_pos, 3).tolist()} "
        f"target={np.round(target, 3).tolist()}"
    )
    return inside, obs


def _skill_navigate(env, obs, skill_config, budget):
    """Approximate navigation with an end-effector approach; the base stays still."""
    target = _find_entity_pos(env, skill_config.get("target_loc"))
    if target is None:
        print(f"[system1] NavigateTo failed: unknown location {skill_config.get('target_loc')}")
        return False, obs

    frame, obs = _action_frame(env, obs)
    goal = target + np.array([0.0, 0.0, NAV_APPROACH_HEIGHT])
    reached, obs = _move_to(env, obs, goal, GRIPPER_OPEN, budget, tol=NAV_TOL, frame=frame)
    if not reached:
        print(
            "[system1] NavigateTo failed: eef could not reach the location. "
            "If the target is outside arm reach, the mobile base dims [7:9] must be driven."
        )
        return False, obs
    print(f"[system1] NavigateTo ok: eef within {NAV_TOL} m of {skill_config.get('target_loc')}")
    return True, obs


def _skill_open_door(env, obs, skill_config, budget):
    """Grasp the handle, pull outward, then confirm the door swung open."""
    fixture, fname = _find_fixture(env, skill_config.get("target_loc"))
    if fixture is None:
        return False, obs

    handle, handle_src = _handle_pos(env, fixture)
    if handle is None:
        print(_door_diagnostics(env, fixture))
        return False, obs
    print(f"[system1] OpenSingleDoor: handle of {fname!r} via {handle_src}")

    angle_before, angle_src = _door_angle(env, fixture)
    body = _body_pos(env, fixture)
    frame, obs = _action_frame(env, obs)

    reached, obs = _move_to(
        env, obs, handle + np.array([0.0, 0.0, DOOR_GRASP_OFFSET]), GRIPPER_OPEN,
        budget // 3, tol=0.02, frame=frame,
    )
    if not reached:
        print(f"[system1] OpenSingleDoor failed: cannot reach the handle of {fname!r}")
        return False, obs

    obs = _close_gripper(env, obs)

    pull_dir = _outward_dir(handle, body)
    pull_target = handle + pull_dir * DOOR_PULL_DIST
    _, obs = _move_to(env, obs, pull_target, GRIPPER_CLOSE, budget // 2, tol=0.02, frame=frame)

    angle_after, _ = _door_angle(env, fixture)
    if angle_before is not None and angle_after is not None:
        delta = abs(angle_after - angle_before)
        print(
            f"[system1] door angle {angle_before:.3f} -> {angle_after:.3f} rad "
            f"(delta {delta:.3f}, via {angle_src})"
        )
        if delta > DOOR_OPEN_ANGLE:
            print("[system1] OpenSingleDoor ok")
            return True, obs
        print("[system1] OpenSingleDoor failed: door angle barely changed")
        return False, obs

    # Fallback: no readable angle, so check whether the handle physically moved.
    handle_after, _ = _handle_pos(env, fixture)
    moved = None if handle_after is None else float(np.linalg.norm(handle_after - handle))
    if moved is not None and moved > DOOR_HANDLE_MOVE:
        print(f"[system1] OpenSingleDoor ok (handle moved {moved:.3f} m, no angle available)")
        return True, obs
    print(f"[system1] OpenSingleDoor failed: door state unreadable (handle moved {moved})")
    return False, obs


def _skill_close_door(env, obs, skill_config, budget):
    """Grasp the handle, push inward, then confirm the door closed."""
    fixture, fname = _find_fixture(env, skill_config.get("target_loc"))
    if fixture is None:
        return False, obs

    handle, handle_src = _handle_pos(env, fixture)
    if handle is None:
        print(_door_diagnostics(env, fixture))
        return False, obs
    print(f"[system1] CloseSingleDoor: handle of {fname!r} via {handle_src}")

    angle_before, angle_src = _door_angle(env, fixture)
    body = _body_pos(env, fixture)
    frame, obs = _action_frame(env, obs)

    reached, obs = _move_to(
        env, obs, handle + np.array([0.0, 0.0, DOOR_GRASP_OFFSET]), GRIPPER_OPEN,
        budget // 3, tol=0.02, frame=frame,
    )
    if not reached:
        print(f"[system1] CloseSingleDoor failed: cannot reach the handle of {fname!r}")
        return False, obs

    obs = _close_gripper(env, obs)

    push_dir = -_outward_dir(handle, body)
    _, obs = _move_to(
        env, obs, handle + push_dir * DOOR_PULL_DIST, GRIPPER_CLOSE, budget // 2, tol=0.02, frame=frame
    )

    angle_after, _ = _door_angle(env, fixture)
    if angle_after is not None:
        print(f"[system1] door angle {angle_before} -> {angle_after:.3f} rad (via {angle_src})")
        if angle_after < DOOR_CLOSED_ANGLE:
            print("[system1] CloseSingleDoor ok")
            return True, obs
        print("[system1] CloseSingleDoor failed: door still open")
        return False, obs

    handle_after, _ = _handle_pos(env, fixture)
    moved = None if handle_after is None else float(np.linalg.norm(handle_after - handle))
    if moved is not None and moved > DOOR_HANDLE_MOVE:
        print(f"[system1] CloseSingleDoor ok (handle moved {moved:.3f} m, no angle available)")
        return True, obs
    print(f"[system1] CloseSingleDoor failed: door state unreadable (handle moved {moved})")
    return False, obs


_SKILL_IMPL = {
    "PickObject": _skill_pick,
    "PlaceObject": _skill_place,
    "NavigateTo": _skill_navigate,
    "OpenSingleDoor": _skill_open_door,
    "CloseSingleDoor": _skill_close_door,
}


def system1_execute(skill_config: dict, env, max_sim_steps: int = 200):
    """
    Execute a single atomic heuristic skill in the RoboCasa simulator.

    Args:
        skill_config: output of SubtaskSkillMapper
        env: robocasa environment instance
        max_sim_steps: hard step budget for one skill, prevents an infinite loop

    Returns:
        (subtask_done_flag: bool, rgb_image: np.ndarray)
    """
    skill_name = skill_config.get("skill_name")
    impl = _SKILL_IMPL.get(skill_name)

    obs = getattr(env, "_last_obs", None)
    if obs is None:
        raise RuntimeError(
            "env has no cached observation; store the obs from env.reset() on env._last_obs "
            "before calling system1_execute()"
        )

    if impl is None:
        raise NotImplementedError(
            f"skill '{skill_name}' has no heuristic implementation yet; "
            f"implemented: {sorted(_SKILL_IMPL)}"
        )

    done_flag, obs = impl(env, obs, skill_config, max_sim_steps)
    env._last_obs = obs
    return bool(done_flag), read_rgb(obs)


if __name__ == "__main__":
    """Interface-only unit test; does not launch a real RoboCasa env."""
    mock_skill_config_list = [
        {"skill_name": "PickObject", "target_obj": "tongs", "target_loc": None,
         "raw_subtask": "robot picks up tongs"},
        {"skill_name": "PlaceObject", "target_obj": "tongs", "target_loc": "cabinet",
         "raw_subtask": "robot places the tongs in the cabinet"},
        {"skill_name": "OpenSingleDoor", "target_obj": None, "target_loc": "cabinet",
         "raw_subtask": "robot opens the cabinet"},
    ]

    class _FakeEnv:
        action_dim = 7
        fixtures = {}
        objects = []
        _last_obs = {
            "robot0_eef_pos": np.zeros(3),
            "robot0_agentview_right_image": np.zeros((256, 256, 3), dtype=np.uint8),
        }

        def step(self, action):
            return self._last_obs, 0.0, False, {}

    env = _FakeEnv()
    for idx, cfg in enumerate(mock_skill_config_list):
        print(f"\n==== Test case {idx + 1} ====")
        try:
            done_flag, rgb = system1_execute(skill_config=cfg, env=env, max_sim_steps=4)
            print(f"done={done_flag}, rgb={rgb.shape}, dtype={rgb.dtype}")
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            print(f"raised as expected on the fake env: {type(exc).__name__}: {exc}")

    print("\nSkeleton interface check finished.")
