# GR00T N1.5 on a rented GPU box — field notes (2026-11-25)

Status at end of day: simulator runs, GR00T N1.5 `atomic_seen/checkpoint-60000`
loads, and one real episode of `PickPlaceCounterToCabinet` completed on the card
(70.8 s, success False — expected, this checkpoint scores 63 % on that task).

## Three files

| file | purpose |
| --- | --- |
| `setup_card.sh` | rebuilds the whole environment on a fresh instance |
| `patch_eagle2.py` | removes the flash-attn dependency from the Eagle2.5 backbone |
| this file | what broke, why, and how to check |

## Must-remember facts

* **Rendering goes through the CPU.** The image has driver stubs but no
  `libEGL_nvidia.so`, so EGL offscreen rendering cannot work. robosuite
  `binding_utils.py` hard-codes EGL unless `MUJOCO_GL` is `osmesa` or `glx`;
  with `GPU_RENDERING` on it silently rewrites any other value to `egl`.
  Working recipe: `export MUJOCO_GL=glx` + `xvfb-run`.
* **Do not install flash-attn.** Prebuilt wheels require glibc ≥ 2.32 (image is
  2.31), and a half-installed package breaks `import transformers` globally, not
  just the model. Use the SDPA patch instead.
* **robosuite must come from source, not PyPI.** PyPI tops out at 1.5.2, which
  lacks the `load_model_on_init` kwarg that current robocasa passes in.
* **`pip install -e . --no-deps` for the GR00T fork.** Otherwise pip pulls
  `torch==2.5.1` and reshuffles the env.
* **`huggingface_hub` must stay < 1.0.** diffusers 0.30.x still calls
  `resolve_revision`, which hub 1.x removed.
* **robocasa main is required**: `utils/dataset_registry.py` (task sets) and
  `utils/groot_utils` (gym wrapper producing `video/state/annotation` keys) only
  exist there. The docs' `scripts/run_eval.py` lives in the GR00T fork, not in
  robocasa.

## Numbers measured today (useful for budgeting)

* weights download on the card: 7.59 GB / 8 files in **2 min 37 s**
* robocasa assets: 23 GB on disk, 16–19 MB/s
* one episode (750 control steps, 256×256 × 3 cameras, CPU rendering):
  **≈ 71 s**
* model footprint: ~6 GB VRAM, 3 B backbone + 550 M DiT + 201 M transformer

## One episode, start to finish

```bash
export MUJOCO_GL=glx
cd /root/gr00t_robocasa && xvfb-run -a -s "-screen 0 1280x1024x24" python - <<'PY'
import threading, time
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    SimulationInferenceClient, SimulationConfig, MultiStepConfig, VideoConfig,
)

CKPT = ("/root/weights/gr00t_n15_atomic/gr00t_n1-5/foundation_model_learning/"
        "target_posttraining/atomic_seen/checkpoint-60000")
dc = DATA_CONFIG_MAP["panda_omron"]
policy = Gr00tPolicy(
    model_path=CKPT,
    modality_config=dc.modality_config(),
    modality_transform=dc.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=4,
)
threading.Thread(target=RobotInferenceServer(policy, port=5555).run, daemon=True).start()
time.sleep(2)

client = SimulationInferenceClient(host="localhost", port=5555)
cfg = SimulationConfig(
    env_name="robocasa/PickPlaceCounterToCabinet",
    split="target",
    n_episodes=1,
    n_envs=1,
    video=VideoConfig(video_dir=None),
    multistep=MultiStepConfig(n_action_steps=16, max_episode_steps=750),
)
name, succ = client.run_simulation(cfg)
print("env:", name, "success:", succ)
PY
```

## Checkpoint contract (read from `experiment_cfg/metadata.json`)

* three cameras, 256×256, 20 fps: `robot0_eye_in_hand`,
  `robot0_agentview_left`, `robot0_agentview_right`
* state, 16 dims: `base_position`(3), `base_rotation` quat(4),
  `end_effector_position_relative`(3), `end_effector_rotation_relative` quat(4),
  `gripper_qpos`(2)
* action, 12 dims in normalised space: `base_motion`(4), `control_mode`(1),
  `end_effector_position`(3), `end_effector_rotation` axis-angle(3),
  `gripper_close`(1)
* `embodiment_tag = "new_embodiment"`, data config `panda_omron`

## What is still missing (next session)

1. an adapter that exposes `act(obs, instruction) -> [T, 12]` on top of the
   loaded policy, so the project's closed loop can call it like the old scripted
   System1;
2. instruction injection: replace the annotation text coming from the robocasa
   gym wrapper with System2's subtask sentence and check the policy still moves.
   This is the core assumption of the whole pipeline.
3. System2 training labels are still free-form ("robot picks up tongs"); they
   must be migrated to the RoboCasa instruction space before the injection test
   can mean anything.

## Machine / billing

matpool, Asia-Pacific region 2, instance `wLgEW5`, RTX 4090 24 GB, ¥2.20/h.
`/mnt` is a 5 GB network disk, too small for weights — only scripts and source
belong there. The Asia-Pacific region 2 is being retired; on the new region
rebuild from `setup_card.sh` (weights re-download in under 3 min).
