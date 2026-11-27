# GR00T N1.5 on a rented GPU box — field notes

## Status

* **2026-11-25 (RTX 4090, Asia-Pacific region 2)**: simulator runs, checkpoint
  loads, one episode completes (70.8 s, success False — expected, this checkpoint
  scores 63 % on that task).
* **2026-11-27 (A100-PCIE-40GB, region 16)**: full rebuild from the scripts
  below. Smoke test `obs keys: 32`, `policy loaded`, first episode **success
  True**, 348.6 s/episode. The 5× slowdown vs the 4090 is not yet diagnosed
  (CPU llvmpipe rendering vs SDPA inference); measure over several episodes
  before the full eval run.

## Scripts in `scripts/setup/`

| file | purpose |
| --- | --- |
| `setup_card.sh` | rebuilds the whole environment on a fresh instance |
| `patch_eagle2.py` | removes the flash-attn dependency from the Eagle2.5 backbone |
| `download_assets_hf.py` | downloads RoboCasa assets from the HF mirror (Box is unreachable) |
| `fix_assets_layout_v2.py` | moves asset categories into the directories robocasa expects |
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

## Asset source: HF mirror, not Box

Upstream `download_kitchen_assets.py` pulls six zips from `utexas.box.com`.
That host timed out after 20 s and transferred 0 B from the rented GPU region
(front page 302 responds, file CDN does not). The same assets exist on the HF
mirror; `download_assets_hf.py` uses them:

* dataset `robocasa/robocasa-assets`: `textures.zip`, `generative_textures.zip`,
  `fixtures.zip`, `objaverse.zip`, `aigen_objs.zip` (≈ 9.9 GB compressed)
* dataset `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF`:
  `fixtures_lightwheel/*.zip`, `objects_lightwheel/*.zip` (86 files)

**Layout trap.** The Box zips carry a `fixtures/` and `lightwheel/` wrapper
level; the HF zips do not. After extraction the fixture categories land at
`assets/<category>/` and object categories at `objects/<category>/`, but
robocasa resolves them under `assets/fixtures/<category>/` and
`objects/lightwheel/<category>/`. `fix_assets_layout_v2.py` merges them into
the right places; conflicting files are compared byte-by-byte, identical copies
are removed, real differences are printed as `DIFFER conflict:`.

## pip mirror warning

The tuna PyPI mirror returned `from versions: none` for `albumentations==1.4.18`
and `huggingface_hub<1.0` even though both exist on PyPI. Official PyPI works.
Aliyun resolves metadata but stalled mid-wheel once; keep it as a fallback, do
not pin it. If pip reports missing versions, check `pip config list` first.

## Numbers measured

* weights download on the card: 7.59 GB / 8 files in **2 min 37 s**
* HF mirror bandwidth on region 16: ≈ 143 MB/s for model/asset zips
* robocasa assets: ≈ 23 GB on disk after extraction (10 GB zips stay in
  `_downloads/` until deleted)
* model footprint: ~6 GB VRAM, 3 B backbone + 550 M DiT + 201 M transformer
* 4090 episode: **≈ 71 s**; A100 episode: **348.6 s** (one sample, diagnose)

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
4. diagnose the A100 348.6 s/episode slowdown before any multi-episode eval.

## Machine / billing

matpool. Region 2 (instance `wLgEW5`, RTX 4090 24 GB, ¥2.20/h) was retired
and force-released; region 16 instance `qe84eM`, A100-PCIE-40GB, 12 vCPU,
90 GB RAM, 200 GB SSD, driver 610.57.04 / CUDA 13.3. `/mnt` is a 5 GB network
disk, too small for weights — only scripts and source belong there. Rebuild from
`setup_card.sh`; weights re-download in under 3 min, assets from the HF mirror.
