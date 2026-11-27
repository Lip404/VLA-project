#!/usr/bin/env bash
# Bootstrap RoboCasa + GR00T N1.5 (atomic-seen checkpoint) on a rented GPU box.
# Tested on: matpool, Ubuntu 20.04, conda env "robocasa_env" (Python 3.10),
#            RTX 4090 and A100-PCIE-40GB.
#
# Usage:
#   conda create -n robocasa_env python=3.10 && conda activate robocasa_env
#   bash setup_card.sh 2>&1 | tee /root/setup_card.log
#
# Notes:
#   - This image has NO libEGL_nvidia, so GPU offscreen rendering is impossible.
#     Rendering works through xvfb + Mesa llvmpipe (CPU). Always run sim code with:
#         export MUJOCO_GL=glx
#         xvfb-run -a -s "-screen 0 1280x1024x24" python your_script.py
#   - Do NOT install flash-attn: the prebuilt wheels need glibc >= 2.32 and this
#     image is 2.31. patch_eagle2.py switches the vision backbone to PyTorch SDPA.
#   - Assets come from HuggingFace mirror, NOT the Box links in upstream robocasa.
#     The utexas.box.com direct download timed out from rented GPU regions.

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------- apt
# The nvidia-cuda apt source in this image is frequently out of sync with the
# mirror; its failure is harmless because torch comes from pip.
apt-get update || true
apt-get install -y libosmesa6-dev libglew-dev libgl1-mesa-glx libglu1-mesa xvfb || true

# --------------------------------------------------------------------- python
# Official PyPI is the default. The tuna mirror was observed to return
# "from versions: none" for albumentations==1.4.18 and huggingface_hub<1.0,
# so do not pin it. If PyPI itself stalls on a large wheel, switch to aliyun:
#   pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
#   pip config set global.trusted-host mirrors.aliyun.com
pip install "PyOpenGL==3.1.6"
pip install torch==2.7.1

# --------------------------------------------------------------------- robosuite (from source)
# PyPI 1.5.2 is too old for current robocasa: it lacks the `load_model_on_init`
# kwarg that robocasa passes to ManipulationEnv.
cd /root
[ -d robosuite ] || git clone https://github.com/ARISE-Initiative/robosuite.git
cd /root/robosuite && pip install -e . --config-settings editable_mode=compat

# --------------------------------------------------------------------- robocasa (from source)
cd /root
[ -d robocasa ] || git clone https://github.com/robocasa/robocasa.git
cd /root/robocasa && pip install -e . --config-settings editable_mode=compat

# --------------------------------------------------------------------- GR00T fork for GR00T N1.5
# NOTE: the upstream NVIDIA/Isaac-GR00T repo is the N1.6/N1.7 generation and
# cannot load the N1.5 checkpoint. This fork is the one that matches.
cd /root
[ -d gr00t_robocasa ] || git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git gr00t_robocasa
cd /root/gr00t_robocasa && pip install -e . --no-deps

pip install albumentations==1.4.18 "av==12.3.0" blessings==1.7 dm_tree==0.1.8 einops==0.8.1 \
    gymnasium==1.0.0 h5py "hydra-core==1.3.2" "imageio==2.34.2" kornia==0.7.4 "matplotlib==3.10.0" \
    numpydantic==1.6.7 omegaconf==2.3.0 "opencv_python_headless==4.11.0.86" "pandas==2.2.3" \
    "pydantic==2.10.6" PyYAML==6.0.2 "Requests==2.32.3" tianshou==0.5.1 "timm==1.0.14" \
    "tqdm==4.67.1" "transformers==4.51.3" "typing_extensions==4.12.2" pyarrow wandb \
    "fastparquet==2024.11.0" accelerate "protobuf==3.20.3" "onnx==1.15.0" tyro pytest "peft==0.17.0" zmq

# diffusers 0.30.x still calls huggingface_hub.resolve_revision(), removed in hub 1.x.
pip install "huggingface_hub<1.0" "diffusers==0.30.2"

# A half-installed flash_attn breaks `import transformers`, not just the model.
pip uninstall -y flash_attn || true

# Apply the SDPA patches to the vendored Eagle2.5 backbone.
python "$SCRIPT_DIR/patch_eagle2.py"
rm -rf /root/.cache/huggingface/modules/transformers_modules

# --------------------------------------------------------------------- assets (HF mirror, not Box)
python "$SCRIPT_DIR/download_assets_hf.py"
python "$SCRIPT_DIR/fix_assets_layout_v2.py"

# --------------------------------------------------------------------- weights (7.6 GB)
mkdir -p /root/weights
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robocasa/robocasa365_checkpoints",
    repo_type="model",
    allow_patterns=["gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000/*"],
    ignore_patterns=["*optimizer.pt"],
    local_dir="/root/weights/gr00t_n15_atomic",
    max_workers=4,
)
print("weights done")
PY

# --------------------------------------------------------------------- verify
cat <<'EOF'

Verification steps (run manually):

1) environment smoke test
   export MUJOCO_GL=glx
   xvfb-run -a -s "-screen 0 1280x1024x24" python - <<'PY'
   from robocasa.utils.env_utils import create_env
   env = create_env(env_name="PickPlaceCounterToCabinet", robots=["PandaOmron"], seed=0,
                    camera_widths=256, camera_heights=256)
   print("obs keys:", len(env.reset())); env.close()
   PY
   # expected: obs keys: 32

2) policy load
   cd /root/gr00t_robocasa && python - <<'PY'
   from gr00t.experiment.data_config import DATA_CONFIG_MAP
   from gr00t.model.policy import Gr00tPolicy
   dc = DATA_CONFIG_MAP["panda_omron"]
   Gr00tPolicy(model_path="/root/weights/gr00t_n15_atomic/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
               modality_config=dc.modality_config(), modality_transform=dc.transform(),
               embodiment_tag="new_embodiment", denoising_steps=4)
   print("policy loaded")
   PY

3) one real episode: see docs/GR00T_SETUP_NOTES.md
   # reference result on 2026-11-27 (A100-PCIE-40GB): episode success True,
   # 348.6 s per episode. On RTX 4090 (2026-11-25): 70.8 s, success False.
   # This checkpoint scores 63% on PickPlaceCounterToCabinet, so failures are normal.

EOF
