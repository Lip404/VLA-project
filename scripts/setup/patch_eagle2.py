#!/usr/bin/env python3
"""Patch the Eagle2.5 vision-language backbone to run without flash-attn.

Why this exists
---------------
GR00T N1.5 builds its backbone from the Eagle2.5 HuggingFace model, whose code

  * imports `flash_attn` at module import time (radio_model.py), and
  * unconditionally monkey-patches timm's attention with a flash-attn version
    (replace_vit_attn_with_flash_attn(), called at import), and
  * pins `_attn_implementation="flash_attention_2"` in both the Python default
    and the serialized config.json.

On a box without flash-attn (prebuilt wheels need glibc >= 2.32, this image has
2.31) the model simply cannot load. The patches below:

  1. make the flash_attn imports optional and skip the monkey patch when it is
     unavailable (timm keeps its own SDPA path),
  2. switch the default attention implementation to "sdpa",
  3. relax the Qwen2 assertion so "sdpa" is accepted,
  4. rewrite any `flash_attention_2` left in the vendored JSON configs to "sdpa".

SDPA is slower than flash-attn but numerically equivalent; measured end-to-end
cost is roughly 1.5 s per policy query, which is fine for evaluation.

Usage: python patch_eagle2.py [repo_root]
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/gr00t_robocasa")
BASE = ROOT / "gr00t" / "model" / "backbone" / "eagle2_hg_model"

if not BASE.is_dir():
    raise SystemExit(f"not found: {BASE}")

# ---------------------------------------------------------------- radio_model.py
p = BASE / "radio_model.py"
s = p.read_text()

OLD_IMPORT = '''try:  # v1
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:  # v2
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
    )

from flash_attn.bert_padding import pad_input, unpad_input'''

NEW_IMPORT = '''try:  # v1
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:  # v2
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
        )
    except ImportError:
        flash_attn_unpadded_qkvpacked_func = None

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    pad_input, unpad_input = None, None'''

if NEW_IMPORT not in s:
    if OLD_IMPORT not in s:
        raise SystemExit("radio_model.py: flash_attn import block not found (already patched?)")
    s = s.replace(OLD_IMPORT, NEW_IMPORT)

OLD_CALL = "\nreplace_vit_attn_with_flash_attn()\n"
NEW_CALL = "\nif flash_attn_unpadded_qkvpacked_func is not None:\n    replace_vit_attn_with_flash_attn()\n"
if NEW_CALL not in s:
    if OLD_CALL not in s:
        raise SystemExit("radio_model.py: replace_vit_attn_with_flash_attn() call not found")
    s = s.replace(OLD_CALL, NEW_CALL)
p.write_text(s)
print("patched:", p.name)

# ---------------------------------------------------------------- configuration_eagle2_5_vl.py
p = BASE / "configuration_eagle2_5_vl.py"
s = p.read_text()
if '_attn_implementation="flash_attention_2"' in s:
    s = s.replace('_attn_implementation="flash_attention_2"', '_attn_implementation="sdpa"', 1)
    p.write_text(s)
    print("patched:", p.name)
else:
    print("already patched:", p.name)

# ---------------------------------------------------------------- modeling_eagle2_5_vl.py
p = BASE / "modeling_eagle2_5_vl.py"
s = p.read_text()
s = s.replace(
    'config.vision_config._attn_implementation = "flash_attention_2"',
    'config.vision_config._attn_implementation = "sdpa"',
)
s, n = re.subn(
    r'assert \(\s*config\.text_config\._attn_implementation == "flash_attention_2"\s*\),'
    r' f"Qwen2 must use flash_attention_2 but got \{config\.text_config\._attn_implementation\}"',
    'assert config.text_config._attn_implementation in ("flash_attention_2", "sdpa")',
    s,
)
if n == 0 and 'in ("flash_attention_2", "sdpa")' not in s:
    raise SystemExit("modeling_eagle2_5_vl.py: Qwen2 assertion not found")
p.write_text(s)
print("patched:", p.name)

# ---------------------------------------------------------------- vendored JSON configs
for jp in sorted(BASE.glob("*.json")):
    try:
        data = json.loads(jp.read_text())
    except Exception:
        continue

    changed = [False]

    def walk(node):
        if isinstance(node, dict):
            for k in list(node):
                if k in ("_attn_implementation", "attn_implementation") and node[k] == "flash_attention_2":
                    node[k] = "sdpa"
                    changed[0] = True
                walk(node[k])
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    if changed[0]:
        jp.write_text(json.dumps(data, indent=2))
        print("patched:", jp.name)

print("done. remember to clear the HF module cache:")
print("  rm -rf /root/.cache/huggingface/modules/transformers_modules")
