import os, pathlib, zipfile
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

ASSETS = pathlib.Path("/root/robocasa/robocasa/models/assets")
DL = ASSETS / "_downloads"
DL.mkdir(parents=True, exist_ok=True)

print("downloading robocasa/robocasa-assets ...")
snapshot_download(
    repo_id="robocasa/robocasa-assets",
    repo_type="dataset",
    local_dir=str(DL),
    max_workers=4,
)

print("downloading nvidia kitchen assets ...")
snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF",
    repo_type="dataset",
    allow_patterns=["fixtures_lightwheel/*.zip", "objects_lightwheel/*.zip"],
    local_dir=str(DL),
    max_workers=4,
)

def top_dirs(z):
    tops = set()
    for n in z.namelist():
        n = n.rstrip("/")
        if n:
            tops.add(n.split("/")[0])
    return tops

def smart_extract(zp, target_dir, expected_top):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        tops = top_dirs(z)
        if tops == {expected_top}:
            z.extractall(target_dir.parent)
            print("extract wrapper:", zp.name, "->", target_dir.parent)
        else:
            z.extractall(target_dir)
            print("extract flat:", zp.name, "->", target_dir)

ROOT_MAP = [
    ("textures.zip", "textures"),
    ("generative_textures.zip", "generative_textures"),
    ("fixtures.zip", "fixtures"),
    ("objaverse.zip", "objects/objaverse"),
    ("aigen_objs.zip", "objects/aigen_objs"),
]
for fname, sub in ROOT_MAP:
    zp = DL / fname
    if zp.exists():
        smart_extract(zp, ASSETS / sub, sub.split("/")[-1])
    else:
        print("missing:", fname)

for zp in sorted((DL / "fixtures_lightwheel").glob("*.zip")):
    smart_extract(zp, ASSETS / "fixtures", zp.stem)

for zp in sorted((DL / "objects_lightwheel").glob("*.zip")):
    smart_extract(zp, ASSETS / "objects/lightwheel", zp.stem)

print("assets done")
