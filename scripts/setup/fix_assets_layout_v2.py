import pathlib, shutil, filecmp

ASSETS = pathlib.Path("/root/robocasa/robocasa/models/assets")

FIXTURE_CATS = [
    "blenders", "cabinets", "coffee_machines", "dishwashers",
    "electric_kettles", "fridges", "handles", "hoods", "microwaves",
    "ovens", "sinks", "stand_mixers", "stoves", "stovetops",
    "toaster_ovens", "toasters", "windows",
]

def merge_move(src, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        target = dst_dir / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                merge_move(item, target)
            elif filecmp.cmp(item, target, shallow=False):
                item.unlink()
            else:
                print("DIFFER conflict:", target)
        else:
            shutil.move(str(item), str(target))

for cat in FIXTURE_CATS:
    src = ASSETS / cat
    if src.is_dir():
        merge_move(src, ASSETS / "fixtures" / cat)
        shutil.rmtree(src)
        print("fixture merged:", cat)

OBJ_KEEP = {"aigen_objs", "objaverse", "lightwheel"}
OBJ = ASSETS / "objects"
for d in sorted(OBJ.iterdir()):
    if d.is_dir() and d.name not in OBJ_KEEP:
        merge_move(d, OBJ / "lightwheel" / d.name)
        shutil.rmtree(d)
        print("object merged:", d.name)

print("layout fixed")
