import imageio
import robocasa
from robocasa.utils.env_utils import create_env

env = create_env(
    env_name="PickPlaceCounterToCabinet",
    robots=["PandaOmron"],
    seed=0,
    camera_widths=256,
    camera_heights=256,
)
obs = env.reset()

# MuJoCo renders images vertically flipped; reverse rows before saving
for key in [
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
]:
    img = obs[key][::-1]
    imageio.imwrite(f"{key}.png", img)
    print(key, img.shape, "mean pixel:", img.mean())
