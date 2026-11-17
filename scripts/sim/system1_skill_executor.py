"""
system1_skill_executor.py
System‑1 execution wrapper for RoboCasa heuristic atomic skill.
Implement system1_execute() interface.
Input: skill_config dict from SubtaskSkillMapper
Output: (subtask_done_flag: bool, rgb_image)
"""
import numpy as np


def system1_execute(skill_config: dict, env, max_sim_steps: int = 200):
    """
    Execute single atomic heuristic skill in RoboCasa simulator.
    Args:
        skill_config: output dictionary from SubtaskSkillMapper
        env: robocasa gym environment instance
        max_sim_steps: maximum simulation steps for one atomic skill, prevent infinite loop
    Returns:
        tuple: (subtask_done_flag: bool, rgb_obs_image: np.ndarray)
            subtask_done_flag: True if atomic skill finished successfully, False for timeout / failure
            rgb_obs_image: rgb observation image after skill execution
    """
    skill_name = skill_config["skill_name"]
    target_obj = skill_config["target_obj"]
    target_loc = skill_config["target_loc"]

    subtask_done_flag = False
    rgb_image = None

    for _ in range(max_sim_steps):
        # ========= Real logic need to be migrated from RoboCasa atomic task source code =========
        # action = get_heuristic_action(env, skill_name, target_obj, target_loc)
        # obs, reward, env_done, info = env.step(action)
        # rgb_image = obs["robot0_agentview_right"]["rgb"]
        # if check_skill_success(env, skill_name, target_obj, target_loc):
        #     subtask_done_flag = True
        #     break
        # ======================================================================================
        pass

    # Temporary placeholder for development, remove after connecting real robocasa logic
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    return subtask_done_flag, rgb_image


if __name__ == "__main__":
    """
    Local skeleton unit test.
    NOTE: This test does NOT launch real RoboCasa env.
    Only verify function interface, input/output format, no‑exception execution.
    """
    # Mock skill_config simulating output from subtask‑to‑robocasa‑skill mapper
    mock_skill_config_list = [
        {
            "skill_name": "PickObject",
            "target_obj": "tongs",
            "target_loc": None,
            "raw_subtask": "robot picks up tongs"
        },
        {
            "skill_name": "PlaceObject",
            "target_obj": "tongs",
            "target_loc": "cabinet",
            "raw_subtask": "robot places the tongs in the cabinet"
        },
        {
            "skill_name": "OpenSingleDoor",
            "target_obj": None,
            "target_loc": "cabinet",
            "raw_subtask": "robot open cabinet door"
        }
    ]

    for idx, mock_cfg in enumerate(mock_skill_config_list):
        print(f"\n==== Test case {idx+1} ====")
        print(f"Input skill_config: {mock_cfg}")
        # Pass None for env, placeholder logic does not use env argument
        done_flag, rgb_img = system1_execute(skill_config=mock_cfg, env=None)
        print(f"subtask_done_flag = {done_flag}")
        print(f"rgb_image shape = {rgb_img.shape}")
        assert isinstance(done_flag, bool), "done_flag must be boolean type"
        assert rgb_img.shape == (480, 640, 3), "rgb image shape should be (480,640,3)"

    print("\n✅ All skeleton unit‑tests passed, interface works without exception.")
