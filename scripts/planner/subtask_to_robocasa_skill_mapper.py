"""
Subtask to RoboCasa skill mapper (System1 glue layer)
Input: natural language subtask string from System2 planner
Output: structured robocasa atomic skill config
This is a rule-based prototype, will upgrade to embedding similarity later.
"""
from typing import Dict, Optional


class SubtaskSkillMapper:
    def __init__(self):
        # simple rule map: keyword -> skill name
        self.rule_map = [
            {"keywords": ["pick up", "picks up", "pick the"], "skill": "PickObject"},
            {"keywords": ["place in", "places the"], "skill": "PlaceObject"},
            {"keywords": ["open cabinet", "open door"], "skill": "OpenSingleDoor"},
            {"keywords": ["close cabinet", "close door"], "skill": "CloseSingleDoor"},
            {"keywords": ["move towards", "moves towards", "navigate to"], "skill": "NavigateTo"},
        ]

    def match_skill(self, subtask_text: str) -> Optional[Dict]:
        """Match subtask text to robocasa atomic skill and extract arguments"""
        lower_text = subtask_text.lower()

        # match skill type
        matched_skill = None
        for item in self.rule_map:
            for kw in item["keywords"]:
                if kw in lower_text:
                    matched_skill = item["skill"]
                    break
            if matched_skill is not None:
                break
        if matched_skill is None:
            return None

        # extract simple arguments for our tongs & cabinet task
        target_obj = None
        target_loc = None
        if "tongs" in lower_text:
            target_obj = "tongs"
        if "cabinet" in lower_text:
            target_loc = "cabinet"
        if "counter" in lower_text:
            target_loc = "counter"

        return {
            "skill_name": matched_skill,
            "target_obj": target_obj,
            "target_loc": target_loc,
            "raw_subtask": subtask_text
        }


if __name__ == "__main__":
    # local test
    mapper = SubtaskSkillMapper()
    test_cases = [
        "robot picks up tongs",
        "robot moves towards the cabinet",
        "robot places the tongs in the cabinet"
    ]
    for case in test_cases:
        res = mapper.match_skill(case)
        print(f"subtask: {case} -> skill: {res}")
