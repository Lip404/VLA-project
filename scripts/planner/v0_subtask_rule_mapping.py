"""
v0_subtask_rule_mapping.py

Subtask -> RoboCasa atomic skill mapper (System-1 glue layer).

Rule-based prototype with a small object / location vocabulary. Two deliberate
choices: place rules are evaluated before pick rules (a sentence may mention both),
and the mapper returns None instead of a half-filled config when required arguments
are missing, so the caller aborts with one clear message.
"""

import re
from typing import Dict, Optional

# Skills the executor can actually run right now. Keep this tuple in sync with
# scripts/sim/system1_skill_executor.py (_SKILL_IMPL).
EXECUTABLE_SKILLS = (
    "PickObject",
    "PlaceObject",
    "NavigateTo",
    "OpenSingleDoor",
    "CloseSingleDoor",
)

RULE_ORDER = (
    ("PlaceObject", (
        "place in", "place the", "places the", "put in", "put the", "puts the",
        "insert", "drop",
    )),
    ("PickObject", (
        "pick up", "picks up", "pick the", "picks the", "grasp", "grab",
    )),
    ("OpenSingleDoor", ("open the cabinet", "open cabinet", "open the door", "open door", "opens the")),
    ("CloseSingleDoor", ("close the cabinet", "close cabinet", "close the door", "close door", "closes the")),
    ("NavigateTo", ("navigate to", "move towards", "moves towards", "go to", "walk to")),
)

KNOWN_OBJECTS = (
    "tongs", "bowl", "cup", "plate", "mug", "bottle", "can", "pot", "pan",
    "apple", "banana", "bread", "towel",
)
KNOWN_LOCATIONS = (
    "cabinet", "counter", "shelf", "drawer", "sink", "stove", "microwave",
    "fridge", "refrigerator",
)

_OBJECT_PATTERN = re.compile(
    r"(?:pick up|picks up|pick the|picks the|place|places|put|puts|insert|drop)"
    r"\s+(?:the\s+|a\s+|an\s+)?([a-z]+)"
)


class SubtaskSkillMapper:
    def __init__(self):
        self.executable_skills = EXECUTABLE_SKILLS

    @staticmethod
    def _match_skill(lower_text: str) -> Optional[str]:
        for skill, keywords in RULE_ORDER:
            if any(keyword in lower_text for keyword in keywords):
                return skill
        return None

    @staticmethod
    def _extract_object(lower_text: str) -> Optional[str]:
        for word in KNOWN_OBJECTS:
            if word in lower_text:
                return word
        match = _OBJECT_PATTERN.search(lower_text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_location(lower_text: str) -> Optional[str]:
        for word in KNOWN_LOCATIONS:
            if word in lower_text:
                return word
        return None

    def match_skill(self, subtask_text: str) -> Optional[Dict]:
        """Map a subtask sentence to a skill config, or None when it cannot be executed."""
        if not subtask_text:
            return None

        lower_text = subtask_text.lower()
        skill_name = self._match_skill(lower_text)
        if skill_name is None:
            print(f"[Mapper] no rule matched: {subtask_text!r}")
            return None

        target_obj = self._extract_object(lower_text)
        target_loc = self._extract_location(lower_text)

        # Argument requirements per skill; missing args would fail deep inside the executor.
        if skill_name == "PickObject" and target_obj is None:
            print(f"[Mapper] PickObject without a known object: {subtask_text!r}")
            return None
        if skill_name == "PlaceObject" and (target_obj is None or target_loc is None):
            print(
                f"[Mapper] PlaceObject needs object and location "
                f"(obj={target_obj}, loc={target_loc}): {subtask_text!r}"
            )
            return None
        if skill_name in ("OpenSingleDoor", "CloseSingleDoor") and target_loc is None:
            print(f"[Mapper] {skill_name} without a known location: {subtask_text!r}")
            return None

        config = {
            "skill_name": skill_name,
            "target_obj": target_obj,
            "target_loc": target_loc,
            "raw_subtask": subtask_text,
            "executable": skill_name in self.executable_skills,
        }
        if not config["executable"]:
            print(f"[Mapper] {skill_name} is not implemented in the executor yet")
        return config


if __name__ == "__main__":
    mapper = SubtaskSkillMapper()
    test_cases = [
        "robot picks up the tongs",
        "robot places the tongs in the cabinet",
        "robot puts the bowl on the counter",
        "robot navigates to the counter",
        "robot opens the cabinet door",
        "some unmappable sentence",
    ]
    for case in test_cases:
        print(f"{case!r:52} -> {mapper.match_skill(case)}")
