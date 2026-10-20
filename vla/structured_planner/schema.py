# vla/structured_planner/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional

# ==============================================
# 1. VALID_OBSERVED_ACTIONS: allowed action vocabulary
# Meaning: VLM observer can only output actions inside this set
# If the model outputs an action not in this set (e.g. kick), the parser will mark it invalid
# ==============================================
VALID_OBSERVED_ACTIONS = {
    "reach",
    "grasp",
    "move",
    "release",
    "rotate",
    "push",
    "pull",
    "move_forward"
}

# ==============================================
# 2. ObservedAction class: template for single action entry
# Meaning: defines the format of each item inside action_sequence list
# ==============================================
class ObservedAction(BaseModel):
    action: str = Field(description="Action observed in video, must in VALID_OBSERVED_ACTIONS")
    # action: name of the action, must come from the vocabulary above
    target_object: Optional[str] = Field(None, description="Object this action acts on")
    # target_object: the object that this manipulation acts on, example: cup
    start_frame: Optional[int] = Field(None, description="Approximate start frame index of this action")
    # start_frame: roughly which frame this action begins
    end_frame: Optional[int] = Field(None, description="Approximate end frame index of this action")
    # end_frame: roughly which frame this action finishes
    spatial_change: Optional[str] = Field(None, description="How object position changed")
    # spatial_change: description of object movement, example: cup moves from left to right

# ==============================================
# 3. VideoBehaviorRecord class: top-level template for full observation record
# Meaning: defines the structure of the final JSON output from VLM
# action_sequence is defined here!
# ==============================================
class VideoBehaviorRecord(BaseModel):
    video_summary: str
    # video_summary: one-sentence summary of what happens in this video clip
    objects_in_scene: List[str]
    # objects_in_scene: list of all objects visible in the video
    action_sequence: List[ObservedAction]
    # action_sequence: chronological list of observed actions
    # List order = time order that actions happen inside the video
    notes: Optional[str] = None
    # notes: for unclear observation or failure notes

# ==============================================
# 4. OBSERVE_VIDEO_SYSTEM_PROMPT: instruction sent to Qwen2VL
# Tells the model its role, task, rules and required output format
# ==============================================
OBSERVE_VIDEO_SYSTEM_PROMPT = """
You are a video behavior observer for robot imitation learning.
You will receive several frames sampled from a continuous video.
Your job is to observe what actions HAPPENED in this video, in chronological order.

Rules:
1. Output JSON wrapped inside ```json ... ``` markdown code block. NO extra natural language explanation, no comments outside the code block.
2. Action must be chosen from this set: reach, grasp, move, release, rotate, push, pull, move_forward
3. Record actions in the SAME TIME ORDER they appear in the video.
4. Fill frame range if you can infer when this action occurs.
5. Follow exactly this schema:
{
  "video_summary": "short one-sentence summary of this video clip",
  "objects_in_scene": ["list of unique distinct visible object names, each object appears only once"],
  "action_sequence": [
    {
      "action": "action_name",
      "target_object": "object name",
      "start_frame": null or integer,
      "end_frame": null or integer,
      "spatial_change": "describe object position change"
    }
  ],
  "notes": null
}
If you cannot identify any manipulation action, leave action_sequence empty and put your reason in notes.

Valid example output:
```json
{
  "video_summary": "The robot moves forward toward the cube block",
  "objects_in_scene": ["cube block", "robot"],
  "action_sequence": [
    {
      "action": "move_forward",
      "target_object": "cube block",
      "start_frame": 0,
      "end_frame": 4,
      "spatial_change": "robot moves closer to the cube block"
    }
  ],
  "notes": null
}
```

IMPORTANT CONSTRAINTS YOU MUST FOLLOW:
1. NEVER infinitely repeat the same object name. List each distinct object ONLY ONCE in objects_in_scene.
2. Merge continuous identical actions with the same action name and same target_object into ONE single entry.
   Do NOT split one continuous movement into many tiny frame-wise slices.
3. Limit the length of action_sequence to at most 4 action steps. Do NOT generate unlimited repeated actions.
4. You must produce a fully complete, well-formed JSON. All quotes, brackets and braces must be properly closed.
5. Do NOT leave sentences, strings or JSON structures unfinished. Stop generation right after finishing the complete JSON object.
"""