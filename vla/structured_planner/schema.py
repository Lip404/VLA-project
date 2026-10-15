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
1. ONLY output valid JSON, NO extra natural language explanation, no comments outside JSON.
2. Action must be chosen from this set: reach, grasp, move, release, rotate, push, pull
3. Record actions in the SAME TIME ORDER they appear in the video.
4. Fill frame range if you can infer when this action occurs.
5. Follow exactly this schema:
{
  "video_summary": "short one-sentence summary of this video clip",
  "objects_in_scene": ["list of objects visible"],
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
If you cannot identify any robot/manipulation action, leave action_sequence empty and explain in notes.
"""