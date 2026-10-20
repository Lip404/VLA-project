# vla/structured_planner/json_parser.py
import json
import re
from json_repair import repair_json
from pydantic import ValidationError
from .schema import VideoBehaviorRecord, VALID_OBSERVED_ACTIONS, ObservedAction


def extract_json_from_text(raw_text: str) -> str:
    """
    Extract JSON block from raw model output.
    The model may wrap JSON inside markdown ```json ... ``` or add extra text.
    """
    # Match markdown json code block, no mandatory newline after ```json and before ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, raw_text)
    if match:
        return match.group(1).strip()
    # If no markdown block, try to find the first { and last }
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return raw_text[start_idx:end_idx + 1].strip()
    return raw_text.strip()


def merge_continuous_actions(action_items: list[ObservedAction]) -> list[ObservedAction]:
    """
    Merge continuous actions with identical action name and target_object.
    Combine frame range and spatial description, then cap total entries to max 4.
    """
    if not action_items:
        return action_items
    merged = [action_items[0].model_copy()]
    for item in action_items[1:]:
        last = merged[-1]
        # Merge if action name and target object are exactly the same
        if item.action == last.action and item.target_object == last.target_object:
            last.end_frame = item.end_frame
            last.spatial_change = f"{last.spatial_change}; {item.spatial_change}"
        else:
            merged.append(item.model_copy())
    # Keep at most 4 action entries after merging
    return merged[:4]


def parse_video_behavior(raw_model_output: str) -> tuple[VideoBehaviorRecord | None, str]:
    """
    Main parse & validate entry.
    Returns (parsed_record, error_message).
    If success: error_message is empty string.
    If failed: record is None, error message explains the failure.
    """
    json_str = extract_json_from_text(raw_model_output)
    try:
        fixed_json_str = repair_json(json_str)
        data = json.loads(fixed_json_str)
    except Exception as e:
        return None, f"JSON repair & decode failed: {str(e)}"

    try:
        record = VideoBehaviorRecord(**data)
    except ValidationError as e:
        return None, f"Schema validation failed: {str(e)}"

    # Extra rule: check all action names are inside valid vocabulary
    for action_item in record.action_sequence:
        if action_item.action not in VALID_OBSERVED_ACTIONS:
            return None, f"Invalid action name: {action_item.action}. Valid set: {VALID_OBSERVED_ACTIONS}"

    # Merge continuous duplicate actions and limit to maximum 4 entries
    record.action_sequence = merge_continuous_actions(record.action_sequence)
    
    return record, ""
