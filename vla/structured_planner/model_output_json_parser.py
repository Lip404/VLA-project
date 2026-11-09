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


def parse_video_behavior(raw_model_output: str) -> tuple[VideoBehaviorRecord | None, str, bool]:
    """
    Main parse & validate entry.
    Returns (parsed_record, error_message, is_recovered_truncated).
    If success: error_message is empty string.
    If failed: record is None, error message explains the failure.
    is_recovered_truncated = True means original raw JSON was broken/truncated, fixed by repair_json.
    """
    json_str = extract_json_from_text(raw_model_output)

    # ===================== MODIFIED: strict raw json check BEFORE repair =====================
    try:
        json.loads(json_str)
        is_raw_json_valid = True
    except json.JSONDecodeError:
        is_raw_json_valid = False
    # =========================================================================================

    try:
        fixed_json_str = repair_json(json_str)
        data = json.loads(fixed_json_str)
    except Exception as e:
        return None, f"JSON repair & decode failed: {str(e)}", False

    try:
        record = VideoBehaviorRecord(**data)
    except ValidationError as e:
        return None, f"Schema validation failed: {str(e)}", False

    # ===================== MODIFIED: mark record and flag if raw json invalid =====================
    if not is_raw_json_valid:
        record.notes = "TRUNCATED_JSON: Model output was incomplete, recovered by json_repair"
    # ==============================================================================================

    # Merge continuous duplicate actions and limit to maximum 4 entries
    record.action_sequence = merge_continuous_actions(record.action_sequence)

    # Extra rule: check all action names are inside valid vocabulary
    for action_item in record.action_sequence:
        if action_item.action not in VALID_OBSERVED_ACTIONS:
            return None, f"Invalid action name: {action_item.action}. Valid set: {VALID_OBSERVED_ACTIONS}", False

    return record, "", (not is_raw_json_valid)


def extract_json_array(raw_text: str):
    """
    Extract and parse JSON array [...] from model raw output string.
    Used for System2 subtask planner output.
    :param raw_text: full raw output string from model
    :return: python list of subtasks
    :raises json.JSONDecodeError: when cannot parse valid json array
    """
    # Strip assistant prefix leftover after skip_special_tokens
    if "assistant" in raw_text:
        candidate = raw_text.split("assistant")[-1].strip()
    else:
        candidate = raw_text.strip()

    # Locate first '[' and last ']'
    left_idx = candidate.find("[")
    right_idx = candidate.rfind("]")
    if left_idx != -1 and right_idx != -1 and right_idx > left_idx:
        json_candidate = candidate[left_idx:right_idx + 1]
    else:
        json_candidate = candidate

    parsed_list = json.loads(json_candidate)
    return parsed_list