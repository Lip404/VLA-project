"""
Unit test for structured planner json parser and schema
Test cases: valid input, json wrapped in markdown, malformed json, invalid action type
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from vla.structured_planner import parse_video_behavior
 
 
def test_valid_json():
    """Test normal valid structured output"""
    raw = '''
    {
        "video_summary": "The robot sees a red cup on the table",
        "objects_in_scene": ["red cup", "table"],
        "action_sequence": [
            {
                "action": "grasp",
                "target": "red cup",
                "confidence": 0.87
            }
        ]
    }
    '''
    res, err = parse_video_behavior(raw)
    print("[PASS] test_valid_json")
    print("parsed result:", res)
    print("err msg:", err)
    assert err is not None and len(err) == 0
 
 
def test_markdown_json():
    """Test json wrapped inside ```json ... ``` block (common LLM output)"""
    raw = '''
Here is the observation result:
```json
{
    "video_summary": "The block is in front of the robot",
    "objects_in_scene": ["cube block", "robot"],
    "action_sequence": [
        {
            "action": "move_forward",
            "target_object": "cube block"
        }
    ]
}
```
'''
    res, err = parse_video_behavior(raw)
    print("\n[PASS] test_markdown_json")
    print("parsed result:", res)
    assert len(err) == 0
 
 
def test_malformed_json():
    """Test broken json, expect error captured"""
    raw = '''
{
"video_summary": "broken test",
"objects_in_scene": [],
"action_sequence": [
    {
        "action": "look",
        "confidence":
    }
]
}
'''
    res, err = parse_video_behavior(raw)
    print("\n[PASS] test_malformed_json")
    print("error msg:", err)
    assert len(err) > 0
 
 
def test_missing_required_field():
    """Test missing required key (video_summary), schema validation fail"""
    raw = '''
{
"objects_in_scene": ["door"],
"action_sequence": [
    {
        "action": "rotate",
        "target": "door",
        "confidence": 0.71
    }
]
}
'''
    res, err = parse_video_behavior(raw)
    print("\n[PASS] test_missing_required_field")
    print("error msg:", err)
    assert len(err) > 0
 
 
def test_invalid_action_name():
    """Test action not in VALID_OBSERVED_ACTIONS"""
    raw = '''
{
    "video_summary": "test invalid action",
    "objects_in_scene": ["cup"],
    "action_sequence": [
        {
            "action": "fly_away",
            "target": "cup",
            "confidence": 0.8
        }
    ]
}
'''
    res, err = parse_video_behavior(raw)
    print("\n[PASS] test_invalid_action_name")
    print("error msg:", err)
    assert len(err) > 0
 
 
if __name__ == "__main__":
    test_valid_json()
    test_markdown_json()
    test_malformed_json()
    test_missing_required_field()
    test_invalid_action_name()
    print("\n✅ All test cases finished")