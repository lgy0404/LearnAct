import re
import json
import math
from difflib import SequenceMatcher
from PIL import Image

# Image cache to avoid repeated loading
image_cache = {}


def get_image_dimensions(image_path):
    """Get image dimensions"""
    global image_cache

    # Check cache
    if image_path in image_cache:
        return image_cache[image_path]

    try:
        with Image.open(image_path) as img:
            dimensions = img.size  # (width, height)
            image_cache[image_path] = dimensions
            return dimensions
    except Exception as e:
        print(f"Error reading image dimensions: {str(e)}")
        # Return default dimensions
        return (1440, 3040)


def calculate_f1_score(pred_text, gt_text):
    """Calculate F1 score for text"""
    if not pred_text and not gt_text:
        return 1.0  # Both empty, perfect match

    if not pred_text or not gt_text:
        return 0.0  # One empty, one not

    # Split text into tokens
    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())

    # Calculate intersection
    intersection = pred_tokens.intersection(gt_tokens)

    # Calculate precision and recall
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
    recall = len(intersection) / len(gt_tokens) if gt_tokens else 0

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def clean_response(response):
    """
    Enhanced response cleaning logic to handle various complex formats

    Args:
        response: Original response from the model

    Returns:
        Cleaned action string
    """
    if not response:
        return ""

    try:
        # First try to parse JSON (handling formats like [{"text":"\"CLICK[628,79]\""}])
        try:
            # If the response is a JSON formatted string
            parsed = json.loads(response)

            # If it's a list, take the first element
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]

            # If it's a dictionary, try to get the text content
            if isinstance(parsed, dict):
                # Common key names: text, content, response, action
                for key in ["text", "content", "response", "action", "message"]:
                    if key in parsed:
                        response = str(parsed[key])
                        break
        except json.JSONDecodeError:
            pass  # Not JSON format, continue processing
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
    except Exception as e:
        print(f"Error preprocessing JSON: {str(e)}")

    # Remove extra escape characters
    response = response.replace('\\"', '"').replace("\\n", " ")

    # Standard cleaning process
    response = response.strip()

    # Apply regex cleaning
    patterns = [
        r"^```(?:json|bash|text)?\s*",  # Remove code block markers at the beginning
        r"```\s*$",  # Remove code block markers at the end
        r'^[\'"]+',  # Remove quotes at the beginning
        r'[\'"]+$',  # Remove quotes at the end
        r"\b(?:please|action:|step:|I will|I'll|Let me)\b.*?(?=CLICK|SWIPE|TYPE|PRESS|TASK_|$)",  # Remove extra explanatory text
        r"^\s*\{.*?\}\s*$",  # Remove complete JSON objects
        r"^\s*\[.*?\]\s*$",  # Remove complete JSON arrays
    ]

    for pattern in patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)

    # Extract the main action
    action_pattern = r"(CLICK\[\d+,\d+\]|SWIPE\[\w+\]|TYPE\[.*?\]|PRESS_BACK|PRESS_HOME|PRESS_ENTER|TASK_COMPLETE(?:\[.*?\])?|TASK_IMPOSSIBLE)"
    action_match = re.search(action_pattern, response, re.IGNORECASE | re.DOTALL)

    if action_match:
        response = action_match.group(0)

    return response.strip()


def parse_action(action):
    """
    Parse action string, supporting the original action space

    Args:
        action: Action string

    Returns:
        Parsed action tuple, format (action_type, param1, param2, ...)
    """
    if not action:
        return None

    # Clean action string
    action = action.strip()

    # Parse CLICK action
    click_match = re.search(r"CLICK\[(\d+),(\d+)\]", action, re.IGNORECASE)
    if click_match:
        x, y = click_match.groups()
        return ("CLICK", int(x), int(y))

    # Parse SWIPE action
    swipe_match = re.search(r"SWIPE\[(\w+)\]", action, re.IGNORECASE)
    if swipe_match:
        direction = swipe_match.group(1).upper()
        return ("SWIPE", direction)

    # Parse TYPE action
    type_match = re.search(r"TYPE\[(.*?)\]", action, re.DOTALL)
    if type_match:
        content = type_match.group(1)
        return ("TYPE", content)

    # Parse PRESS_BACK action
    if action == "PRESS_BACK":
        return ("PRESS_BACK",)

    # Parse PRESS_HOME action
    if action == "PRESS_HOME":
        return ("PRESS_HOME",)

    # Parse PRESS_ENTER action
    if action == "PRESS_ENTER":
        return ("PRESS_ENTER",)

    # Parse TASK_COMPLETE action
    task_complete_match = re.search(r"TASK_COMPLETE\[(.*?)\]", action, re.DOTALL)
    if task_complete_match:
        answer = task_complete_match.group(1)
        return ("TASK_COMPLETE", answer)

    # Compatibility with old format
    if action == "TASK_COMPLETE":
        return ("TASK_COMPLETE", "")

    # Parse TASK_IMPOSSIBLE action
    if action == "TASK_IMPOSSIBLE":
        return ("TASK_IMPOSSIBLE",)

    # If parsing fails, return the original string
    return (action,)


def is_action_match(
    pred, gt, image_path=None, tolerance_ratio=0.14, text_f1_threshold=0.5
):
    """
    Determine if the predicted action matches the ground truth action

    Args:
        pred: Predicted action string
        gt: Ground truth action string
        image_path: Image path, used to get screen dimensions
        tolerance_ratio: Coordinate tolerance ratio (percentage of screen width)
        text_f1_threshold: Text F1 score threshold

    Returns:
        (full_match, type_match): Boolean values indicating full match and type match
    """
    # Parse actions
    pred_parts = parse_action(pred)
    gt_parts = parse_action(gt)

    if not pred_parts or not gt_parts:
        return False, False

    # Get action types
    pred_type = pred_parts[0]
    gt_type = gt_parts[0]

    # Check if types match
    type_match = pred_type == gt_type

    # Return early if types don't match
    if not type_match:
        return False, False

    # Get screen dimensions
    screen_width, screen_height = (1440, 3040)  # Default dimensions
    if image_path:
        try:
            screen_width, screen_height = get_image_dimensions(image_path)
        except Exception as e:
            print(f"Failed to get image dimensions: {str(e)}")

    # Calculate click tolerance in pixels
    tolerance_pixels = int(screen_width * tolerance_ratio)

    # Check for full match based on action type

    # Click action: coordinates within tolerance
    if gt_type == "CLICK":
        try:
            x1, y1 = pred_parts[1], pred_parts[2]
            x2, y2 = gt_parts[1], gt_parts[2]
            # Calculate Euclidean distance between points
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # Match if distance is within tolerance
            return distance <= tolerance_pixels, True
        except Exception as e:
            print(f"Error checking CLICK action: {str(e)}")
            return False, True

    # Long press action: check coordinates only, not duration
    elif gt_type == "LONG_PRESS":
        try:
            x1, y1 = pred_parts[1], pred_parts[2]
            x2, y2 = gt_parts[1], gt_parts[2]
            # Calculate Euclidean distance between points
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # Match if distance is within tolerance
            return distance <= tolerance_pixels, True
        except Exception as e:
            print(f"Error checking LONG_PRESS action: {str(e)}")
            return False, True

    # Type action: text F1 score
    elif gt_type == "TYPE":
        try:
            pred_text = pred_parts[1].lower() if len(pred_parts) > 1 else ""
            gt_text = gt_parts[1].lower() if len(gt_parts) > 1 else ""

            # Calculate F1 score
            f1_score = calculate_f1_score(pred_text, gt_text)
            return f1_score >= text_f1_threshold, True
        except Exception as e:
            print(f"Error checking TYPE action: {str(e)}")
            return False, True

    # Swipe action: check if directions match
    elif gt_type == "SWIPE":
        try:
            # Extract directions
            pred_direction = pred_parts[1] if len(pred_parts) > 1 else ""
            gt_direction = gt_parts[1] if len(gt_parts) > 1 else ""

            # Match if directions are identical
            direction_match = pred_direction == gt_direction
            return direction_match, True
        except Exception as e:
            print(f"Error checking SWIPE action: {str(e)}")
            return False, True

    # TASK_COMPLETE action: check type and content
    elif gt_type == "TASK_COMPLETE":
        try:
            # If predicted type is not TASK_COMPLETE, type match but not full match
            if pred_type != "TASK_COMPLETE":
                return False, True

            # If action type is correct, count as fully correct
            return True, True
        except Exception as e:
            print(f"Error checking TASK_COMPLETE action: {str(e)}")
            return False, True

    # Other simple actions: press_back, press_home, press_enter
    elif gt_type in ["PRESS_BACK", "PRESS_HOME", "PRESS_ENTER"]:
        return True, True

    # Unknown action type
    return False, True
