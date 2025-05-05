import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Enhanced caching mechanism
EPISODE_CACHE = {}
IMAGE_SIZE_CACHE = {}


def hash_file(path: Path) -> str:
    """Calculate file hash for cache validation"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_episode_steps(episode_id: str, data_path: Path) -> dict:
    """Load episode steps with caching and validation, adapted to new data format"""
    cache_key = f"{episode_id}_{hash_file(data_path)}"

    if cache_key not in EPISODE_CACHE:
        try:
            # Load merged data file
            with open(data_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)

            # Extract data for specified episode
            if episode_id in all_data:
                episode_data = all_data[episode_id]
                # Validate required fields
                if not all(key in episode_data for key in ["instruction", "steps"]):
                    raise ValueError(f"Invalid episode format: {episode_id}")
                EPISODE_CACHE[cache_key] = episode_data
            else:
                print(f"Episode {episode_id} not found in data file")
                return None
        except Exception as e:
            print(f"Error loading episode {episode_id} from {data_path}: {str(e)}")
            return None

    return EPISODE_CACHE[cache_key]


def get_image_size(image_path: Path) -> tuple:
    """Get image dimensions with caching"""
    cache_key = str(image_path)
    if cache_key not in IMAGE_SIZE_CACHE:
        try:
            with Image.open(image_path) as img:
                IMAGE_SIZE_CACHE[cache_key] = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return (0, 0)
    return IMAGE_SIZE_CACHE[cache_key]


def build_step_prompt(
    step_data: dict,
    episode_data: dict,
    image_path: Path,
    action_history: list,
    examples: list,
) -> str:
    """Build prompt for a single step, adapting to original action space format"""
    img_width, img_height = (1080, 1920)
    if image_path and Path(image_path).exists():
        img_width, img_height = get_image_size(image_path)

    prompt = "You are a smartphone assistant to help users complete tasks by interacting with apps. I will give you a screenshot of the current phone screen."

    # Add few-shot examples
    if examples:
        prompt += "## Example Tasks ##\n"
        for idx, example in enumerate(examples, 1):
            prompt += (
                f"\nExample {idx}: {example['instruction']}\n"
                "Steps taken in this example:\n"
                + "\n".join(
                    [f"Step-{i+1}: {step}" for i, step in enumerate(example["steps"])]
                )
            )
    prompt += "\n\n"

    # Current task information
    prompt += "\n### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {img_width} pixels and its height is {img_height} pixels."
    prompt += f"The user's instruction is: {episode_data['instruction']}"
    prompt += "\n\n"

    if len(action_history):
        prompt += "### History operations ###\n"
        prompt += "Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These operations are as follow:\n"
        for i in range(len(action_history)):
            prompt += f"Step-{i + 1}: {action_history[i]}\n"
    prompt += "\n\n"

    prompt += "### Response requirements ###\n"
    prompt += (
        "Now you need to combine all of the above to decide just one action on the current page. "
        "You must choose one of the actions below:\n"
    )
    prompt += (
        '"SWIPE[UP]": Swipe the screen up.\n'
        '"SWIPE[DOWN]": Swipe the screen down.\n'
        '"SWIPE[LEFT]": Swipe the screen left.\n'
        '"SWIPE[RIGHT]": Swipe the screen right.\n'
    )
    prompt += '"CLICK[x,y]": Click the screen at the coordinates (x, y). x is the pixel from left to right and y is the pixel from top to bottom\n'
    prompt += '"TYPE[text]": Type the given text in the current input field.\n'
    prompt += (
        '"PRESS_BACK": Press the back button.\n'
        '"PRESS_HOME": Press the home button.\n'
        '"PRESS_ENTER": Press the enter button.\n'
        '"TASK_COMPLETE[answer]": Mark the task as complete. If the instruction requires answering a question, provide the answer inside the brackets. If no answer is needed, use empty brackets "TASK_COMPLETE[]".\n'
    )

    prompt += "\n\n"

    prompt += "### Response Example ###\n"
    prompt += (
        "Your output should be a string and nothing else, containing only the action type you choose from the list above.\n"
        "For example:\n"
        '"SWIPE[UP]"\n'
        '"CLICK[156,2067]"\n'
        '"TYPE[Rome]"\n'
        '"PRESS_BACK"\n'
        '"PRESS_HOME"\n'
        '"PRESS_ENTER"\n'
        '"TASK_COMPLETE[1h30m]"\n'
        '"TASK_COMPLETE[]"\n'
    )

    return prompt


def process_single_task(
    task_entry: dict, data_path: Path, screenshot_dir: Path
) -> dict:
    """Process a single task entry, generate data for each step, and classify by train/test"""
    try:
        # Load complete step data for query
        query_episode = load_episode_steps(task_entry["query"]["episode_id"], data_path)
        if not query_episode or not query_episode.get("steps"):
            return {"train": [], "test": []}

        # Load support set examples
        examples = []
        for support in task_entry["support"]:
            support_episode = load_episode_steps(
                support["episode"]["episode_id"], data_path
            )
            if support_episode and support_episode.get("steps"):
                examples.append(
                    {
                        "instruction": support_episode["instruction"],
                        "steps": [
                            "Action: "
                            + s["action"]
                            + " "
                            + "Action Description: "
                            + s["action_description"]
                            for s in support_episode["steps"]
                        ],
                    }
                )

        # Process each step
        results = {"train": [], "test": []}
        split = task_entry["split"]  # Get split type (train/test)

        action_history = []
        for step_idx, step in enumerate(query_episode["steps"]):
            # Get screenshot path for current step
            step_id = step.get("step_id", step_idx + 1)  # Use step_id or index+1

            image_path = None
            # Process image with screenshot_dir
            if "image_path" in step:
                image_filename = step["image_path"]
                image_path = screenshot_dir / image_filename

                # Check if file exists
                if not image_path.exists():
                    print(
                        f"Image not found: {image_path} for episode {task_entry['query']['episode_id']} step {step_id}"
                    )
                    # Continue anyway, but warn about missing image
            else:
                print(
                    f"No image_path in step data for episode {task_entry['query']['episode_id']} step {step_id}"
                )
                # Continue anyway, but warn about missing image path

            # Build prompt for current step
            current_prompt = build_step_prompt(
                step_data=step,
                episode_data=query_episode,
                image_path=image_path if image_path and image_path.exists() else None,
                action_history=action_history,
                examples=examples[: task_entry["k_shot"]],
            )

            # Build metadata
            metadata = {
                "app": task_entry["query"].get("detected_app", "unknown"),
                "domain": task_entry["query"].get("domain", ""),
                "split": split,
                "k_shot": task_entry["k_shot"],
                "support_ids": [
                    s["episode"]["episode_id"] for s in task_entry["support"]
                ],
                "similaritys": [s["similarity"] for s in task_entry["support"]],
                "similaritys_ui": [s["ui_similarity"] for s in task_entry["support"]],
                "similaritys_action": [s["action_similarity"] for s in task_entry["support"]],
                "query_goal": task_entry["query"]["goal"],
                "current_step": step_idx + 1,
                "total_steps": len(query_episode["steps"]),
            }

            # Build data entry
            result = {
                "episode_id": task_entry["query"]["episode_id"],
                "step_id": step_id,
                "query": current_prompt,
                "response": step["action"],  # Use original action format
                "action_description": step["action_description"],
                "images": [str(image_path)]
                if image_path and image_path.exists()
                else [],
                "metadata": metadata,
            }

            # Add to appropriate list based on split
            results[split].append(result)

            # Update action history
            action_history.append(
                "Action: "
                + step["action"]
                + " "
                + "Action Description: "
                + step["action_description"]
            )

        return results

    except Exception as e:
        print(f"Error processing task {task_entry['query']['episode_id']}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"train": [], "test": []}


def validate_jsonl_file(file_path: Path):
    """Validate format of generated jsonl file"""
    try:
        line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    # Validate required fields
                    assert (
                        "episode_id" in record
                    ), f"Line {line_num} missing episode_id field"
                    assert "step_id" in record, f"Line {line_num} missing step_id field"
                    assert "query" in record, f"Line {line_num} missing query field"
                    assert (
                        "response" in record
                    ), f"Line {line_num} missing response field"
                    assert "images" in record, f"Line {line_num} missing images field"
                    assert (
                        "metadata" in record
                    ), f"Line {line_num} missing metadata field"
                    line_count += 1
                except json.JSONDecodeError:
                    print(f"Line {line_num} is not valid JSON: {line[:50]}...")
                except AssertionError as e:
                    print(f"Validation error: {str(e)}")

        print(f"Validation complete: {file_path} contains {line_count} valid records")
        return line_count
    except Exception as e:
        print(f"Error validating file {file_path}: {str(e)}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate step-wise k-shot prompt dataset"
    )

    parser.add_argument(
        "--task_files",
        nargs="+",
        required=True,
        help="List of task files to process",
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to the episode data file"
    )
    parser.add_argument(
        "--screenshot_dir",
        required=True,
        help="Directory containing screenshot images",
    )
    parser.add_argument(
        "--output_dir",
        default="output/prompts",
        help="Directory to save output files (default: output/prompts)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker threads (default: 8)"
    )

    args = parser.parse_args()

    # Initialize paths
    data_path = Path(args.data_path)
    screenshot_dir = Path(args.screenshot_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split file
    for task_file_path in args.task_files:
        task_file = Path(task_file_path)

        # Extract identifier from filename for output file naming
        file_identifier = task_file.stem  # Get filename without extension

        print(f"\nProcessing file: {file_identifier}")

        # Set up train and test output files
        train_output_file = output_dir / f"{file_identifier}_train.jsonl"
        test_output_file = output_dir / f"{file_identifier}_test.jsonl"

        # Load task data
        try:
            with open(task_file, "r") as f:
                tasks = json.load(f)

            print(f"Loaded {len(tasks)} tasks")

            # Multi-threaded processing
            train_processed = 0
            test_processed = 0

            # Create output files
            train_file = open(train_output_file, "w", encoding="utf-8")
            test_file = open(test_output_file, "w", encoding="utf-8")

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_single_task, task, data_path, screenshot_dir
                    ): task
                    for task in tasks
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {file_identifier}",
                ):
                    results = future.result()

                    # Write train data (jsonl format)
                    for record in results["train"]:
                        train_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        train_processed += 1

                    # Write test data (jsonl format)
                    for record in results["test"]:
                        test_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        test_processed += 1

            # Close files
            train_file.close()
            test_file.close()

            print(
                f"Successfully processed {file_identifier}: {train_processed} train records and {test_processed} test records"
            )
            print(f"Train data saved to: {train_output_file}")
            print(f"Test data saved to: {test_output_file}")

            # Validate generated jsonl files
            print("\nValidating generated jsonl files:")
            train_count = validate_jsonl_file(train_output_file)
            test_count = validate_jsonl_file(test_output_file)

            # Verify record counts
            if train_count != train_processed:
                print(
                    f"Warning: Train record count mismatch - processed {train_processed}, validated {train_count}"
                )
            if test_count != test_processed:
                print(
                    f"Warning: Test record count mismatch - processed {test_processed}, validated {test_count}"
                )

        except Exception as e:
            print(f"Error processing file {task_file}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue  # Continue with next file

    print("\nAll files processed successfully!")


if __name__ == "__main__":
    main()
