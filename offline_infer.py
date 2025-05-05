import sys
import time
import json
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PIL import Image

# Get absolute path of the current script and add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Replace vivo_api import with llm import
from utils.llm import LLMClient

# Import unified utility functions
from utils.offline_infer_utils import (
    clean_response,
    parse_action,
    is_action_match,
    get_image_dimensions,
    calculate_f1_score,
)

# Thread lock for safe writing
file_lock = threading.Lock()

# Image cache to avoid repeated loading
image_cache = {}

# Initialize LLM instance
llm_instance = LLMClient()


def read_jsonl(path):
    """Read JSONL file and preserve complete metadata"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                item = json.loads(line)
                # Standardize metadata field
                item["metadata"] = item.get("metadata", {})
                data.append(item)
            except json.JSONDecodeError:
                print(f"Invalid JSON line: {line.strip()}")
    return data


def process_item(item, output_file):
    """Process a single data item"""
    try:
        # Preserve complete metadata
        metadata = item.get("metadata", {})

        # Path processing - ensure image path is correct
        image_path = (
            item["images"][0] if isinstance(item["images"], list) else item["images"]
        )

        # Ensure image path exists
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Warning: Image does not exist: {image_path}")
            return False, False

        response = llm_instance.get_response_vlm(
            prompt=item["query"],
            image_path=str(image_path),
            model="gpt-4o-mini",  # You may need to adjust this based on your configuration
            max_retries=3,
            retry_delay=2,
        )

        if not response:
            print(
                f"Warning: Model returned empty response, episode_id={item['episode_id']}, step_id={item['step_id']}"
            )
            return False, False

        # Clean and validate
        cleaned_pred = clean_response(response)
        gt_action = item["response"]

        # Parse actions
        pred_parts = parse_action(cleaned_pred)
        gt_parts = parse_action(gt_action)

        # Use is_action_match with image path
        full_match, type_match = is_action_match(
            cleaned_pred, gt_action, str(image_path)
        )

        # Special handling for TASK_COMPLETE actions
        if gt_parts and gt_parts[0] == "TASK_COMPLETE":
            # Add TASK_COMPLETE info to metadata
            metadata["task_complete_info"] = {
                "gt_action": gt_action,
                "gt_parsed": str(gt_parts),
                "pred_action": cleaned_pred,
                "pred_parsed": str(pred_parts),
            }

        # Special handling for SWIPE actions
        if gt_parts and gt_parts[0] == "SWIPE":
            # Add SWIPE info to metadata
            metadata["swipe_info"] = {
                "gt_action": gt_action,
                "gt_parsed": str(gt_parts),
                "pred_action": cleaned_pred,
                "pred_parsed": str(pred_parts),
            }

            # If prediction is also a SWIPE, record direction match status
            if pred_parts and pred_parts[0] == "SWIPE":
                pred_direction = pred_parts[1] if len(pred_parts) > 1 else ""
                gt_direction = gt_parts[1] if len(gt_parts) > 1 else ""
                direction_match = pred_direction == gt_direction
                metadata["swipe_info"]["direction_match"] = direction_match

        # Build result record
        result = {
            **{k: v for k, v in item.items() if k != "metadata"},
            "metadata": {
                **metadata,
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "gpt-4o-mini",  # Updated model name
                "image_width": get_image_dimensions(str(image_path))[0]
                if image_path.exists()
                else None,
            },
            "prediction": cleaned_pred,
            "ground_truth": gt_action,
            "action_type_correct": type_match,
            "action_fully_correct": full_match,
        }

        # Safe write
        with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return full_match, type_match

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return False, False


def analyze_results(output_path):
    """Calculate and display overall type and full accuracy"""
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    # Read results file
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                continue

    if not results:
        print("No results data found for analysis")
        return

    # Calculate overall statistics
    total_samples = len(results)
    type_correct = sum(1 for r in results if r.get("action_type_correct", False))
    full_correct = sum(1 for r in results if r.get("action_fully_correct", False))

    type_acc = type_correct / total_samples if total_samples > 0 else 0
    full_acc = full_correct / total_samples if total_samples > 0 else 0

    # Print overall accuracy
    print(f"Total samples: {total_samples}")
    print(f"Type Accuracy: {type_acc:.4f} ({type_correct}/{total_samples})")
    print(f"Full Accuracy: {full_acc:.4f} ({full_correct}/{total_samples})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on smartphone assistant tasks"
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input JSONL file containing test data",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to save results (default: input_filename_results.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for inference (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker threads (default: 8)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    # If output file is not specified, create one based on input filename
    if args.output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_results.jsonl"
    else:
        output_path = Path(args.output_file)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # Initialize progress file
    progress_path = output_path.with_suffix(".progress")
    processed = set()
    if progress_path.exists():
        with open(progress_path) as f:
            processed = set(line.strip() for line in f)

    # Process data
    data = read_jsonl(input_path)
    total = len(data)

    print(f"Loaded {total} items, starting processing...")

    # Check how many items have already been processed
    need_process = []
    for item in data:
        item_id = f"{item['episode_id']}_{item['step_id']}"
        if item_id not in processed:
            need_process.append(item)

    # If all data has been processed
    if not need_process:
        print(f"All {total} items have been processed, no need to reprocess")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Analyzing existing results file: {output_path}")
            analyze_results(output_path)
        return

    print(f"Found {len(need_process)} unprocessed items, starting processing...")

    # Statistics variables
    total_processed = 0
    type_correct = 0
    fully_correct = 0

    # Create result file (if it doesn't exist)
    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8") as f:
            pass

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, item in enumerate(need_process):
            item_id = f"{item['episode_id']}_{item['step_id']}"
            future = executor.submit(process_item, item, output_path)
            futures[future] = (idx, item_id)

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    idx, item_id = futures[future]
                    full_match, type_match = future.result()

                    # Update statistics
                    total_processed += 1
                    if type_match:
                        type_correct += 1
                    if full_match:
                        fully_correct += 1

                    # Update progress
                    with open(progress_path, "a") as f:
                        f.write(f"{item_id}\n")

                    # Display real-time accuracy
                    type_acc = (
                        type_correct / total_processed if total_processed > 0 else 0
                    )
                    full_acc = (
                        fully_correct / total_processed if total_processed > 0 else 0
                    )

                    # Update progress bar description
                    pbar.set_description(
                        f"Processing | Type Acc: {type_acc:.4f} | Full Acc: {full_acc:.4f}"
                    )
                    pbar.update(1)

                    # Print detailed info every 10 samples
                    if total_processed % 10 == 0:
                        print(
                            f"\nProcessed: {total_processed}/{len(futures)} | Type Acc: {type_acc:.4f} | Full Acc: {full_acc:.4f}"
                        )

                except Exception as e:
                    print(f"Processing failed: {str(e)}")
                    pbar.update(1)

    # Print final results
    if total_processed > 0:
        final_type_acc = type_correct / total_processed
        final_full_acc = fully_correct / total_processed
        print(f"\nProcessing complete! Processed {total_processed} items")
        print(f"Type Accuracy: {final_type_acc:.4f} ({type_correct}/{total_processed})")
        print(
            f"Full Accuracy: {final_full_acc:.4f} ({fully_correct}/{total_processed})"
        )

        # Analyze results file
        analyze_results(output_path)
    else:
        print("No items were processed")
        # Check if results file exists and is not empty
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Found existing results file: {output_path}")
            print(f"Analyzing existing results...")
            analyze_results(output_path)


if __name__ == "__main__":
    main()
