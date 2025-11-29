#!/usr/bin/env python3
"""
Anti-UAV Data Preparation Script
=================================
Converts Anti-UAV dataset (videos + JSON annotations) to YOLO format.

The Anti-UAV dataset structure:
    Anti-UAV-RGBT/
    ├── train/
    │   └── <sequence_name>/
    │       ├── infrared.mp4
    │       ├── infrared.json (annotations)
    │       ├── visible.mp4
    │       └── visible.json (annotations)
    ├── val/
    └── test/

Output YOLO format:
    processed/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage:
    # Extract and convert full dataset
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed

    # Convert only infrared (thermal) images
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed --modality ir

    # Sample every 5th frame (faster, less data)
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed --sample-rate 5

Author: Anti-UAV Project
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_annotation_json(json_path: Path) -> dict:
    """
    Parse Anti-UAV JSON annotation file.

    Format:
    {
        "exist": [1, 1, 1, 0, 0, ...],  # 1 = visible, 0 = not visible
        "gt_rect": [[x, y, w, h], [x, y, w, h], ...]  # bounding boxes
    }
    """
    with open(json_path) as f:
        data = json.load(f)
    return data


def bbox_to_yolo(bbox: list[float], img_width: int, img_height: int) -> str | None:
    """
    Convert [x, y, w, h] to YOLO format [class, x_center, y_center, width, height].

    YOLO format uses normalized coordinates (0-1).
    """
    x, y, w, h = bbox

    # Skip invalid boxes
    if w <= 0 or h <= 0:
        return None

    # Convert to center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height

    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_w = max(0, min(1, norm_w))
    norm_h = max(0, min(1, norm_h))

    # Class 0 = drone
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def extract_frames_with_annotations(
    video_path: Path,
    json_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    sequence_name: str,
    modality: str,
    sample_rate: int = 1,
    max_frames: int | None = None,
) -> tuple[int, int]:
    """
    Extract frames from video and create YOLO label files.

    Returns:
        (num_frames_extracted, num_frames_with_objects)
    """
    # Parse annotations
    annotations = parse_annotation_json(json_path)
    exist_flags = annotations.get("exist", [])
    gt_rects = annotations.get("gt_rect", [])

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠️  Could not open video: {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure output directories exist
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    frames_extracted = 0
    frames_with_objects = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # Max frames limit
        if max_frames and frames_extracted >= max_frames:
            break

        # Generate filename
        filename = f"{sequence_name}_{modality}_{frame_idx:06d}"

        # Save image
        img_path = output_images_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Create label file
        label_path = output_labels_dir / f"{filename}.txt"

        # Check if object exists in this frame
        has_object = False
        if frame_idx < len(exist_flags) and exist_flags[frame_idx] == 1:
            if frame_idx < len(gt_rects):
                bbox = gt_rects[frame_idx]
                yolo_line = bbox_to_yolo(bbox, img_width, img_height)
                if yolo_line:
                    has_object = True
                    with open(label_path, "w") as f:
                        f.write(yolo_line + "\n")

        # Create empty label file if no object (optional for YOLO)
        if not has_object:
            label_path.write_text("")

        frames_extracted += 1
        if has_object:
            frames_with_objects += 1

        frame_idx += 1

    cap.release()
    return frames_extracted, frames_with_objects


def process_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    modality: str,
    sample_rate: int,
    max_frames_per_sequence: int | None,
) -> dict:
    """Process all sequences in a split (train/val/test)."""
    split_dir = input_dir / split
    if not split_dir.exists():
        print(f"  ⚠️  Split directory not found: {split_dir}")
        return {"sequences": 0, "frames": 0, "objects": 0}

    # Get all sequence directories
    sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"\n  Found {len(sequences)} sequences in {split}/")

    output_images = output_dir / "images" / split
    output_labels = output_dir / "labels" / split

    total_frames = 0
    total_objects = 0

    for seq_dir in tqdm(sequences, desc=f"  Processing {split}"):
        # Determine video and annotation files based on modality
        if modality in ["ir", "infrared", "thermal"]:
            video_file = seq_dir / "infrared.mp4"
            json_file = seq_dir / "infrared.json"
            mod_name = "ir"
        elif modality in ["rgb", "visible"]:
            video_file = seq_dir / "visible.mp4"
            json_file = seq_dir / "visible.json"
            mod_name = "rgb"
        else:
            # Both modalities
            for mod, vid, ann in [
                ("ir", "infrared.mp4", "infrared.json"),
                ("rgb", "visible.mp4", "visible.json"),
            ]:
                v_path = seq_dir / vid
                a_path = seq_dir / ann
                if v_path.exists() and a_path.exists():
                    frames, objects = extract_frames_with_annotations(
                        v_path,
                        a_path,
                        output_images,
                        output_labels,
                        seq_dir.name,
                        mod,
                        sample_rate,
                        max_frames_per_sequence,
                    )
                    total_frames += frames
                    total_objects += objects
            continue

        # Single modality processing
        if video_file.exists() and json_file.exists():
            frames, objects = extract_frames_with_annotations(
                video_file,
                json_file,
                output_images,
                output_labels,
                seq_dir.name,
                mod_name,
                sample_rate,
                max_frames_per_sequence,
            )
            total_frames += frames
            total_objects += objects
        else:
            print(f"  ⚠️  Missing files in {seq_dir.name}")

    return {"sequences": len(sequences), "frames": total_frames, "objects": total_objects}


def create_dataset_yaml(output_dir: Path, dataset_name: str = "drone") -> Path:
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Anti-UAV Dataset Configuration
# Auto-generated by prepare_data.py

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['drone']

# Dataset info
# Generated from Anti-UAV-RGBT dataset
"""
    yaml_path = output_dir / f"{dataset_name}.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Anti-UAV dataset to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/raw/Anti-UAV-RGBT"),
        help="Input directory containing Anti-UAV dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for YOLO format data",
    )
    parser.add_argument(
        "--modality",
        "-m",
        choices=["ir", "rgb", "both"],
        default="ir",
        help="Which modality to extract (default: ir for thermal)",
    )
    parser.add_argument(
        "--sample-rate", "-s", type=int, default=5, help="Extract every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per sequence (default: unlimited)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to process (default: train val)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Anti-UAV Data Preparation")
    print("=" * 60)
    print(f"\n  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Modality:    {args.modality}")
    print(f"  Sample rate: every {args.sample_rate} frames")
    print(f"  Splits:      {', '.join(args.splits)}")

    # Check input exists
    if not args.input.exists():
        # Try to extract from zip
        zip_path = args.input.with_suffix(".zip")
        if not zip_path.exists():
            zip_path = args.input.parent / "Anti-UAV-RGBT.zip"

        if zip_path.exists():
            print(f"\n📦 Extracting {zip_path}...")
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(args.input.parent)
            print("  ✅ Extraction complete")
        else:
            print(f"\n❌ Error: Input not found: {args.input}")
            print(f"   Also tried: {zip_path}")
            sys.exit(1)

    # Process each split
    stats = {}
    for split in args.splits:
        stats[split] = process_split(
            args.input, args.output, split, args.modality, args.sample_rate, args.max_frames
        )

    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_frames = 0
    total_objects = 0
    for split, s in stats.items():
        print(f"\n  {split}:")
        print(f"    Sequences: {s['sequences']}")
        print(f"    Frames:    {s['frames']}")
        print(f"    With UAV:  {s['objects']}")
        total_frames += s["frames"]
        total_objects += s["objects"]

    print(f"\n  Total frames:  {total_frames}")
    print(f"  Total with UAV: {total_objects}")
    print(f"  Dataset YAML:  {yaml_path}")

    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("\nNext steps:")
    print(f"  1. Verify images in: {args.output / 'images'}")
    print(f"  2. Verify labels in: {args.output / 'labels'}")
    print(f"  3. Train with: python scripts/train.py --data {yaml_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
