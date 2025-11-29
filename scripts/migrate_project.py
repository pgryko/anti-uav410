#!/usr/bin/env python3
"""
Project Migration Script
========================
Migrates the old Anti-UAV410 project structure to the new organized layout.

Usage:
    python scripts/migrate_project.py --dry-run    # Preview changes
    python scripts/migrate_project.py              # Execute migration

Author: Anti-UAV Project
"""

import argparse
import os
import shutil
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.absolute()


# Migration mapping: (source, destination, action)
# action: 'copy' = copy files, 'move' = move files, 'merge' = merge into existing
MIGRATIONS: list[tuple[str, str, str]] = [
    # Detection module
    ("Codes/detect_wrapper/Detectoruav.py", "src/detection/detector.py", "copy"),
    ("Codes/detect_wrapper/detect_drone.py", "src/detection/detect.py", "copy"),
    ("Codes/detect_wrapper/models/common.py", "src/detection/models/common.py", "copy"),
    ("Codes/detect_wrapper/models/detect_model.py", "src/detection/models/yolo.py", "copy"),
    ("Codes/detect_wrapper/models/experimental.py", "src/detection/models/experimental.py", "copy"),
    ("Codes/detect_wrapper/utils/general.py", "src/detection/utils/general.py", "copy"),
    ("Codes/detect_wrapper/utils/torch_utils.py", "src/detection/utils/torch_utils.py", "copy"),
    ("Codes/detect_wrapper/utils/datasets.py", "src/detection/utils/datasets.py", "copy"),
    ("Codes/detect_wrapper/utils/activations.py", "src/detection/utils/activations.py", "copy"),
    # Training scripts
    ("Codes/detect_wrapper/train_drone.py", "scripts/train.py", "copy"),
    ("Codes/detect_wrapper/test_drone.py", "scripts/evaluate.py", "copy"),
    # Tracking module
    ("Codes/tracking_wrapper/runtracker.py", "src/tracking/runner.py", "copy"),
    ("Codes/tracking_wrapper/dronetracker", "src/tracking/dronetracker", "copy"),
    ("Codes/tracking_wrapper/drtracker", "src/tracking/drtracker", "copy"),
    # Evaluation
    ("anti_uav_jittor/pysot_toolkit/test.py", "src/evaluation/test_tracker.py", "copy"),
    ("anti_uav_jittor/pysot_toolkit/eval.py", "src/evaluation/eval.py", "copy"),
    ("anti_uav_jittor/pysot_toolkit/bbox.py", "src/evaluation/bbox.py", "copy"),
    ("anti_uav_jittor/pysot_toolkit/toolkit", "src/evaluation/toolkit", "copy"),
    # Configs
    ("Codes/detect_wrapper/data/drone.yaml", "configs/data/drone.yaml", "copy"),
    ("Codes/detect_wrapper/data/hyp.scratch.yaml", "configs/training/default.yaml", "copy"),
    ("Codes/detect_wrapper/data/hyp.finetune.yaml", "configs/training/finetune.yaml", "copy"),
    ("Codes/detect_wrapper/models/detects.yaml", "configs/models/yolov5s.yaml", "copy"),
    ("Codes/detect_wrapper/models/detectm.yaml", "configs/models/yolov5m.yaml", "copy"),
    ("Codes/detect_wrapper/models/detectl.yaml", "configs/models/yolov5l.yaml", "copy"),
    ("Codes/detect_wrapper/models/detectx.yaml", "configs/models/yolov5x.yaml", "copy"),
    # Notebooks
    ("anti_uav_jittor/demo.ipynb", "notebooks/demo.ipynb", "copy"),
    # Documentation images
    ("Fig", "docs/images", "copy"),
    # Datasets (for reference, actual data stays in data/)
    ("anti_uav_jittor/anti_uav410_jit/datasets/antiuav410.py", "src/data/antiuav410.py", "copy"),
    # Data converter
    ("cvat_converter.py", "src/data/cvat_converter.py", "copy"),
    # Archive legacy code
    ("Codes/CameralinkApplication", "archive/CameralinkApplication", "move"),
    ("Codes/metric_uav", "archive/metric_uav", "move"),
]

# Files/directories to delete after migration
CLEANUP: list[str] = [
    "Codes/testvideo",
    "Codes/result",
    "anti_uav_jittor/result",
    "anti_uav_jittor/dectect",  # Typo directory
    "anti_uav_jittor/model_1",
]

# Data files to move to data/ directory
DATA_MOVES: list[tuple[str, str]] = [
    ("Anti-UAV-RGBT.zip", "data/raw/Anti-UAV-RGBT.zip"),
    ("Codes/detect_wrapper/drone_data", "data/samples/drone_data"),
]


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def copy_item(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Copy file or directory."""
    if not src.exists():
        print(f"  ⚠️  Source not found: {src}")
        return False

    if dry_run:
        print(f"  📋 Would copy: {src} → {dst}")
        return True

    ensure_dir(dst.parent)

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

    print(f"  ✅ Copied: {src.name} → {dst}")
    return True


def move_item(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Move file or directory."""
    if not src.exists():
        print(f"  ⚠️  Source not found: {src}")
        return False

    if dry_run:
        print(f"  📦 Would move: {src} → {dst}")
        return True

    ensure_dir(dst.parent)

    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    shutil.move(str(src), str(dst))
    print(f"  ✅ Moved: {src.name} → {dst}")
    return True


def delete_item(path: Path, dry_run: bool = False) -> bool:
    """Delete file or directory."""
    if not path.exists():
        return True

    if dry_run:
        print(f"  🗑️  Would delete: {path}")
        return True

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    print(f"  🗑️  Deleted: {path}")
    return True


def create_init_files(dry_run: bool = False) -> None:
    """Create __init__.py files in all src directories."""
    src_dir = ROOT / "src"

    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirpath = Path(dirpath)
        init_file = dirpath / "__init__.py"

        if not init_file.exists():
            if dry_run:
                print(f"  📄 Would create: {init_file}")
            else:
                init_file.write_text(f'"""{dirpath.name} module."""\n')
                print(f"  📄 Created: {init_file}")


def create_gitkeep_files(dry_run: bool = False) -> None:
    """Create .gitkeep files in empty directories."""
    keep_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "weights",
        "weights/pretrained",
        "weights/trained",
        "outputs",
        "outputs/runs",
        "outputs/results",
    ]

    for dir_name in keep_dirs:
        dir_path = ROOT / dir_name
        gitkeep = dir_path / ".gitkeep"

        if not gitkeep.exists():
            if dry_run:
                print(f"  📄 Would create: {gitkeep}")
            else:
                ensure_dir(dir_path)
                gitkeep.write_text("")
                print(f"  📄 Created: {gitkeep}")


def run_migration(dry_run: bool = False) -> None:
    """Execute the full migration."""
    print("\n" + "=" * 60)
    print("Anti-UAV410 Project Migration")
    print("=" * 60)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")

    # Step 1: Copy/move main files
    print("\n📁 Step 1: Migrating source files...")
    success_count = 0
    for src, dst, action in MIGRATIONS:
        src_path = ROOT / src
        dst_path = ROOT / dst

        if action == "copy":
            if copy_item(src_path, dst_path, dry_run):
                success_count += 1
        elif action == "move":
            if move_item(src_path, dst_path, dry_run):
                success_count += 1

    print(f"\n  Migrated {success_count}/{len(MIGRATIONS)} items")

    # Step 2: Move data files
    print("\n📦 Step 2: Moving data files...")
    for src, dst in DATA_MOVES:
        src_path = ROOT / src
        dst_path = ROOT / dst
        move_item(src_path, dst_path, dry_run)

    # Step 3: Create __init__.py files
    print("\n📄 Step 3: Creating Python package files...")
    create_init_files(dry_run)

    # Step 4: Create .gitkeep files
    print("\n📄 Step 4: Creating .gitkeep files...")
    create_gitkeep_files(dry_run)

    # Step 5: Cleanup
    print("\n🗑️  Step 5: Cleaning up old directories...")
    for path_str in CLEANUP:
        path = ROOT / path_str
        delete_item(path, dry_run)

    print("\n" + "=" * 60)
    if dry_run:
        print("✅ Dry run complete. Run without --dry-run to apply changes.")
    else:
        print("✅ Migration complete!")
        print("\nNext steps:")
        print("  1. Review the new structure")
        print("  2. Run: pip install -e .")
        print("  3. Test imports: python -c 'from src.detection import detector'")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Migrate Anti-UAV410 project to new structure")
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Preview changes without applying them"
    )
    parser.add_argument(
        "--skip-cleanup", action="store_true", help="Skip deletion of old directories"
    )

    args = parser.parse_args()

    if args.skip_cleanup:
        global CLEANUP
        CLEANUP = []

    run_migration(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
