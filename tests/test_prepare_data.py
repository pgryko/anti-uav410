"""
Unit tests for scripts/prepare_data.py

Tests the data preparation functions that convert Anti-UAV dataset to YOLO format.
"""

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import bbox_to_yolo, parse_annotation_json


class TestBboxToYolo:
    """Tests for bbox_to_yolo conversion function."""

    def test_basic_conversion(self):
        """Test basic bounding box conversion."""
        # Box at (100, 100) with size 50x50 in 640x480 image
        result = bbox_to_yolo([100, 100, 50, 50], 640, 480)

        assert result is not None
        parts = result.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Expected: center at (125, 125) -> normalized (125/640, 125/480)
        assert abs(x_center - 125 / 640) < 0.0001
        assert abs(y_center - 125 / 480) < 0.0001
        assert abs(width - 50 / 640) < 0.0001
        assert abs(height - 50 / 480) < 0.0001

    def test_center_box(self):
        """Test box at image center."""
        result = bbox_to_yolo([270, 190, 100, 100], 640, 480)

        parts = result.split()
        x_center = float(parts[1])
        y_center = float(parts[2])

        # Center should be at (320, 240) -> normalized (0.5, 0.5)
        assert abs(x_center - 0.5) < 0.0001
        assert abs(y_center - 0.5) < 0.0001

    def test_edge_box(self):
        """Test box at image edge (should be clamped)."""
        # Box partially outside image
        result = bbox_to_yolo([600, 450, 100, 100], 640, 480)

        parts = result.split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Values should be clamped to [0, 1]
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1

    def test_invalid_zero_width(self):
        """Test that zero-width box returns None."""
        result = bbox_to_yolo([100, 100, 0, 50], 640, 480)
        assert result is None

    def test_invalid_zero_height(self):
        """Test that zero-height box returns None."""
        result = bbox_to_yolo([100, 100, 50, 0], 640, 480)
        assert result is None

    def test_invalid_negative_dimensions(self):
        """Test that negative dimensions return None."""
        result = bbox_to_yolo([100, 100, -10, 50], 640, 480)
        assert result is None

        result = bbox_to_yolo([100, 100, 50, -10], 640, 480)
        assert result is None

    def test_small_box(self):
        """Test very small bounding box (typical for distant drones)."""
        result = bbox_to_yolo([300, 200, 5, 5], 640, 480)

        assert result is not None
        parts = result.split()
        width = float(parts[3])
        height = float(parts[4])

        # Should be small but valid
        assert width > 0
        assert height > 0
        assert width < 0.1
        assert height < 0.1

    def test_output_format(self):
        """Test that output has correct YOLO format."""
        result = bbox_to_yolo([100, 100, 50, 50], 640, 480)

        # Should be: "class x_center y_center width height"
        parts = result.split()
        assert len(parts) == 5

        # All values should be valid floats
        class_id = int(parts[0])
        assert class_id == 0

        for val in parts[1:]:
            f = float(val)
            assert 0 <= f <= 1


class TestParseAnnotationJson:
    """Tests for JSON annotation parsing."""

    def test_parse_valid_json(self, temp_dir):
        """Test parsing a valid annotation file."""
        annotations = {
            "exist": [1, 1, 0, 1],
            "gt_rect": [
                [100, 100, 50, 50],
                [110, 105, 48, 52],
                [0, 0, 0, 0],
                [120, 110, 46, 54],
            ],
        }

        json_path = temp_dir / "test_annotations.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)

        result = parse_annotation_json(json_path)

        assert "exist" in result
        assert "gt_rect" in result
        assert len(result["exist"]) == 4
        assert len(result["gt_rect"]) == 4
        assert result["exist"] == [1, 1, 0, 1]

    def test_parse_empty_annotations(self, temp_dir):
        """Test parsing annotations with no detections."""
        annotations = {"exist": [0, 0, 0], "gt_rect": [[0, 0, 0, 0]] * 3}

        json_path = temp_dir / "empty_annotations.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)

        result = parse_annotation_json(json_path)

        assert all(e == 0 for e in result["exist"])

    def test_parse_missing_file(self, temp_dir):
        """Test that missing file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            parse_annotation_json(temp_dir / "nonexistent.json")


class TestDataPipelineIntegration:
    """Integration tests for the data preparation pipeline."""

    def test_full_pipeline(self, mock_video_dataset, temp_dir):
        """Test the full data preparation pipeline."""
        from prepare_data import extract_frames_with_annotations

        input_dir = mock_video_dataset
        output_images = temp_dir / "images" / "train"
        output_labels = temp_dir / "labels" / "train"

        video_path = input_dir / "train" / "test_sequence" / "infrared.mp4"
        json_path = input_dir / "train" / "test_sequence" / "infrared.json"

        frames_extracted, frames_with_objects = extract_frames_with_annotations(
            video_path=video_path,
            json_path=json_path,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            sequence_name="test_sequence",
            modality="ir",
            sample_rate=1,  # Extract every frame
        )

        # Check that frames were extracted
        assert frames_extracted == 10
        assert frames_with_objects == 10

        # Check output files exist
        image_files = list(output_images.glob("*.jpg"))
        label_files = list(output_labels.glob("*.txt"))

        assert len(image_files) == 10
        assert len(label_files) == 10

        # Check label content
        sample_label = label_files[0].read_text().strip()
        parts = sample_label.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class id

    def test_sample_rate(self, mock_video_dataset, temp_dir):
        """Test that sample_rate correctly reduces frames."""
        from prepare_data import extract_frames_with_annotations

        input_dir = mock_video_dataset
        output_images = temp_dir / "images" / "train"
        output_labels = temp_dir / "labels" / "train"

        video_path = input_dir / "train" / "test_sequence" / "infrared.mp4"
        json_path = input_dir / "train" / "test_sequence" / "infrared.json"

        frames_extracted, _ = extract_frames_with_annotations(
            video_path=video_path,
            json_path=json_path,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            sequence_name="test_sequence",
            modality="ir",
            sample_rate=5,  # Extract every 5th frame
        )

        # 10 frames / 5 = 2 frames
        assert frames_extracted == 2
