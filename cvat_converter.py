import os
import json
import cv2
import xml.etree.ElementTree as ET

# Annotation Converter for Anti-UAV / RGBT Drone Datasets

# This script automatically converts Anti-UAV–style JSON tracking annotations into CVAT-compatible XML files (format: CVAT for Video 1.1).
# It is designed to batch-process large datasets containing hundreds of video/JSON pairs.

# ✔ Supported annotation structure

# The converter reads tracking-style JSON files such as the one in this dataset
# (which includes exist flags and gt_rect bounding boxes)

# .

# It automatically detects the correct bounding-box field among:

# gt_rect, rgb_rect, ir_rect, visible_rect,
# gt, bbox, target, annotations, rect, boxes

# ✔ Frame skipping with exist

# Frames where exist = 0 are ignored, meaning no <box> element is created for that frame.
# This avoids generating empty or false annotations in the final XML.

# ✔ Output format

# For each video:

# the script generates one XML file

# named exactly like the video (e.g., infrared.xml)

# placed in the same folder as the video and JSON

# overwriting previous XML files

# ✔ Automatic resolution detection

# Video width/height is extracted directly from the video file
# (.mp4, .mov, .avi, .mkv supported).

# ✔ Recursive batch processing

# The script:

# walks through all subfolders

# finds matching video.* + video.json pairs

# generates XML automatically

# handles multiple videos inside the same folder

# ✔ CVAT-ready output

# The produced XML can be directly uploaded into CVAT using:

# UI:

# Upload Annotations → CVAT for Video 1.1


# or the CVAT REST API, which this script is designed to integrate with in the next stage.

# ------------------------------
# Detect valid annotation keys
# ------------------------------
def extract_boxes(data):
    POSSIBLE_KEYS = [
        "gt_rect", "rgb_rect", "ir_rect", "visible_rect",
        "gt", "bbox", "target", "annotations", "rect", "boxes"
    ]

    for key in POSSIBLE_KEYS:
        if key in data and isinstance(data[key], list):
            print(f"   ✔ Using bbox key: {key}")
            return data[key]

    print("   ❌ No valid box array found")
    return []


# ------------------------------
# Auto-detect tracked/exist flags
# ------------------------------
def extract_exist(data):
    if "exist" in data and isinstance(data["exist"], list):
        print("   ✔ Using exist flags")
        return data["exist"]
    return None


# ------------------------------
# Get video resolution
# ------------------------------
def get_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"   ✔ Video resolution: {w}x{h}")
    return w, h


# ------------------------------
# Create CVAT XML file
# ------------------------------
def create_xml(folder, xml_name, width, height, boxes, exist):
    xml_path = os.path.join(folder, xml_name)
    print(f"   ⚠️ Overwriting: {xml_path}")

    root = ET.Element("annotations")

    # meta
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = xml_name
    ET.SubElement(task, "size").text = str(len(boxes))

    labels = ET.SubElement(task, "labels")
    lbl = ET.SubElement(labels, "label")
    ET.SubElement(lbl, "name").text = "drone"
    ET.SubElement(lbl, "color").text = "#ff0000"
    ET.SubElement(lbl, "type").text = "rectangle"

    # Track ID
    track = ET.SubElement(root, "track", {
        "id": "0",
        "label": "drone"
    })

    for idx, rect in enumerate(boxes):

        # Skip frames where exist flag = 0
        if exist and exist[idx] == 0:
            continue

        # Skip invalid
        if not isinstance(rect, list) or len(rect) != 4:
            continue

        x, y, w, h = rect
        if w <= 0 or h <= 0:
            continue

        xtl, ytl = x, y
        xbr, ybr = x + w, y + h

        ET.SubElement(track, "box", {
            "frame": str(idx),
            "xtl": str(xtl),
            "ytl": str(ytl),
            "xbr": str(xbr),
            "ybr": str(ybr),
            "outside": "0",
            "occluded": "0",
            "keyframe": "1"
        })

    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"   ✔ Saved: {xml_path}\n")


# ------------------------------
# Process all folders
# ------------------------------
def process_all(base="."):
    print("🚀 Starting updated conversion...\n")

    for root, dirs, files in os.walk(base):

        # All video extensions supported for later API upload
        video_files = [f for f in files if f.lower().endswith(
            (".mp4", ".mov", ".avi", ".mkv")
        )]

        json_files = [f for f in files if f.endswith(".json")]

        for vid in video_files:

            basename = vid.rsplit(".", 1)[0]
            json_name = basename + ".json"

            if json_name not in json_files:
                continue

            video_path = os.path.join(root, vid)
            json_path = os.path.join(root, json_name)

            print(f"⚡ Pair found:")
            print(f"   VIDEO → {video_path}")
            print(f"   JSON  → {json_path}")

            with open(json_path, "r") as f:
                data = json.load(f)

            boxes = extract_boxes(data)
            exist = extract_exist(data)
            width, height = get_resolution(video_path)

            xml_name = basename + ".xml"

            create_xml(root, xml_name, width, height, boxes, exist)

    print("🏁 Conversion finished.\n")


if __name__ == "__main__":
    process_all(".")
