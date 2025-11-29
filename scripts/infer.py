"""Inference script for Anti-UAV detection.

This script performs UAV detection on images, videos, or streams using trained YOLO models.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "detection"))

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (
    check_img_size,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, time_synchronized


def detect(
    weights: str = "weights/best.pt",
    source: str = "data/images",
    output: str = "runs/detect",
    img_size: int = 640,
    conf_thres: float = 0.5,
    iou_thres: float = 0.5,
    device: str = "",
    view_img: bool = False,
    save_txt: bool = False,
    save_conf: bool = False,
    classes: list = None,
    agnostic_nms: bool = False,
    augment: bool = False,
) -> None:
    """Run detection on images, videos, or streams.

    Args:
        weights: Path to model weights file(s).
        source: Input source - file, folder, 0 for webcam, or stream URL.
        output: Directory to save results.
        img_size: Inference image size (pixels).
        conf_thres: Object confidence threshold.
        iou_thres: IOU threshold for NMS.
        device: CUDA device (e.g., '0' or '0,1,2,3' or 'cpu').
        view_img: Display results in window.
        save_txt: Save results to text files.
        save_conf: Save confidence scores in text output.
        classes: Filter by class indices.
        agnostic_nms: Class-agnostic NMS.
        augment: Augmented inference.
    """
    webcam = (
        source.isnumeric()
        or source.startswith(("rtsp://", "rtmp://", "http://"))
        or source.endswith(".txt")
    )

    # Initialize
    set_logging()
    device = select_device(device)
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    half = device.type != "cpu"

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != "cpu" else None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
        )
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], f"{i}: ", im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            save_path = str(Path(output) / Path(p).name)
            txt_path = str(Path(output) / Path(p).stem) + (
                f"_{dataset.frame}" if dataset.mode == "video" else ""
            )
            s += f"{img.shape[2]}x{img.shape[3]} "
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, conf, *xywh) if save_conf else (cls, *xywh)
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line) + "\n") % line)

                    if view_img:
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3
                        )

            # Print time
            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):
                    raise StopIteration

            # Save results
            if dataset.mode == "images":
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    fourcc = "mp4v"
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                    )
                vid_writer.write(im0)

    if save_txt:
        print(f"Results saved to {Path(output)}")
    print(f"Done. ({time.time() - t0:.3f}s)")


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Anti-UAV Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="weights/best.pt",
        help="model weights path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="source (file/folder/webcam/stream)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect",
        help="output directory",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IOU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        default="",
        help="cuda device (e.g., 0 or 0,1,2,3 or cpu)",
    )
    parser.add_argument(
        "--view-img",
        action="store_true",
        help="display results",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="save results to *.txt",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="save confidences in --save-txt labels",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class (e.g., --classes 0 or --classes 0 2 3)",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="class-agnostic NMS",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="augmented inference",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="update all models (strip optimizers)",
    )
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        if args.update:
            for weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect(weights=weights)
                strip_optimizer(weights)
        else:
            detect(
                weights=args.weights,
                source=args.source,
                output=args.output,
                img_size=args.img_size,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                device=args.device,
                view_img=args.view_img,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                classes=args.classes,
                agnostic_nms=args.agnostic_nms,
                augment=args.augment,
            )


if __name__ == "__main__":
    main()
