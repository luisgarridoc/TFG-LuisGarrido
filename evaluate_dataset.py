import os
import glob
import math
import argparse
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """Parse CLI arguments for input/output folders and options."""
    ap = argparse.ArgumentParser(description="Histograms for YOLOv11-OBB")
    ap.add_argument("--labels_dir", required=True, help="Folder with OBB .txt label files")
    ap.add_argument("--images_dir", required=True, help="Folder with images")
    ap.add_argument("--out_dir", required=True, help="Output folder for figures")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"],
                    help="Image extensions to try, in order")
    ap.add_argument("--class_map", default="",
                    help='Optional map "0:car,1:bus" to rename class IDs')
    return ap.parse_args()

def load_class_map(s):
    """Convert '0:car,1:bus' into {0: 'car', 1: 'bus'}."""
    if not s:
        return {}
    mapping = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        mapping[int(k.strip())] = v.strip()
    return mapping

def find_image_for_label(images_dir, stem, exts):
    """Find the first image file matching the label stem and allowed extensions."""
    for ext in exts:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def obb_edges_lengths_px(points_norm, w, h):
    """
    Compute the 4 edge lengths (in pixels) of a 4-point OBB polygon given
    normalized coordinates and image size.
    """
    pts_px = points_norm.copy()
    pts_px[:, 0] = pts_px[:, 0] * w
    pts_px[:, 1] = pts_px[:, 1] * h

    rolled = np.vstack([pts_px, pts_px[0:1, :]])
    edges = []
    for i in range(4):
        dx = rolled[i+1, 0] - rolled[i, 0]
        dy = rolled[i+1, 1] - rolled[i, 1]
        edges.append(math.hypot(dx, dy))
    return edges

def obb_width_height_px(points_norm, w, h):
    """
    Estimate OBB width/height (in pixels) as the average of opposite edges,
    assigning width = longer side, height = shorter side.
    """
    e0, e1, e2, e3 = obb_edges_lengths_px(points_norm, w, h)
    a = 0.5 * (e0 + e2)
    b = 0.5 * (e1 + e3)
    width = max(a, b)
    height = min(a, b)
    return width, height

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    class_map = load_class_map(args.class_map)

    # Accumulators
    class_counts = Counter()
    widths_px = []
    heights_px = []

    missing_images = []

    # Discover label files
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    if not label_files:
        print(f"[Notice] No .txt files found in {args.labels_dir}")
        return

    # Iterate labels and associated images
    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = find_image_for_label(args.images_dir, stem, args.exts)

        # Track missing or unreadable images
        if img_path is None:
            missing_images.append(stem)
            continue

        img = cv2.imread(img_path)
        if img is None:
            missing_images.append(stem)
            continue
        h, w = img.shape[:2]

        # Read all OBB lines for this image
        with open(lf, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for ln in lines:
            parts = ln.split()
            # Expect: class_id + 8 coords = 9 tokens
            if len(parts) != 9:
                continue
            try:
                cls_id = int(float(parts[0]))
                coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue

            # Normalize if file accidentally contains pixel coords, and clip
            if np.any(coords < 0) or np.any(coords > 1.0001):
                if np.any(coords[:, 0] > 1.0) or np.any(coords[:, 1] > 1.0):
                    coords[:, 0] = coords[:, 0] / float(w)
                    coords[:, 1] = coords[:, 1] / float(h)
                coords = np.clip(coords, 0.0, 1.0)

            # Accumulate width/height and class counts
            width_px, height_px = obb_width_height_px(coords, w, h)
            widths_px.append(width_px)
            heights_px.append(height_px)
            class_counts[cls_id] += 1

    # Plot class distribution
    if class_counts:
        ids = sorted(class_counts.keys())
        names = [class_map.get(i, str(i)) for i in ids]
        counts = [class_counts[i] for i in ids]

        plt.figure(figsize=(8, 5))
        plt.bar(names, counts)
        plt.title("Class distribution (YOLOv11-OBB)")
        plt.xlabel("Class")
        plt.ylabel("Instance count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "class_distribution.png"), dpi=200)
        plt.show()
    else:
        print("[Notice] No instances recorded (empty files?).")

    # Plot width histogram
    if widths_px:
        plt.figure(figsize=(8, 5))
        plt.hist(widths_px, bins=50)
        plt.title("OBB width distribution (px)")
        plt.xlabel("Width (px)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_width_px.png"), dpi=200)
        plt.show()

    # Plot height histogram
    if heights_px:
        plt.figure(figsize=(8, 5))
        plt.hist(heights_px, bins=50)
        plt.title("OBB height distribution (px)")
        plt.xlabel("Height (px)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_height_px.png"), dpi=200)
        plt.show()

    # Report missing images (if any)
    if missing_images:
        print(f"[Notice] {len(missing_images)} images not found for their labels. Examples: {missing_images[:5]}")

    # Final summary
    total_boxes = sum(class_counts.values())
    print(f"Total instances: {total_boxes}")
    for k in sorted(class_counts.keys()):
        name = class_map.get(k, str(k))
        print(f"  Class {name}: {class_counts[k]}")

if __name__ == "__main__":
    main()
