import yaml
import sys
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import albumentations as A
import numpy as np
from collections import defaultdict


# ---------------- Geometric/cropping helpers ----------------
def rotate_image_to_horizontal(img, mask, pt1, pt2):
    """
    Rotate an image around the center of a line so that the line becomes horizontal,
    and return the rotated positions of pt1 and pt2.

    Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask.
        pt1 (tuple): First point of the line (x, y).
        pt2 (tuple): Second point of the line (x, y).

    Returns:
        rotated_img (numpy.ndarray): Rotated image.
        rotated_mask (numpy.ndarray): Rotated mask.
        rotated_pts (tuple): Rotated positions of pt1 and pt2 as ((x1, y1), (x2, y2)).
    """
    # Compute the center of the line
    center_x = (pt1[0] + pt2[0]) / 2
    center_y = (pt1[1] + pt2[1]) / 2
    center = (center_x, center_y)

    # Compute the angle of the line relative to horizontal
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = np.degrees(np.arctan2(dy, dx))  # angle in degrees
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180

    # Rotation matrix to rotate around the center of the line
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to image and mask
    rotated_img = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (mask.shape[1], mask.shape[0]))

    # Apply rotation to pt1 and pt2
    pts = np.array([[pt1, pt2]], dtype=np.float32)  # shape (1, 2, 2)
    rotated_pts = cv2.transform(pts, rot_matrix)    # shape (1, 2, 2)
    rotated_pts = [tuple(map(int, p)) for p in rotated_pts[0]]

    return rotated_img, rotated_mask, rotated_pts

def mask_outside_contour(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Sets all pixels outside the given contour to 0.

    Parameters:
    - img: np.ndarray, input image
    - contour: np.ndarray, contour points (e.g., from cv2.findContours)

    Returns:
    - masked image (np.ndarray)
    """
    # Create a mask with the same size as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Fill the contour area on the mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the image
    if len(img.shape) == 3:  # Color image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
    else:  # Grayscale image
        masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def crop_to_mask(img, mask, midline_pts=None):
    """
    Crop both image and mask to the smallest rectangle containing all non-zero pixels in the mask.
    Optionally adjust midline points to the cropped coordinate system.

    Parameters
    ----------
    img : np.ndarray
        Image array (H×W or H×W×C).
    mask : np.ndarray
        Mask array (H×W), non-zero pixels indicate region of interest.
    midline_pts : list of tuples or None
        [(x1, y1), (x2, y2)] points in original image coordinates.

    Returns
    -------
    cropped_img : np.ndarray
        Cropped image.
    cropped_mask : np.ndarray
        Cropped mask.
    cropped_midline_pts : list of tuples or None
        Midline points adjusted to cropped coordinates.
    bbox : tuple
        (ymin, ymax, xmin, xmax) indices in original image.
    """
    # Ensure mask is binary or non-zero for ROI
    roi = mask > 0

    if not np.any(roi):
        return img, mask, midline_pts

    # Find bounding box of non-zero mask pixels
    rows = np.where(roi.sum(axis=1) > 0)[0]
    cols = np.where(roi.sum(axis=0) > 0)[0]

    ymin, ymax = rows[0], rows[-1]
    xmin, xmax = cols[0], cols[-1]

    cropped_img = img[ymin:ymax+1, xmin:xmax+1]
    cropped_mask = mask[ymin:ymax+1, xmin:xmax+1]

    # Adjust midline points if provided
    cropped_midline_pts = None
    if midline_pts is not None:
        cropped_midline_pts = [(x - xmin, y - ymin) for (x, y) in midline_pts]

    return cropped_img, cropped_mask, cropped_midline_pts

def pad_and_resize(img, mask, final_size=224, pad_color=(0, 0, 0), pad_mask_value=0):
    """
    Pad image and mask to square and resize to final_size x final_size.
    Preserves aspect ratio for both.

    Args:
        img: Original image (H x W x C)
        mask: Corresponding mask (H x W)
        final_size: Target size for ViT input
        pad_color: Color for image padding
        pad_mask_value: Value for mask padding

    Returns:
        img_resized, mask_resized
    """
    h, w = img.shape[:2]
    max_side = max(h, w)

    # Create square canvas for image and mask
    padded_img = np.full((max_side, max_side, 3), pad_color, dtype=img.dtype)
    padded_mask = np.full((max_side, max_side), pad_mask_value, dtype=mask.dtype)

    # Compute offsets for centering
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2

    # Place original image and mask in the center
    padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    padded_mask[y_offset:y_offset+h, x_offset:x_offset+w] = mask

    # Resize both to final_size x final_size
    img_resized = cv2.resize(padded_img, (final_size, final_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(padded_mask, (final_size, final_size), interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized

# ---------------- Pixelwise Augmentations ----------------
def perform_pixelwise_augmentations(img, difficulty = 'hard'):
    "Augment the pixels of an image according to the specified difficulty level."
    if difficulty == 'easiest': pixelwise_transform = A.Compose([])

    elif difficulty == 'easy':
        pixelwise_transform = A.Compose([
        #Ligthing artifacts
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomToneCurve(scale = 0.3, p=0.2),
        ])
        
    elif difficulty == 'medium':
        pixelwise_transform = A.Compose([
        ### Blur
        A.MotionBlur(blur_limit=10, p=0.1),

        #Ligthing artifacts
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        A.RandomToneCurve(scale = 0.3, p=0.2),
        A.HueSaturationValue(p=0.2),
        ])

    elif difficulty == 'hard':
        pixelwise_transform = A.Compose([
            ### Blur
            A.MotionBlur(blur_limit=15, p=0.2),
            A.MedianBlur(p=0.2),

            #Ligthing artifacts
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
            A.RandomToneCurve(scale = 0.3, p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Illumination(intensity_range=[0.01, 0.1], p=0.2),
        ])

    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img_uint8 = img.astype(np.uint8, copy=False)
    else:
        img_uint8 = img

    augmented = pixelwise_transform(image=img_uint8)
    return augmented['image']

# ---------------- Perspective Augmentations ----------------
def perform_perspective_augmentations(img, label, scale=0.05, max_attempts=3, patch_names = ['Q1', 'Q2']):
    """
    Perform perspective augmentation on image, mask and midline points.
    Retry up to max_attempts times to ensure masks are inside the bounds of the augmented image
    """
    h, w = img.shape[:2]

    # Prepare original contours for validation
    contours = {name: label[name + '_contour'] for name in patch_names}

    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    for _ in range(max_attempts):
        # Generate random destination points
        max_dx = scale * w
        max_dy = scale * h
        dst_pts = src_pts + np.column_stack([
            np.random.uniform(-max_dx, max_dx, 4),
            np.random.uniform(-max_dy, max_dy, 4)
        ]).astype(np.float32)

        # Compute perspective matrix
        M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Validate: transform contours and check bounds
        all_inside = True
        for name in patch_names:
            pts = np.array(contours[name], dtype=np.float32).reshape(-1, 1, 2)
            transformed_pts = cv2.perspectiveTransform(pts, M_perspective)
            if not np.all((0 <= transformed_pts[:, 0, 0]) & (transformed_pts[:, 0, 0] < w) &
                          (0 <= transformed_pts[:, 0, 1]) & (transformed_pts[:, 0, 1] < h)):
                all_inside = False
                break

        if all_inside:
            # Apply transform
            img_aug = cv2.warpPerspective(img, M_perspective, (w, h))
            patch_masks = {}
            for name in patch_names:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [contours[name]], 255)
                patch_masks[name] = cv2.warpPerspective(mask, M_perspective, (w, h))

            # Transform midline points
            xyxy_dir = label['direction'].reshape(4)
            pnts = {}
            for patch_name in patch_names:
                if patch_name == 'Q1':
                    midline_pnts = [label[patch_name + '_c3'][0], label[patch_name + '_c0'][0]] if xyxy_dir[0] < xyxy_dir[2] else [label[patch_name + '_c1'][0], label[patch_name + '_c2'][0]]
                elif patch_name == 'Q2':
                    midline_pnts = [label[patch_name + '_c2'][0], label[patch_name + '_c1'][0]] if xyxy_dir[0] < xyxy_dir[2] else [label[patch_name + '_c0'][0], label[patch_name + '_c3'][0]]
                else:
                    midline_pnts = [label[patch_name + '_c0'][0], label[patch_name + '_c1'][0]] if xyxy_dir[0] < xyxy_dir[2] else [label[patch_name + '_c2'][0], label[patch_name + '_c0'][0]]

                pts = np.array(midline_pnts, dtype=np.float32).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, M_perspective)
                pnts[patch_name] = [tuple(map(int, pt[0])) for pt in transformed_pts]

            return img_aug, patch_masks, pnts

    # If all attempts fail, return original image and masks
    patch_masks = {}
    for name in patch_names:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contours[name]], 255)
        patch_masks[name] = mask

    return img, patch_masks, {name: [[0, 0], [0, 0]] for name in patch_names}

def perspective_augment_minimal(img, scale=0.05):
    """
    Apply a random perspective transformation to an image.
    
    Parameters:
        img (np.ndarray): Input image (H x W x C).
        scale (float): Max fraction of width/height for point displacement.
    
    Returns:
        np.ndarray: Augmented image.
    """
    h, w = img.shape[:2]

    # Original corner points
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Randomly perturb corners
    max_dx = scale * w
    max_dy = scale * h
    dst_pts = src_pts + np.column_stack([
        np.random.uniform(-max_dx, max_dx, 4),
        np.random.uniform(-max_dy, max_dy, 4)
    ]).astype(np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply warp
    img_aug = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img_aug

# ----------------- Slice helpers -----------------
def crop_horizontal_patches(img, mask, midline_pts, fractions, xyxy_dir, token_px=16):
    """
    Split an image (and mask) into horizontal patches along a salmon midline,
    with fixed overlap between patches. Patch 0 = tail side, Patch N = head side.

    Direction:
      - xyxy_dir[0] >= xyxy_dir[2] -> Salmon swims Left→Right (head on left)
      - else                       -> Salmon swims Right→Left (head on right)

    Args:
        img:        (H, W, C) image array
        mask:       (H, W) or (H, W, 1) mask array
        midline_pts: [(x1,y1), (x2,y2)] midline endpoints (only x used)
        fractions:  sorted fractions in (0,1) for interior cuts
        xyxy_dir:   [x1, y1, x2, y2] direction hint
        token_px:   overlap in pixels (default 16)

    Returns:
        slices_img:  list of cropped image arrays
        slices_mask: list of cropped mask arrays
        x_ranges:    list of (x_min, x_max) for each patch
    """
    # Image dimensions
    h, w = img.shape[:2]

    # Midline endpoints as floats
    p1, p2 = np.array(midline_pts[0], float), np.array(midline_pts[1], float)

    # Validate fractions (keep only 0<f<1)
    fractions = sorted(f for f in fractions if 0 < f < 1)

    # Horizontal span based on midline x-difference
    L = abs(p2[0] - p1[0])

    # Clamp helper: ensures valid (xmin, xmax) inside [0, w]
    clamp = lambda a, b: (max(0, min(w, int(a))), max(0, min(w, int(b)))) if b > a else None

    # Determine swimming direction: True if Left→Right
    lr = xyxy_dir[0] >= xyxy_dir[2]

    # Compute interior cut positions along x from head anchor
    cuts = [(p1[0] + L*f) if lr else (p2[0] - L*f) for f in fractions]

    # Build full list of boundary edges
    edges = ([0] if lr else [w]) + [int(x) for x in cuts] + ([w] if lr else [0])

    # Ensure ascending order for pairing
    if not lr:
        edges.reverse()

    # Nominal (non-overlapping) segments from consecutive edge pairs
    nominal = [clamp(a, b) for a, b in zip(edges, edges[1:]) if clamp(a, b)]

    # Apply overlap between adjacent segments
    x_ranges = [nominal[0]]
    for i in range(1, len(nominal)):
        xmin, xmax = nominal[i]
        if lr:
            xmin = max(0, nominal[i-1][1] - token_px)  # overlap from previous right edge
        else:
            xmax = min(w, xmax + token_px)            # extend slightly right for overlap
        x_ranges.append(clamp(xmin, xmax))

    # Normalize order so patch 0 = tail, patch N = head
    if not lr:
        x_ranges.reverse()

    # Slice image and mask along computed ranges
    if x_ranges is None or len(x_ranges) == 0:
        return None, None, None

    slices_img = []
    slices_mask = []
    for rng in x_ranges:
        if rng is None or len(rng) != 2:
            return None, None, None
        x0, x1 = rng
        if x1 <= x0:  # invalid range
            return None, None, None
        slices_img.append(img[:, x0:x1])
        slices_mask.append(mask[:, x0:x1])

    # If any slice is empty, bail out
    if any(s.shape[1] == 0 for s in slices_img):
        return None, None, None

    return slices_img, slices_mask, x_ranges

# ---------------- Check helpers ----------------
def wrong_shape(img):
    # Gråskala (H, W) -> (H, W, 3)
    if img.ndim == 2:
        return True

    # Enkeltkanal (H, W, 1) -> (H, W, 3)
    if img.ndim == 3 and img.shape[2] == 1:
       return True
    return False

# ----------------- Draw helpers -----------------
def draw_contours_and_boxes(
    img,
    imgs_to_load,
    parent_names,
    patch_masks,
    requested_slices,
    slice_split_idces,
    pnts,
    patches,
    contour_config=None,
):
    contoured = img.copy()

    cfg = contour_config or {}
    default_parent = cfg.get("default_parent", {"color": (255, 0, 0), "thickness": 2})
    default_slice  = cfg.get("default_slice",  {"color": (0, 255, 0), "thickness": 2})
    parent_overrides = cfg.get("parents", {})
    slice_overrides  = cfg.get("slices", {})
    box_overrides    = cfg.get("boxes", {})
    default_box_style = {"color": (255, 255, 255), "thickness": 2}

    def style_for_parent(name):
        o = parent_overrides.get(name, {})
        return tuple(o.get("color", default_parent["color"])), int(o.get("thickness", default_parent["thickness"]))

    def style_for_slice(parent_name, idx):
        o = slice_overrides.get(f"{parent_name}_s{idx}", {})
        return tuple(o.get("color", default_slice["color"])), int(o.get("thickness", default_slice["thickness"]))

    def style_for_box(bp_name):
        o = box_overrides.get(bp_name, {})
        return tuple(o.get("color", default_box_style["color"])), int(o.get("thickness", default_box_style["thickness"]))

    def draw_mask_contours(dst_img, mask, color, thickness):
        if mask is None: return
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours: cv2.drawContours(dst_img, contours, -1, color, thickness)

    def slice_mask_open_ended(parent_mask, p0, p1, a_frac=None, b_frac=None):
        h, w = parent_mask.shape[:2]
        ys, xs = np.where(parent_mask > 0)
        if ys.size == 0: return np.zeros((h, w), np.uint8)
        p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
        vec = p1 - p0; L = float(np.linalg.norm(vec))
        if L < 1e-6: return np.zeros((h, w), np.uint8)
        u = vec / L
        s = (np.stack([xs - p0[0], ys - p0[1]], axis=1) @ u)
        cond_low = np.ones_like(s, bool) if a_frac is None else (s >= a_frac * L)
        cond_high = np.ones_like(s, bool) if b_frac is None else (s < b_frac * L)
        sel = cond_low & cond_high
        smask = np.zeros((h, w), np.uint8)
        smask[ys[sel], xs[sel]] = 255
        return smask

    for parent in parent_names:
        if parent in patch_masks:
            if parent in imgs_to_load:
                color, thickness = style_for_parent(parent)
                draw_mask_contours(contoured, patch_masks[parent], color, thickness)

            if parent in requested_slices and requested_slices[parent]:
                splits = [0.0] + list(slice_split_idces) + [1.0]
                p0, p1 = map(tuple, pnts[parent])
                last_idx = len(splits) - 2
                for i in requested_slices[parent]:
                    # Make both ends open-ended:
                    a_frac = None if i == 0 else splits[i]      # First slice ignores lower bound
                    b_frac = None if i == last_idx else splits[i + 1]  # Last slice ignores upper bound
                    smask = slice_mask_open_ended(patch_masks[parent], p0, p1, a_frac, b_frac)
                    color, thickness = style_for_slice(parent, i)
                    draw_mask_contours(contoured, smask, color, thickness)

    for bp in ['head', 'dorsal_fin', 'tail_fin', 'adi_fin']:
        if bp in imgs_to_load and bp in patches and 'bbox' in patches[bp]:
            x1, y1, x2, y2 = map(int, patches[bp]['bbox'])
            color, thickness = style_for_box(bp)
            cv2.rectangle(contoured, (x1, y1), (x2, y2), color, thickness)

    return contoured

# ------------------ Body part loading helpers -----------------
BODY_PARTS = ('head', 'dorsal_fin', 'tail_fin', 'adi_fin')

def _read_analysis_labels(analysis_path: Path) -> np.ndarray:
    """Read the *_results_refined.txt into a numpy array of strings."""
    rows = []
    with open(analysis_path, 'r') as f:
        for line in f:
            rows.append(line.rstrip().split(','))
    return np.array(rows)

def get_lr_dir_from_analysis(full_frame_label: np.ndarray) -> str:
    """
    Determine left/right direction based on head and tail positions in analysis labels.
    
    Parameters
    ----------
    full_frame_label : np.ndarray
        Rows for the current frame/camera from *_results_refined.txt.
        Assumes columns: [frame, cid, x, y, w, h, ..., class] with class at index 10.
    
    Returns
    -------
    str
        'r' if head.x > tail.x else 'l'. Defaults to 'l' if missing data.
    """
    try:
        head = full_frame_label[full_frame_label[:, 10] == 'head'][0][2:4].astype(float)
        tail = full_frame_label[full_frame_label[:, 10] == 'tail_fin'][0][2:4].astype(float)
        return 'r' if head[0] > tail[0] else 'l'
    except Exception:
        return 'l'  # fallback if head/tail not found

def _load_body_part_patches(
    img: np.ndarray,
    file_name: Path,
    imgs_to_load,
    bp_margin: dict,
    difficulty: str,
    scale: float,
    include_bbox: bool = False,
    analysis_arr: np.ndarray = None,
    analysis_path: Path = None,
    lr_dir=None,
    perspective_augment_fn=None,  # e.g. perspective_augment_minimal
    pixel_augment_fn=None,         # e.g. perform_pixelwise_augmentations
):
    """
    Common body-part loader used by both load_patches and load_tracker_patches.

    Parameters
    ----------
    img : np.ndarray
        RGB image (H,W,3)
    file_name : Path
        Path to the label file; used to derive frame/camera ids
    imgs_to_load : list[str]
        Names requested; we only process intersection with BODY_PARTS
    bp_margin : dict
        Extra padding per body part (pixels), e.g., {'head':20, ...}
    difficulty : str
        Aug difficulty for pixel-level augmentations
    scale : float
        Perspective augmentation scale for the minimal crop augment
    include_bbox : bool
        If True, include 'bbox' (xyxy in the *cropped fish* frame space) in result
    analysis_arr : np.ndarray or None
        If provided, reuse parsed analysis labels. Otherwise read analysis_path.
    analysis_path : Path or None
        If analysis_arr is None, this is used to read labels. If None, it's inferred.
    lr_dir_hint : str
        Fallback direction if head/tail data is missing
    perspective_augment_fn : callable
        Function(crop_img, scale) -> augmented crop
    pixel_augment_fn : callable
        Function(image, difficulty) -> augmented image

    Returns
    -------
    patches : dict
        For each requested body part found: {'valid': True, 'img': <np.ndarray>, 'dir': 'l'|'r', ['bbox': np.ndarray]}
    lr_dir : str
        Derived or hinted direction
    """
    if analysis_arr is None:
        if analysis_path is None:
            # Default path: <folder>/<folder>_results_refined.txt (as in your current code)
            analysis_path = file_name.parent.parent / Path(str(file_name.parent.parent.name) + '_results_refined.txt')
        analysis_arr = _read_analysis_labels(analysis_path)

    # Extract frame and camera id the same way you do today
    frame_num = int(file_name.name.split('_')[4].split('.')[0])
    c_id = int(file_name.name.split('_')[2])

    mask = np.bitwise_and(analysis_arr[:, 0] == str(frame_num), analysis_arr[:, 1] == str(c_id))
    full_frame_label = analysis_arr[mask, :]

    patches = {}

    # Find salmon row (for fish crop/offset)
    try:
        salmon_full_frame = full_frame_label[full_frame_label[:, 10] == 'salmon', :][0]
    except IndexError:
        # No salmon -> cannot place parts; return empty (or mark invalid)
        return patches

    salmon_xywh = salmon_full_frame[2:6].astype(float)
    salmon_xyxy = np.array([
        salmon_xywh[0] - 0.5 * salmon_xywh[2],
        salmon_xywh[1] - 0.5 * salmon_xywh[3],
        salmon_xywh[0] + 0.5 * salmon_xywh[2],
        salmon_xywh[1] + 0.5 * salmon_xywh[3],
    ]).astype(int)

    salmon_offset = np.array([
        max(salmon_xyxy[0] - 20, 0),
        max(salmon_xyxy[1] - 20, 0),
    ])

    # Process requested parts
    for bp in BODY_PARTS:
        if bp not in imgs_to_load:
            continue

        part_label = full_frame_label[full_frame_label[:, 10] == bp]
        if len(part_label) == 0:
            continue  # silently skip if missing

        xywh_full_frame = part_label[0][2:6].astype(float)

        # Shift center by the salmon offset, expand box by margin in width/height
        # (note negative margin in vector to expand)
        xywh_crop = xywh_full_frame - np.concatenate([salmon_offset, [-bp_margin[bp], -bp_margin[bp]]])

        # Convert xywh -> xyxy
        xyxy_crop = np.array([
            xywh_crop[0] - 0.5 * xywh_crop[2],
            xywh_crop[1] - 0.5 * xywh_crop[3],
            xywh_crop[0] + 0.5 * xywh_crop[2],
            xywh_crop[1] + 0.5 * xywh_crop[3],
        ]).astype(int)

        # Clamp to image bounds
        xyxy_crop[0] = max(xyxy_crop[0], 0)
        xyxy_crop[1] = max(xyxy_crop[1], 0)
        xyxy_crop[2] = min(xyxy_crop[2], img.shape[1])
        xyxy_crop[3] = min(xyxy_crop[3], img.shape[0])

        # Crop and augment
        crop_img = img[xyxy_crop[1]:xyxy_crop[3], xyxy_crop[0]:xyxy_crop[2], :]

        if perspective_augment_fn is not None:
            crop_img = perspective_augment_fn(crop_img, scale=scale)
        if pixel_augment_fn is not None:
            crop_img = pixel_augment_fn(crop_img, difficulty=difficulty)

        if lr_dir is None:
            lr_dir = get_lr_dir_from_analysis(full_frame_label)

        out = {'valid': True, 'img': crop_img, 'dir': lr_dir}
        if include_bbox:
            out['bbox'] = xyxy_crop

        patches[bp] = out

    return patches

#---------------- Main loading functions ----------------
def load_patches(
    file_name,
    valid_threshs={'Q1': 0.25, 'Q2': 0.25},
    difficulty='hard',
    imgs_to_load=['Q1', 'Q1_s0', 'head', 'dorsal_fin'],
    crop_slices_to_mask=True,
    slice_split_idces=[0.3, 0.7],
    contour_config=None,
    bp_margin={'head': 20, 'dorsal_fin': 20, 'tail_fin': 20, 'adi_fin': 10}
):
    """
    Extended load_patches:
    - Q1/Q2 patches and optional slices.
    - 'img_slice_viz' for Q1/Q2: black lines at slice boundaries.
    - Body parts: head, dorsal_fin, tail_fin, adi_fin (cropped + augmented).
    - 'complete_img_contours': contours for parents/slices and bounding boxes for body parts.
    - Styling via contour_config:
        {
          "default_parent": {"color": (255,0,0), "thickness": 2},
          "default_slice":  {"color": (0,255,0), "thickness": 2},
          "parents": {"Q1": {...}, "Q2": {...}},
          "slices":  {"Q1_s0": {...}, ...},
          "boxes":   {"head": {...}, "dorsal_fin": {...}, ...}
        }
    """
    # --- Load image ---
    img_path = file_name.parent.parent / 'images' / file_name.name.replace('label', 'image').replace('.txt', '.png')
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure RGB shape
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # --- Difficulty scale ---
    scale_map = {'easiest': 0, 'easy': 0.05, 'medium': 0.15, 'hard': 0.15}
    scale = scale_map.get(difficulty, 0.15)

    # --- Parse label file for Q1/Q2 ---
    label = {}
    with open(str(file_name), 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                label[key] = np.array(val.strip().split(',')).reshape(-1, 2).astype(int)

    xyxy_dir = label['direction'].reshape(4)
    lr_dir = 'r' if xyxy_dir[0] > xyxy_dir[2] else 'l'

    # --- Perspective augmentation (original augmented image + masks + midlines) ---
    img, patch_masks, pnts = perform_perspective_augmentations(img, label, scale=scale)

    patches = {}
    parent_names = {n.split('_')[0] for n in imgs_to_load}

    # --- Determine requested slices (e.g., {'Q1':[0,2], 'Q2':[1]}) ---
    requested_slices = {}
    for name in imgs_to_load:
        if '_s' in name:
            parent, s = name.split('_s', 1)
            try:
                idx = int(s)
            except ValueError:
                continue
            requested_slices.setdefault(parent, []).append(idx)

    # --- Q1/Q2 logic ---
    for parent in parent_names:
        if parent not in patch_masks:
            patches[parent] = {'valid': False, 'img': None, 'dir': lr_dir}
            continue

        cnt_img = img.copy()
        mask = patch_masks[parent]
        midline = pnts[parent]

        # Rotate and crop to mask
        cnt_img, mask, midline = rotate_image_to_horizontal(cnt_img, mask, midline[0], midline[1])
        cnt_img, mask, midline = crop_to_mask(cnt_img, mask, midline)

        # Optional slices on the rotated/cropped parent
        slices_img, slices_mask = [], []
        if parent in ['Q1', 'Q2']:
            slices_img, slices_mask, _ = crop_horizontal_patches(
                cnt_img, mask, midline, slice_split_idces, xyxy_dir,
                token_px=int(round(16 * (cnt_img.shape[1] / 224)))
            )
        if slices_img is None or slices_mask is None:
            return {k: {'valid': False, 'img': None, 'dir': lr_dir} for k in imgs_to_load}
        expected = len(slice_split_idces) + 1
        if parent in ['Q1', 'Q2'] and len(slices_img) != expected:
            return {k: {'valid': False, 'img': None, 'dir': lr_dir} for k in imgs_to_load}

        # Augment parent patch
        try:
            #cnt_img_shape = cnt_img.shape
            #cnt_img_resized = cv2.resize(cnt_img, (224, 224), interpolation=cv2.INTER_CUBIC)
            #cnt_img_resized = perform_pixelwise_augmentations(cnt_img_resized, difficulty=difficulty)
            #cnt_img = cv2.resize(cnt_img_resized, (cnt_img_shape[1], cnt_img_shape[0]), interpolation=cv2.INTER_CUBIC)
            cnt_img = perform_pixelwise_augmentations(cnt_img, difficulty=difficulty)
        except Exception:
            print('Could not pixel-augment: ' + str(file_name))
            return {k: {'valid': False, 'img': None, 'dir': None} for k in imgs_to_load}

        cnt_img = cv2.bitwise_and(cnt_img, cnt_img, mask=mask)

        # Validate parent
        coverage = np.sum(mask) / (255.0 * mask.size)
        valid = coverage > valid_threshs.get(parent, 0) and not wrong_shape(cnt_img)

        # Store parent 
        patches[parent] = {'valid': valid, 'img': cnt_img, 'dir': lr_dir}

        # Store slices if requested
        for i, (s_img, s_mask) in enumerate(zip(slices_img, slices_mask)):
            slice_name = f"{parent}_s{i}"
            if slice_name in imgs_to_load:
                s_img, s_mask = pad_and_resize(s_img, s_mask)
                s_img = perform_pixelwise_augmentations(s_img, difficulty=difficulty)
                s_img = cv2.bitwise_and(s_img, s_img, mask=s_mask)
                if crop_slices_to_mask:
                    s_img, s_mask, _ = crop_to_mask(s_img, s_mask)
                #s_img_shape = s_img.shape
                #s_img_resized = cv2.resize(s_img, (224, 224), interpolation=cv2.INTER_CUBIC)
                #s_img_resized = perform_pixelwise_augmentations(s_img_resized, difficulty=difficulty)
                #s_img = cv2.resize(s_img_resized, (s_img_shape[1], s_img_shape[0]), interpolation=cv2.INTER_CUBIC)
                #s_img = cv2.bitwise_and(s_img, s_img, mask=s_mask)
                patches[slice_name] = {'valid': not wrong_shape(s_img), 'img': s_img, 'dir': lr_dir}

    # --- Body parts via shared helper ---
    if any(bp in imgs_to_load for bp in BODY_PARTS):
        analysis_path = file_name.parent.parent / Path(str(file_name.parent.parent.name) + '_results_refined.txt')
        bp_patches = _load_body_part_patches(
            img=img,
            file_name=file_name,
            imgs_to_load=imgs_to_load,
            bp_margin=bp_margin,
            difficulty=difficulty,
            scale=scale,
            include_bbox=True,  # keep bbox for contour drawing
            analysis_arr=None,  # read from analysis_path
            analysis_path=analysis_path,
            lr_dir=lr_dir,
            perspective_augment_fn=perspective_augment_minimal,
            pixel_augment_fn=perform_pixelwise_augmentations,
        )
        patches.update(bp_patches)

    # --- Optionally return original augmented image ---
    if 'complete_img' in imgs_to_load:
        patches['complete_img'] = {'valid': True, 'img': img}

    # --- Draw contours (parents/slices) + bounding boxes (body parts) ---
    if 'complete_img_contours' in imgs_to_load:
        contoured = draw_contours_and_boxes(
            img,
            imgs_to_load,
            parent_names,
            patch_masks,
            requested_slices,
            slice_split_idces,
            pnts,
            patches,
            contour_config=contour_config,
        )

        patches['complete_img_contours'] = {'valid': True, 'img': contoured}

    return patches


def load_tracker_patches(
    file_name,
    difficulty='hard',
    imgs_to_load=['head', 'dorsal_fin', 'tail_fin', 'adi_fin'],
    bp_margin = {'head': 20, 'dorsal_fin': 20, 'tail_fin': 20, 'adi_fin': 10},
):
     # --- Load image ---
    img_path = file_name.parent.parent / 'images' / file_name.name.replace('label', 'image').replace('.txt', '.png')
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure RGB shape
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

     # --- Difficulty scale ---
    scale_map = {'easiest': 0, 'easy': 0.05, 'medium': 0.15, 'hard': 0.15}
    scale = scale_map.get(difficulty, 0.15)
    
    analysis_path = file_name.parent.parent / Path(str(file_name.parent.parent.name) + '_results_refined.txt')

    patches = _load_body_part_patches(
        img=img,
        file_name=file_name,
        imgs_to_load=imgs_to_load,
        bp_margin=bp_margin,
        difficulty=difficulty,
        scale=scale,
        include_bbox=False,                # tracker does not need bbox
        analysis_arr=None,
        analysis_path=analysis_path,
        lr_dir=None,
        perspective_augment_fn=perspective_augment_minimal,
        pixel_augment_fn=perform_pixelwise_augmentations,
    )

    return patches
