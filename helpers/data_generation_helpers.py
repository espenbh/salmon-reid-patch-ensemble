import sys
import cv2
# See https://github.com/espenbh/BoostCompTrack.
tracking_base = 'C:\\Users\\espen\\Documents\\work\\PhD\\papers\\2_salmon_tracking\\code\\salmon_component_tracking\\'
sys.path.append(tracking_base)
sys.path.append(tracking_base + 'helpers')
sys.path.append(tracking_base + 'associator')
sys.path.append(tracking_base + 'associator\\CompTrack')
sys.path.append(tracking_base + 'associator\\BoostTrack')
sys.path.append(tracking_base + 'associator\\BoostTrack\\external')

import keybox_utils as ku
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import imutils


def subsample_trackers(trackers, valid_ids, subsample_rate=5, start_offset=2):
    subsampled_trackers = []
    for idx in valid_ids:
        id_trackers = trackers[trackers[:,1]==idx]
        frame_nums = np.unique(id_trackers[:,0].astype(int))
        frame_nums_to_extract = frame_nums[start_offset::subsample_rate]
        id_subsampled_trackers = id_trackers[np.isin(id_trackers[:,0].astype(int), frame_nums_to_extract)]
        subsampled_trackers.append(id_subsampled_trackers)
    subsampled_trackers = np.vstack(subsampled_trackers)
    return subsampled_trackers

def comp_bbox_from_id_trackers(id_trackers, frame_num, comp = 'salmon', margin=40):
    id_frame_trackers = id_trackers[id_trackers[:,0] == str(frame_num)]
    comp_tracker = id_frame_trackers[id_frame_trackers[:,10] == comp][0]
    comp_bbox = ku.xywh2xyxy(comp_tracker[2:6].astype(float) + np.array([0, 0, margin, margin]))
    return comp_bbox

def comp_bboxes_from_id_trackers(id_trackers, frame_num, margins=[40, 80, 80, 0, 0, 0, 0], comps = ['salmon', 'head', 'body', 'pelv_fin', 'pec_fin', 'dorsal_fin', 'tail_fin']):
    return {comp: comp_bbox_from_id_trackers(id_trackers, frame_num, comp = comp, margin = margins[i]) for i, comp in enumerate(comps)}

def get_comp_bboxes_in_salmon(comp_bboxes):
    salmon_bbox = comp_bboxes['salmon']
    return {comp: comp_bboxes[comp] - np.array([salmon_bbox[0], salmon_bbox[1], salmon_bbox[0], salmon_bbox[1]]) for comp in comp_bboxes if comp != 'salmon'}

def get_four_corners_from_line(mask: np.ndarray,
                               dir_line: np.ndarray,
                               angle_deg: float = 45.0) -> np.ndarray:
    """
    Return four corners as support points of the convex hull in directions
    close to the direction of the given line, at ±angle_deg and their opposites.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (bool or 0/255 uint8). Nonzero => foreground.
    dir_line : np.ndarray
        Two points that define a direction line in image coordinates (x right, y down).
        Accepts either shape (4,) as [x0, y0, x1, y1], or shape (2, 2) as [[x0, y0], [x1, y1]].
    angle_deg : float
        Angle offset (in degrees) relative to the line direction. Smaller than 45
        brings the scanning directions closer to parallel to the object's direction.

    Returns
    -------
    corners_xy : np.ndarray, shape (4, 2), dtype float32
        Corners in (x, y): [front_right, front_left, back_left, back_right].
        "Front" is along the p0->p1 direction of dir_line; swapping the endpoints flips
        front/back but preserves the cyclic order.
    """
    # ---- prep mask & points ----
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        raise ValueError("mask has no foreground")
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # ---- convex hull ----
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)

    # ---- parse dir_line -> base unit direction (x,y) ----
    dl = np.asarray(dir_line, dtype=np.float32)
    if dl.ndim == 1 and dl.size == 4:
        x0, y0, x1, y1 = dl.tolist()
    elif dl.ndim == 2 and dl.shape == (2, 2):
        (x0, y0), (x1, y1) = dl.tolist()
    else:
        raise ValueError("dir_line must be [x0, y0, x1, y1] or [[x0, y0], [x1, y1]]")

    base = np.array([x1 - x0, y1 - y0], np.float32)
    norm = float(np.linalg.norm(base))
    if norm < 1e-6:
        raise ValueError("dir_line endpoints are too close (near-zero length)")
    base /= norm  # unit (x, y) direction

    # ---- rotation matrices for ±angle_deg ----
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    R_plus  = np.array([[c, -s], [s,  c]], np.float32)  # +angle
    R_minus = np.array([[c,  s], [-s, c]], np.float32)  # -angle

    # four unit directions (x,y)
    u_fr = (R_plus  @ base)   # front-right  (+angle)
    u_fl = (R_minus @ base)   # front-left   (-angle)
    u_bl = -u_fr              # back-left
    u_br = -u_fl              # back-right

    def support(direction_xy: np.ndarray) -> np.ndarray:
        # Return hull vertex with maximal projection onto direction
        s = hull @ direction_xy
        i = int(np.argmax(s))
        return hull[i]

    # First pass
    fr = support(u_fr)
    fl = support(u_fl)
    bl = support(u_bl)
    br = support(u_br)

    corners = np.stack([fr, fl, bl, br], axis=0).astype(np.float32)

    # Optional de-duplication in near-degenerate angles
    def nearly_same(p, q, eps=1.0):
        return np.linalg.norm(p - q) < eps

    # Check pairs that are likely to collide when angle is tiny
    pairs = [(0, 1, +1), (2, 3, -1)]  # (fr,fl) and (bl,br)
    for i, j, sign in pairs:
        if nearly_same(corners[i], corners[j]):
            # jitter angle by ±1.5° for this pair
            da = np.deg2rad(1.5) * sign
            cj, sj = np.cos(a + da), np.sin(a + da)
            if i == 0:    # fr/fl pair
                Rj = np.array([[cj, -sj], [sj, cj]], np.float32)
                u_fr_j = (Rj @ base)
                u_fl_j = (np.array([[cj, sj], [-sj, cj]], np.float32) @ base)
                corners[0] = support(u_fr_j)
                corners[1] = support(u_fl_j)
            else:         # bl/br pair
                Rj = np.array([[cj, -sj], [sj, cj]], np.float32)
                u_bl_j = -(Rj @ base)
                u_br_j = -(np.array([[cj, sj], [-sj, cj]], np.float32) @ base)
                corners[2] = support(u_bl_j)
                corners[3] = support(u_br_j)

    return corners


def save_salmon_images(trackers, config, salmon_margin, analysis_path):
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)

    start_frame = config['start_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(imutils.rotate_bound(frame, config['rotate']), cv2.COLOR_BGR2RGB)
        frame_salmon_trackers = trackers[np.bitwise_and(trackers[:,0].astype(int) == frame_num, trackers[:,10] == 'salmon')]
        if len(frame_salmon_trackers) > 0:
            for trk in frame_salmon_trackers:
                xyxy = np.array(ku.xywh2xyxy(trk[2:6].astype(float) + np.array([0,0,salmon_margin,salmon_margin]))).astype(int)
                id = int(trk[1])
                salmon_img = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
                if salmon_img.shape[0] == 0 or salmon_img.shape[1] == 0:
                    continue
                out_img_path = (analysis_path / 'images' /f"a{int(analysis_path.name.split('ysis')[1]):01d}_salmon_{id:03d}_frame_{frame_num:05d}.png")
                cv2.imwrite(str(out_img_path), cv2.cvtColor(salmon_img, cv2.COLOR_RGB2BGR))
        frame_num += 1
        if frame_num > config['end_frame']:
            break
    cap.release()


def create_and_save_labels(trackers, results_path, out_root, segmentation_model_path, config):
    seg_model = YOLO(segmentation_model_path)
    image_path = Path(out_root) / Path(results_path).name / 'images'
    for file in image_path.iterdir():
        salmon_id = int(file.name.split('_')[2])
        frame_num = int(file.name.split('_')[4].split('.')[0])
        comp_bboxes = comp_bboxes_from_id_trackers(trackers[trackers[:,1]==str(salmon_id)], frame_num)
        comp_bboxes_in_salmon = get_comp_bboxes_in_salmon(comp_bboxes)
        dir_line = np.concatenate([ku.xyxy2xywh(comp_bboxes_in_salmon['head'])[:2], ku.xyxy2xywh(comp_bboxes_in_salmon['tail_fin'])[:2]])
        salmon_img = cv2.imread(str(file))

        res = seg_model(salmon_img)
        to_label = {}
        to_label['direction'] = ','.join(np.array(dir_line).astype(int).astype(str))
        if res[0].masks is not None:
            for mask, cls_idx, box in zip(res[0].masks.data, res[0].boxes.cls, res[0].boxes.xyxy):
                in_body = (box[0] > comp_bboxes_in_salmon['body'][0] and box[1] > comp_bboxes_in_salmon['body'][1] and box[2] < comp_bboxes_in_salmon['body'][2] and box[3] < comp_bboxes_in_salmon['body'][3])
                cls_name = res[0].names[int(cls_idx)]

                if cls_name == 'Q1' and in_body:
                    Q1_mask = cv2.resize(mask.cpu().numpy().astype(np.uint8) * 255, (salmon_img.shape[1], salmon_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    contours, hierarchy = cv2.findContours(Q1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    to_label['Q1_contour'] = ','.join([str(c) for c in contours[0][:,0,:].flatten()])
                    Q1_corners = get_four_corners_from_line((Q1_mask * 255).astype(np.uint8), dir_line, angle_deg=70.0)
                    for corner, i in zip(Q1_corners, range(len(Q1_corners))):
                        to_label['Q1_c' + str(i)] = f'{int(corner[0])},{int(corner[1])}'

                if cls_name == 'Q2' and in_body:
                    Q2_mask = cv2.resize(mask.cpu().numpy().astype(np.uint8) * 255, (salmon_img.shape[1], salmon_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    contours, hierarchy = cv2.findContours(Q2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    to_label['Q2_contour'] = ','.join([str(c) for c in contours[0][:,0,:].flatten()])
                    Q2_corners = get_four_corners_from_line((Q2_mask * 255).astype(np.uint8), dir_line, angle_deg=70.0)
                    for corner, i in zip(Q2_corners, range(len(Q2_corners))):
                        to_label['Q2_c' + str(i)] = f'{int(corner[0])},{int(corner[1])}'
        if len(list(to_label.keys()))==11:
            with open(Path(out_root) / Path(results_path).name / 'labels' / file.name.replace('image', 'label').replace('.png', '.txt'), 'w') as f:
                    for k,v in to_label.items():
                        f.write(f'{k}:{v}\n')             
