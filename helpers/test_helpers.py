import torch
import numpy as np
import pandas as pd
import os
import csv
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from training_helpers import compute_map
from pathlib import Path





# ---------- Verification helpers ----------
def ensure_2d_embedding(x: torch.Tensor, pool: str = "cls") -> torch.Tensor:
    """Return [B, D] from either [B, D] or [B, T, D]."""
    if not isinstance(x, torch.Tensor):
        raise ValueError("ensure_2d_embedding: input is not a torch.Tensor")
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        return x[:, 0, :] if pool == "cls" else x.mean(dim=1)
    else:
        raise ValueError(f"Expected 2D/3D tensor, got {x.dim()}D {tuple(x.shape)}")
    
def _canonical_label(x):
    try:
        s = str(x).strip()
        return s[:-2] if s.endswith(".0") else s
    except Exception:
        return str(x)

def _normalize_array_as_str(arr):
    return np.array([_canonical_label(a) for a in np.asarray(arr)], dtype=object)

# ---------- Fusion helpers ----------
def _temperature_similarity(distances: np.ndarray, tau: float) -> np.ndarray:
    sims = np.exp(-distances / max(tau, 1e-6))
    rng = sims.max() - sims.min()
    if rng > 0:
        sims = (sims - sims.min()) / rng
    return sims

def _weighted_rrf(ranks_by_part, weights, k=20):
    n = len(next(iter(ranks_by_part.values())))
    score = np.zeros(n, dtype=float)
    for p, ranks in ranks_by_part.items():
        score += weights.get(p, 1.0) * (1.0 / (k + ranks))
    return score

def _build_fused_cross_similarity(
    S_by_part, strengths, tau_by_part, lambda_mix=0.75, rrf_k=20,
):
    """Merge part similarities into a single ensamble similarity matrix."""
    for fn_name in ['_temperature_similarity', '_weighted_rrf']:
        assert fn_name in globals(), f"Helper '{fn_name}' must be available."
    parts = list(S_by_part.keys())
    S0 = next(iter(S_by_part.values()))
    Nq, Ng = S0.shape
    S_fused = np.zeros((Nq, Ng), dtype=float)
    for i in range(Nq):
        sim_by_part_i, ranks_by_part_i = {}, {}
        for p in parts:
            S_row = S_by_part[p][i].copy()
            d_p = 1.0 - np.clip(S_row, -1.0, 1.0)
            s_p = _temperature_similarity(d_p, tau_by_part.get(p, 0.7))
            sim_by_part_i[p] = s_p
            order = np.argsort(-S_row)
            ranks = np.empty(Ng, dtype=int); ranks[order] = np.arange(1, Ng + 1)
            ranks_by_part_i[p] = ranks
        rank_score = _weighted_rrf(ranks_by_part=ranks_by_part_i, weights=strengths, k=rrf_k)
        sim_score = sum(strengths.get(p, 1.0) * sim_by_part_i[p] for p in parts)
        fused = lambda_mix * rank_score + (1.0 - lambda_mix) * sim_score
        S_fused[i] = fused
    return S_fused

# ---------- IO helpers ----------
def load_id_match_map_from_traj_excel(
    xlsx_path,
    query_files,
    query_global_ids,
    gallery_files,
    gallery_global_ids,
    query_col="Query",
    gallery_col="Gallery"
):
    """
    Load a trajectory-ID Excel file and return a mapping with IDs that match the dataset (relative IDs):
        {query_id (str) : gallery_id (str)}
    """

    # Load Excel
    df = pd.read_excel(xlsx_path)

    # Validate columns
    if not {query_col, gallery_col}.issubset(df.columns):
        raise ValueError(
            f"Expected columns ['{query_col}','{gallery_col}'], got {df.columns}"
        )

    # Build traj → relative lookup tables
    query_traj_to_rel = {
        int(f.name.split('_')[2]): rel
        for f, rel in zip(query_files, query_global_ids)
    }
    gallery_traj_to_rel = {
        int(f.name.split('_')[2]): rel
        for f, rel in zip(gallery_files, gallery_global_ids)
    }

    # Build mapping
    mapping = {}

    for traj_q, traj_g in zip(df[query_col], df[gallery_col]):
        if pd.isna(traj_q) or pd.isna(traj_g):
            continue

        traj_q = int(traj_q)
        traj_g = int(traj_g)

        # Convert trajectory → relative
        rel_q = query_traj_to_rel.get(traj_q)
        rel_g = gallery_traj_to_rel.get(traj_g)

        if rel_q is None or rel_g is None:
            continue

        # Convert to strings
        mapping[str(rel_q)] = str(rel_g)

    return mapping

def _write_ap_csv(folder, q_labels, per_query_AP, id_match_map, mapped_mask):
    """Write a CSV file with AP for all queries that have GT gallery matches."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "ap_per_query.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Query", "Gallery_GT", "AP"])
        for i, qid in enumerate(q_labels):
            if not mapped_mask[i]: continue
            w.writerow([qid, id_match_map.get(qid, ""), float(per_query_AP[i])])

def _write_ap_csv_val(folder, q_labels, per_query_AP):
    """Write a CSV file with AP for all queries in the validation (query→query) setting."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "ap_per_query_validation.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Query", "AP"])
        for i, qid in enumerate(q_labels):
            w.writerow([qid, float(per_query_AP[i])])

def _write_map_txt(folder, mAP_value, mapped_count=None, total_queries=None):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "mAP.txt")
    with open(path, "w") as f:
        f.write(f"mAP_over_mapped_queries: {float(mAP_value):.6f}\n")
        if mapped_count is not None and total_queries is not None:
            f.write(f"mapped_queries/total: {mapped_count}/{total_queries}\n")

# ---------- Plotting helpers ----------
def plot_cross_top_matches(
    query_images,
    query_labels,
    query_embeddings,
    gallery_images,
    gallery_labels,
    gallery_embeddings,
    cross_similarity,  # shape: (Nq, Ng), higher = more similar (assumed cosine)
    save_folder,
    queries_per_sheet=15,
    matches_per_query=6,
    per_query_AP=None,      # optional (AP per query for the chosen similarity)
    plot_q_indices=None,    # optional subset or pre-ordered indices
    mark_correct=True,     # draw green rectangle for correct matches (via id_match_map)
    id_match_map=None       # dict mapping query_id -> gallery_id (GT across sets)
):
    """
    Visualizes cross-set retrieval: for each query, shows top-K gallery matches.
    Titles show both cosine similarity and L2 distance (computed from embeddings).

    If mark_correct=True and id_match_map is provided, a green rectangle is drawn when:
        gallery_labels[gj] == id_match_map[query_labels[qi]].
    """

    # --- Normalize inputs ---
    S = np.asarray(cross_similarity)
    Q = np.asarray(query_embeddings)
    G = np.asarray(gallery_embeddings)
    q_labels = np.asarray(query_labels)
    g_labels = np.asarray(gallery_labels)

    Nq, Ng = S.shape
    assert Q.shape[0] == Nq, "query_embeddings rows must match cross_similarity rows"
    assert G.shape[0] == Ng, "gallery_embeddings rows must match cross_similarity cols"

    # Determine which queries to plot
    if plot_q_indices is None:
        ordered_q_indices = list(range(Nq))
    else:
        ordered_q_indices = list(plot_q_indices)

    # --- Utility: convert images to HWC if needed ---
    def _to_hwc(img):
        if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
            try:
                # torch.Tensor
                return img.permute(1, 2, 0).detach().cpu().numpy()
            except Exception:
                try:
                    # NumPy C,H,W
                    return np.moveaxis(img, 0, -1)
                except Exception:
                    return img
        return img

    # --- Plot sheets ---
    os.makedirs(save_folder, exist_ok=True)

    num_queries = len(ordered_q_indices)
    for sheet_start in range(0, num_queries, queries_per_sheet):
        rows_this_sheet = min(queries_per_sheet, num_queries - sheet_start)

        fig, axes = plt.subplots(
            rows_this_sheet,
            matches_per_query + 1,
            figsize=(2 * (matches_per_query + 1), 1 * rows_this_sheet),
        )
        if rows_this_sheet == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, global_row in enumerate(range(sheet_start, sheet_start + rows_this_sheet)):
            qi = ordered_q_indices[global_row]
            q_lab = q_labels[qi]

            # --- Query image ---
            q_img = _to_hwc(query_images[qi])
            axes[r, 0].imshow(q_img)
            ap_text = f"{per_query_AP[qi]:.2f}" if per_query_AP is not None else ""
            axes[r, 0].set_title(f"Q:{q_lab}" + (f" | AP:{ap_text}" if ap_text else ""), fontsize=8)
            axes[r, 0].axis("off")

            # --- Top-K gallery matches by similarity ---
            sims_row = S[qi]  # (Ng,)
            ranked_g = np.argsort(-sims_row)[:matches_per_query]

            for c, gj in enumerate(ranked_g, start=1):
                m_img = _to_hwc(gallery_images[gj])
                axes[r, c].imshow(m_img)

                cos_sim = float(sims_row[gj])
                l2_dist = float(np.linalg.norm(Q[qi] - G[gj]))

                axes[r, c].set_title(
                    f"G:{g_labels[gj]} (#{c})\nSim:{cos_sim:.3f}",
                    fontsize=8
                )
                axes[r, c].axis("off")

                # ✅ Correctness via GT mapping (query_id -> gallery_id)
                if mark_correct and id_match_map is not None:
                    gt_gid = id_match_map.get(q_lab, None)
                    if gt_gid is not None and g_labels[gj] == gt_gid:
                        h, w = m_img.shape[:2]
                        rect = patches.Rectangle((0, 0), w, h, linewidth=2.0, edgecolor="green",
                                                 facecolor="none", zorder=6)
                        axes[r, c].add_patch(rect)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        out_path = os.path.join(save_folder, f"cross_sheet_{sheet_start // queries_per_sheet}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

def _order_queries_by_top1(S, q_labels, subset_indices=None, group_ids=True, strict_global=False):
    """
    Orders query indices based on their top-1 similarity score.

    Parameters
    ----------
    S : array
        Similarity matrix (N_query x N_gallery).
    q_labels : array
        Query identity labels.
    subset_indices : list or None
        If given, only reorder these queries (returning global indices).
    group_ids : bool
        If True, group queries by label and sort groups by their best top-1 score.
    strict_global : bool
        If True, ignore labels and sort all queries purely by top-1 score.

    Returns
    -------
    list[int]
        Ordered list of global query indices.
    """

    S = np.asarray(S); q_labels = np.asarray(q_labels)
    if subset_indices is None:
        idxs = list(range(S.shape[0])); S_view = S; q_view = q_labels
    else:
        idxs = list(subset_indices); S_view = S[idxs]; q_view = q_labels[idxs]
    top1 = S_view.max(axis=1)
    if strict_global:
        local = sorted(range(len(idxs)), key=lambda k: top1[k], reverse=True)
        return [idxs[k] for k in local]
    if group_ids:
        lab2q = defaultdict(list)
        for k, lab in enumerate(q_view): lab2q[lab].append(k)
        lab_score = {lab: float(np.max(top1[lab2q[lab]])) for lab in lab2q}
        ordered_labs = sorted(lab2q.keys(), key=lambda lab: lab_score[lab], reverse=True)
        order = []
        for lab in ordered_labs:
            inside = sorted(lab2q[lab], key=lambda k: top1[k], reverse=True)
            order.extend(inside)
        return [idxs[k] for k in order]
    return idxs

# ---------- Evaluation helpers ----------
def compute_map_with_gt_cross(S, q_labels, g_labels, id_match_map):
    """ Compute mAP given a cross-similarity matrix and a GT mapping (query_id -> gallery_id). """
    S = np.asarray(S)
    q_labels = np.asarray(q_labels)
    g_labels = np.asarray(g_labels)

    Nq = len(q_labels)
    per_query_AP = np.zeros(Nq, dtype=np.float32)
    per_query_CMC_1 = np.zeros(Nq, dtype=np.float32)
    per_query_CMC_5 = np.zeros(Nq, dtype=np.float32)

    # Which queries have a valid mapping?
    mapped_mask = np.array([lbl in id_match_map for lbl in q_labels], dtype=bool)

    for i in np.where(mapped_mask)[0]:
        target = id_match_map[q_labels[i]]
        pos_mask = (g_labels == target)
        if not pos_mask.any():
            continue

        sims = S[i]
        order = np.argsort(-sims)
        pos_order = pos_mask[order]
        per_query_CMC_1[i] = float(pos_order[:1].any())
        per_query_CMC_5[i] = float(pos_order[:5].any())

        # Compute AP
        ranks = np.arange(1, len(pos_order) + 1)
        hits = pos_order.astype(np.float32)
        cum_hits = np.cumsum(hits)
        if hits.sum() > 0:
            per_query_AP[i] = (cum_hits[hits == 1] / ranks[hits == 1]).mean()
            
    cmc_at_1 = per_query_CMC_1[mapped_mask].mean()
    cmc_at_5 = per_query_CMC_5[mapped_mask].mean()
    mAP = per_query_AP[mapped_mask].mean() if mapped_mask.any() else 0.0
    return mAP, per_query_AP, mapped_mask, cmc_at_1, cmc_at_5

# ---------- Main function ----------
def calculate_cross_q1_and_ensemble_plots_fused(
    parts_cfg,
    query_loader,
    gallery_loader,
    save_folder,
    device='cuda',
    plot_q_indices=None, queries_per_sheet=15, matches_per_query=6,
    part_strengths={}, tau_by_part={},
    lambda_mix=0.6, rrf_k=60,
    group_ids=True, strict_global=False,
    idmatch_xlsx_path="data/reid/a15_a16_idmatch_with_traj_IDs.xlsx",
    match_files_and_ids=None,
    quarter_models='sliced',
    holdout_patches = [],
    plot_images = True,
):

    # ------------------------------------------------------------------------------------------
    # Helper: forward batch through all part-models
    # ------------------------------------------------------------------------------------------
    def forward_parts_batch(batch):
        """
        Returns:
        {
            part: emb_tensor,
            "images": {part: [imgs...]},
            "labels": [...],
            "filenames": [...]
        }
        """

        (g_q1, q1s0, q1s1, q1s2,
         g_q2, q2s0, q2s1, q2s2,
         head_imgs, dorsal_imgs, complete_imgs,
         file_names, anchor_labels) = batch

        def proc(p, imgs):
            return p(images=imgs, return_tensors="pt")["pixel_values"].to(device)

        out = {
            "Q1": None, "Q2": None, "head": None, "dorsal_fin": None,
            "complete_img": None,
            "images": {
                "Q1": list(g_q1),
                "Q2": list(g_q2),
                "head": list(head_imgs),
                "dorsal_fin": list(dorsal_imgs),
                "complete_img": list(complete_imgs) if "complete_img" in parts_cfg else None,
            },
            "labels": anchor_labels.cpu().tolist(),
            "filenames": list(file_names),
        }

        use_amp = ("cuda" in str(device))

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):

            # ---------------------- Q1 ----------------------
            m1, p1 = parts_cfg["Q1"]["model"], parts_cfg["Q1"]["processor"]
            if quarter_models == "sliced":
                r = m1(
                    x_q1 = proc(p1, g_q1),
                    x_q1s0 = proc(p1, q1s0),
                    x_q1s1 = proc(p1, q1s1),
                    x_q1s2 = proc(p1, q1s2),
                )
                out["Q1"] = ensure_2d_embedding(r[-1]).cpu()
            else:
                out["Q1"] = ensure_2d_embedding(m1(proc(p1, g_q1))).cpu()

            # ---------------------- Q2 ----------------------
            m2, p2 = parts_cfg["Q2"]["model"], parts_cfg["Q2"]["processor"]
            if quarter_models == "sliced":
                r = m2(
                    x_q2 = proc(p2, g_q2),
                    x_q2s0 = proc(p2, q2s0),
                    x_q2s1 = proc(p2, q2s1),
                    x_q2s2 = proc(p2, q2s2),
                )
                out["Q2"] = ensure_2d_embedding(r[-1]).cpu()
            else:
                out["Q2"] = ensure_2d_embedding(m2(proc(p2, g_q2))).cpu()

            # ---------------------- HEAD ----------------------
            mh, ph = parts_cfg["head"]["model"], parts_cfg["head"]["processor"]
            out["head"] = ensure_2d_embedding(mh(proc(ph, head_imgs))).cpu()

            # ---------------------- DORSAL --------------------
            md, pd = parts_cfg["dorsal_fin"]["model"], parts_cfg["dorsal_fin"]["processor"]
            out["dorsal_fin"] = ensure_2d_embedding(md(proc(pd, dorsal_imgs))).cpu()

            # ---------------------- COMPLETE IMG --------------
            if "complete_img" in parts_cfg:
                mc = parts_cfg["complete_img"]["model"]
                pc = parts_cfg["complete_img"]["processor"]
                out["complete_img"] = ensure_2d_embedding(mc(proc(pc, complete_imgs))).cpu()

        return out

    # ==========================================================================================
    # RUN FORWARD PASSES FOR QUERY + GALLERY
    # ==========================================================================================
    def accumulate_outputs(loader):
        """Runs forward pass and accumulates embeddings + images."""
        out = {
            "Q1": [], "Q2": [], "head": [], "dorsal_fin": [], "complete_img": [],
            "images": {"Q1": [], "Q2": [], "head": [], "dorsal_fin": [], "complete_img": []},
            "labels": [],
            "filenames": [],
        }

        for batch in loader:
            b = forward_parts_batch(batch)

            for part in ["Q1","Q2","head","dorsal_fin","complete_img"]:
                if b[part] is not None:
                    out[part].append(b[part])

            for part in out["images"]:
                imgs = b["images"][part]
                if imgs:
                    out["images"][part].extend(imgs)

            out["labels"].extend(b["labels"])
            out["filenames"].extend(b["filenames"])

        # convert list of chunks → np arrays
        for part in ["Q1","Q2","head","dorsal_fin","complete_img"]:
            if out[part]:
                out[part] = np.concatenate(out[part], axis=0)
            else:
                out[part] = None

        out["labels"] = _normalize_array_as_str(out["labels"])
        return out

    print("[INFO] Forward pass: queries")
    Q = accumulate_outputs(query_loader)

    print("[INFO] Forward pass: gallery")
    G = accumulate_outputs(gallery_loader)

    # ==========================================================================================
    # VALIDATION = QUERY→QUERY
    # ==========================================================================================
    def compute_val_mAP(emb):
        S = cosine_similarity(emb, emb)
        np.fill_diagonal(S, -np.inf)
        return compute_map(S, Q["labels"])

    validation_results = {}
    per_query_ap_val = {}
    for part in ["Q1","Q2","head","dorsal_fin","complete_img"]:
        if Q[part] is not None:
            validation_results[part], per_query_ap_val[part] = compute_val_mAP(Q[part])

    # Ensemble validation
    def fuse_self(emb_dict):
        S_by_part = {}
        for part in ["Q1","Q2","head","dorsal_fin"]:
            if part in holdout_patches:
                continue
            S_part = cosine_similarity(emb_dict[part], emb_dict[part])
            np.fill_diagonal(S_part, -np.inf)
            S_by_part[part] = S_part

        return _build_fused_cross_similarity(
            S_by_part=S_by_part,
            strengths=part_strengths,
            tau_by_part=tau_by_part,
            lambda_mix=lambda_mix, rrf_k=rrf_k,
        )

    S_val_ens = fuse_self(Q)
    validation_results["ensemble"], per_query_ap_val["ensemble"] = compute_map(S_val_ens, Q["labels"])

    # ==========================================================================================
    # CROSS-SET TEST (QUERY→GALLERY)
    # ==========================================================================================
    # Part-to-part similarities
    S_cross = {
        part: cosine_similarity(Q[part], G[part])
        for part in ["Q1","Q2","head","dorsal_fin","complete_img"]
        if Q[part] is not None
    }
    print(S_cross['Q2'].shape)

    # Fused ensemble
    S_ensemble_cross = _build_fused_cross_similarity(
        S_by_part={k: S_cross[k] for k in ["Q1","Q2","head","dorsal_fin"] if k not in holdout_patches},
        strengths=part_strengths,
        tau_by_part=tau_by_part,
        lambda_mix=lambda_mix,
        rrf_k=rrf_k,
    )

    # Load GT matching
    id_match_map = load_id_match_map_from_traj_excel(
        idmatch_xlsx_path,
        match_files_and_ids["query_files"],
        match_files_and_ids["query_ids"],
        match_files_and_ids["gallery_files"],
        match_files_and_ids["gallery_ids"],
    )

    # mAP for ensemble
    mAP_ens, ap_ens, mapped_mask_ens, cmc_at_1_ens, cmc_at_5_ens = compute_map_with_gt_cross(
        S_ensemble_cross, Q["labels"], G["labels"], id_match_map
    )
    
    # (optional) write ensemble AP + summary
    ensemble_dir = os.path.join(save_folder, "plots_complete_to_complete__ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    _write_ap_csv(ensemble_dir, Q["labels"], ap_ens, id_match_map, mapped_mask_ens)
    _write_ap_csv_val(ensemble_dir, Q["labels"], per_query_ap_val["ensemble"])
    _write_map_txt(ensemble_dir, mAP_ens, mapped_mask_ens.sum(), len(Q["labels"]))

    # mAP for individual parts
    mAP_parts = {}
    cmc_parts = {}
    for part in S_cross:
        mAP_p, _, _, cmc_at_1_p, cmc_at_5_p = compute_map_with_gt_cross(
            S_cross[part], Q["labels"], G["labels"], id_match_map
        )
        mAP_parts[part] = mAP_p
        cmc_parts[part] = (cmc_at_1_p, cmc_at_5_p)

    # ==========================================================================================
    # WRITE SUMMARY
    # ==========================================================================================
    os.makedirs(save_folder, exist_ok=True)
    summary_path = os.path.join(save_folder, "all_results_summary.txt")

    with open(summary_path, "w") as f:
        f.write("==== VALIDATION RESULTS (QUERY→QUERY) ====\n")
        for part, map_score in validation_results.items():
            f.write(f"{part}: mAP {map_score:.4f}\n")

        f.write("\n==== TEST RESULTS (QUERY→GALLERY) ====\n")
        f.write(f"ensemble: mAP {mAP_ens:.4f}\n")
        for part, score in mAP_parts.items():
            f.write(f"{part}: mAP {score:.4f}\n")
        f.write(f"ensemble: CMC@1 {cmc_at_1_ens:.4f}, CMC@5 {cmc_at_5_ens:.4f}\n")
        for part, score in cmc_parts.items():
            f.write(f"{part}: CMC@1 {score[0]:.4f}, CMC@5 {score[1]:.4f}\n")

    print(f"[INFO] Summary written to {summary_path}")

    # ==========================================================================================
    # PLOTTING — SLOW, DONE LAST
    # ==========================================================================================

    subset = list(map(int, plot_q_indices)) if plot_q_indices else None
    ordered_q_for_ens = _order_queries_by_top1(
        S_ensemble_cross, Q["labels"],
        subset_indices=subset, group_ids=group_ids, strict_global=strict_global
    )

    # Folders
    d_q1 = os.path.join(save_folder, "plots_q1_to_q1__ensemble")
    d_c  = os.path.join(save_folder, "plots_complete_to_complete__ensemble")
    os.makedirs(d_q1, exist_ok=True)
    os.makedirs(d_c,  exist_ok=True)

    if plot_images:
        print("[INFO] Plotting ensemble...")
        plot_cross_top_matches(
           query_images=Q["images"]["Q1"],
           query_labels=Q["labels"],
           query_embeddings=Q["Q1"],
           gallery_images=G["images"]["Q1"],
           gallery_labels=G["labels"],
           gallery_embeddings=G["Q1"],
           cross_similarity=S_ensemble_cross,
           save_folder=d_q1,
           queries_per_sheet=queries_per_sheet,
           matches_per_query=matches_per_query,
           per_query_AP=ap_ens,
           plot_q_indices=ordered_q_for_ens,
           mark_correct=True,
           id_match_map=id_match_map
        )

        if "complete_img" in parts_cfg:
            plot_cross_top_matches(
               query_images=Q["images"]["complete_img"],
               query_labels=Q["labels"],
               query_embeddings=Q["Q1"],     # same as your previous design
               gallery_images=G["images"]["complete_img"],
               gallery_labels=G["labels"],
               gallery_embeddings=G["Q1"],
               cross_similarity=S_ensemble_cross,
               save_folder=d_c,
               queries_per_sheet=queries_per_sheet,
               matches_per_query=matches_per_query,
               per_query_AP=ap_ens,
               plot_q_indices=ordered_q_for_ens,
               mark_correct=True,
               id_match_map=id_match_map
            )

    # Per-part plots
    print("[INFO] Plotting parts...")
    for part in S_cross:
        part_dir = os.path.join(save_folder, f"part2part__{part}")
        os.makedirs(part_dir, exist_ok=True)

        S_part = S_cross[part]
        mAP_p, ap_p, mapped_mask_p, cmc_at_1_p, cmc_at_5_p = compute_map_with_gt_cross(
            S_part, Q["labels"], G["labels"], id_match_map
        )

        ordered_q = _order_queries_by_top1(
            S_part, Q["labels"], subset_indices=subset,
            group_ids=group_ids, strict_global=strict_global
        )
        
        if plot_images:
            plot_cross_top_matches(
                query_images=Q["images"][part],
                query_labels=Q["labels"],
                query_embeddings=Q[part],
                gallery_images=G["images"][part],
                gallery_labels=G["labels"],
                gallery_embeddings=G[part],
                cross_similarity=S_part,
                save_folder=part_dir,
                queries_per_sheet=queries_per_sheet,
                matches_per_query=matches_per_query,
                per_query_AP=ap_p,
                plot_q_indices=ordered_q,
                mark_correct=True,
                id_match_map=id_match_map
            )

        _write_ap_csv(part_dir, Q["labels"], ap_p, id_match_map, mapped_mask_p)
        _write_ap_csv_val(part_dir, Q["labels"], per_query_ap_val[part])
        _write_map_txt(part_dir, mAP_p, mapped_mask_p.sum(), len(Q["labels"]))

    print("[INFO] All plotting done.")

    # ==========================================================================================
    # RETURN RESULTS
    # ==========================================================================================
    return {
        "validation_mAP": validation_results,
        "test_mAP_ensemble": mAP_ens,
        "test_mAP_parts": mAP_parts,
        "S_ensemble_cross": S_ensemble_cross,
    }


