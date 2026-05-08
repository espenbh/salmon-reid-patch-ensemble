
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from pathlib import Path
import random
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from augmentation_helpers import *
from torchvision import transforms
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from transformers import AutoModel, AutoImageProcessor
from pytorch_metric_learning.losses import ArcFaceLoss
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler
from pytorch_metric_learning import losses, reducers
from itertools import zip_longest
from collections.abc import Mapping, Sequence
from torch.utils.data._utils.collate import default_collate
from typing import Union, Optional, Dict, List, Sequence, Tuple, Any
import pandas as pd


# --- General helpers ---
def create_analysis_path(config):
    if 'analysis_path' not in config.keys():
        # Ensure out_root is a Path object
        out_root = Path(config['out_root'])
        
        # Find next available analysis number
        i = 1
        while True:
            analysis_path = out_root / f'analysis{i}'
            if not analysis_path.exists():
                analysis_path.mkdir(parents=True, exist_ok=False)
                break
            i += 1
        
        # Save analysis path in config
        config['analysis_path'] = str(analysis_path)
        
        # Convert all Path objects in config to strings
        serializable_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        
        # Save config as JSON inside analysis folder
        config_file = analysis_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(serializable_config, f, indent=4)
        return config

def get_file_names_and_labels(data_root, folders):
    'Get all file names and assign global IDs across folders'
    id_num = 1
    files = []
    global_ids = []
    for folder in data_root.iterdir():
        if folder.name in folders:
            folder_files = list((folder/Path('labels')).iterdir())

            local_to_global_id = {}
            ids = []
            for f in folder_files:
                id = int(f.name.split('_')[2])
                if id in local_to_global_id.keys():
                    ids.append(local_to_global_id[id])
                else:
                    local_to_global_id[id] = id_num
                    id_num += 1
                    ids.append(local_to_global_id[id])
            global_ids.extend(ids)
            files.extend(folder_files)
    return files, global_ids

def excel_match_file_to_traj_IDs(data_root, query_folder, gallery_folder, match_excel_path, output_excel_path):
    """
    Replace global IDs in match Excel file with original trajectory IDs from filenames,
    using separate folder calls for query and gallery.
    
    Parameters:
    - data_root: Path to the root directory containing analysis folders.
    - query_folder: Name of the folder for query data (e.g., 'analysis15').
    - gallery_folder: Name of the folder for gallery data (e.g., 'analysis16').
    - match_excel_path: Path to the existing Excel file with global ID matches.
    - output_excel_path: Path to save the new Excel file with original trajectory IDs.
    """
    # Get query files and global IDs
    query_files, query_global_ids = get_file_names_and_labels(data_root, [query_folder])
    for f, gid in zip(query_files, query_global_ids):
        print(f.name, gid)
    query_id_map = {gid: int(f.name.split('_')[2]) for f, gid in zip(query_files, query_global_ids)}

    # Get gallery files and global IDs
    gallery_files, gallery_global_ids = get_file_names_and_labels(data_root, [gallery_folder])
    gallery_id_map = {gid: int(f.name.split('_')[2]) for f, gid in zip(gallery_files, gallery_global_ids)}

    # Read the existing match file
    df = pd.read_excel(match_excel_path)

    # Replace global IDs with trajectory IDs
    df['Query'] = df['Query'].map(query_id_map)
    df['Gallery'] = df['Gallery'].map(gallery_id_map)

    # Save the updated DataFrame
    df.to_excel(output_excel_path, index=False)


def get_positive_sample_idces(anchor_idx, labels):
    'Get indices of all positive samples for a given anchor index'
    anchor_label = labels[anchor_idx]
    return [i for lab, i in zip(labels, range(len(labels))) if lab == anchor_label and i != anchor_idx]

def get_negative_sample_idces(anchor_idx, labels):
    'Get indices of all negative samples for a given anchor index'
    anchor_label = labels[anchor_idx]
    return [i for lab, i in zip(labels, range(len(labels))) if lab != anchor_label]

def subsample_test_set(files, global_ids, samples_per_id=5, test_num_IDs = 'all'):
    """
    Subsample the test set by uniformly picking a fixed number of samples per ID (deterministic),
    discarding the first and last sample for each ID.

    Args:
        files (list): List of file paths.
        global_ids (list or array): List of class IDs corresponding to files.
        samples_per_id (int): Number of samples to keep per ID.

    Returns:
        refined_files (np.ndarray): Subsampled file paths.
        refined_global_ids (np.ndarray): Subsampled IDs.
    """

    ids_to_keep = []

    unique_labels = sorted(set(global_ids))  # consistent order
    
    if test_num_IDs != 'all':
        unique_labels = unique_labels[:test_num_IDs]  # take first N IDs

    for label in unique_labels:
        # Get all indices for this label
        lab_indices = [i for i, l in enumerate(global_ids) if l == label]

        # Discard first and last
        if len(lab_indices) > 2:
            lab_indices = lab_indices[1:-1]

        # If fewer samples than requested, take all
        if len(lab_indices) <= samples_per_id:
            chosen = lab_indices
        else:
            # Uniformly spaced indices
            step = len(lab_indices) / samples_per_id
            chosen = [lab_indices[int(round(i * step))] for i in range(samples_per_id)]

        ids_to_keep.extend(chosen)

    refined_files = np.array(files)[ids_to_keep]
    refined_global_ids = np.array(global_ids)[ids_to_keep]
    return refined_files, refined_global_ids

# --- Collate function ---
def collate_unified(batch, network='vit_ensamble_singleforward'):
    # Filter valid items
    valid_items = []
    for item in batch:
        if item is None:
            continue
        if any((x is None) for x in item):
            continue
        valid_items.append(item)
    if not valid_items:
        return []

    length = len(valid_items[0])
    result = [[] for _ in range(length)]
    for item in valid_items:
        for i in range(length):
            result[i].append(item[i])

    result[-1] = torch.Tensor(result[-1])  # labels

    #if network in ('vit_complete_img', 'vit_dino', 'convnext_dino'):
    #    result[1] = torch.Tensor(result[1])

    #elif network == 'vit_ensamble_decoupled':
    #    result[-1] = torch.Tensor(result[-1])  # labels

    #elif network == 'vit_ensamble_joint' or network == 'vit_ensamble_joint_inc_compimg':
        # dual-source joint: keep image lists; convert labels to Tensor
    #    result[-1] = torch.Tensor(result[-1])

    return result


# --- Datasets ---
class SalmonBaseDataset(Dataset):
    def __init__(self, files, labels, transform, difficulty,
                 imgs_to_load, crop_slices_to_mask, slice_split_idces, network):
        self.data = files
        self.labels = labels
        self.transform = transform
        self.difficulty = difficulty
        self.imgs_to_load = imgs_to_load
        self.crop_slices_to_mask = crop_slices_to_mask
        self.slice_split_idces = slice_split_idces
        self.network = network

    def __len__(self):
        return len(self.data)

    def load_img_patches(self, label_path):
        if any(k in self.imgs_to_load for k in ['Q1', 'Q2', 'complete_img']):
            return load_patches(
                label_path,
                difficulty=self.difficulty,
                imgs_to_load=self.imgs_to_load,
                crop_slices_to_mask=self.crop_slices_to_mask,
                slice_split_idces=self.slice_split_idces
            )
        else:
            return load_tracker_patches(
                label_path,
                difficulty=self.difficulty,
                imgs_to_load=self.imgs_to_load
            )

    def load_and_augment(self, label_path):
        patches = self.load_img_patches(label_path)

        # vit_dino / convnext / vit_complete_img
        if self.network in ['vit_dino', 'vit_complete_img', 'convnext_dino']:
            assert len(self.imgs_to_load) == 1
            key = self.imgs_to_load[0]
            return self.transform(patches[key]['img']), patches[key]['valid']

        # ensemble
        if self.network == 'vit_ensamble_decoupled' or self.network == 'test_ensamble_vit':
            res = []
            valid = True

            candidates = ["Q1", "Q2", "head", "dorsal_fin", "complete_img"]
            parent_names = [name for name in candidates if any(k.startswith(name) for k in patches.keys())]

            #parent_names = sorted({n.split('_')[0] for n in patches.keys()})
            #if 'complete' in parent_names:
            #    # Exchange complete with complete_img
            #    parent_names.remove('complete')
            #    parent_names.append('complete_img')

            for parent_name in parent_names:
                res.append(patches[parent_name]['img'])
                valid = valid and patches[parent_name]['valid']

                children = sorted([n for n in patches.keys()
                                   if n.startswith(parent_name + '_')])
                for child in children:
                    res.append(patches[child]['img'])

            res.append(valid)
            return res

        raise NotImplementedError(f"Unknown network: {self.network}")

class SalmonTestDataset(SalmonBaseDataset):
    def __getitem__(self, idx):
        result = self.load_and_augment(self.data[idx])

        if self.network == 'vit_ensamble_decoupled' or self.network == 'test_ensamble_vit':
            if result[-1]:
                return result[:-1] + [self.data[idx]] + [self.labels[idx]]
            return None

        # standard
        img, valid = result
        return (img, self.labels[idx]) if valid else None

class SalmonTrainDataset(SalmonBaseDataset):
    def __getitem__(self, idx):
        lbl = int(self.labels[idx])

        # try original
        result = self.load_and_augment(self.data[idx])

        if self.network == 'vit_ensamble_decoupled':
            if result[-1]:
                return result[:-1] + [lbl]
        else:
            img, valid = result
            if valid:
                return img, lbl

        # resampling
        same_cls = [i for i, l in enumerate(self.labels) if int(l) == lbl and i != idx]

        for _ in range(5):
            if not same_cls:
                break
            cand = random.choice(same_cls)
            result = self.load_and_augment(self.data[cand])

            if self.network == 'vit_ensamble_decoupled':
                if result[-1]:
                    return result[:-1] + [lbl]
            else:
                img, valid = result
                if valid:
                    return img, lbl

        # fallback placeholder
        placeholder = np.zeros((64, 256, 3), dtype=np.uint8)
        if self.network == 'vit_ensamble_decoupled':
            return [placeholder] * (len(self.imgs_to_load)) + [lbl]
        return placeholder, lbl

class TrueMPerClassSampler(torch.utils.data.Sampler):
    def __init__(self, labels, m, batch_size, length_before_new_iter=None):
        self.labels = np.array(labels)
        self.m = m
        self.batch_size = batch_size
        self.unique_labels = np.unique(labels)
        self.num_classes_per_batch = batch_size // m
        self.length_before_new_iter = length_before_new_iter or len(labels)

    def __iter__(self):
        num_batches = self.length_before_new_iter // self.batch_size
        for _ in range(num_batches):
            chosen_labels = np.random.choice(self.unique_labels, self.num_classes_per_batch, replace=False)
            batch = []
            for label in chosen_labels:
                idxs = np.where(self.labels == label)[0]
                # Always pick exactly m samples (duplicate if needed)
                batch.extend(np.random.choice(idxs, self.m, replace=True))
            yield batch

    def __len__(self):
        return self.length_before_new_iter // self.batch_size

def get_dataloader(config, difficulty, shuffle, loader_type = 'train', load_raw_images = False, num_IDs=4, samples_per_ID=5, test_num_IDs = 'all', imgs_to_load = ['Q1_s1']):
    # Transform pipeline
    if load_raw_images:
        transform = transforms.Compose([])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts HWC [0-255] NumPy → CHW FloatTensor [0-1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Optional normalization
            ])

    if loader_type  == 'train':
        train_files, train_global_ids = get_file_names_and_labels(config['data_root'], config['training_folders'])
        train_dataset = SalmonTrainDataset(train_files, train_global_ids, transform=transform, difficulty=difficulty, imgs_to_load = imgs_to_load, crop_slices_to_mask=config['crop_slices_to_mask'], slice_split_idces=config['slice_split_idces'], network=config['network'])
        sampler = TrueMPerClassSampler(train_global_ids, m=samples_per_ID, batch_size=samples_per_ID*num_IDs, length_before_new_iter=len(train_global_ids))
        dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=lambda batch: collate_unified(batch, network=config['network']))

    elif loader_type == 'test':
        test_files, test_global_ids = get_file_names_and_labels(config['data_root'], config['test_folders'])
        test_files, test_global_ids = subsample_test_set(test_files, test_global_ids, samples_per_id=samples_per_ID, test_num_IDs = test_num_IDs)
        test_dataset = SalmonTestDataset(test_files, test_global_ids, transform=transform, difficulty = difficulty, imgs_to_load = imgs_to_load, crop_slices_to_mask=config['crop_slices_to_mask'], slice_split_idces=config['slice_split_idces'], network=config['network'])
        dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=lambda batch: collate_unified(batch, network=config['network']))

    return dataloader


# --- Networks ---
class ConvnextWithProjection(nn.Module):
    def __init__(self, base_model, embedding_dim=1024, use_layernorm=True, dropout=0.1):
        super().__init__()
        self.base_model = base_model

        # A light, stable projector
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

        # Stabilize feature scales before normalization
        self.norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        features = outputs.pooler_output  # [B, embedding_dim]

        proj = self.projection(features)
        proj = self.norm(proj)
        proj = F.normalize(proj, p=2, dim=1)  # unit-length embeddings for ArcFace & Triplet
        return proj

class ViTWithProjection(nn.Module):
    def __init__(self, base_model, embedding_dim=768, use_layernorm=True, dropout=0.1):
        super().__init__()
        self.base_model = base_model

        # Light projector for CLS token
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

        self.norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()

    def forward(self, pixel_values, return_tokens=False):
        outputs = self.base_model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, embedding_dim]
        patch_tokens = outputs.last_hidden_state[:, 1:197, :]  # [B, num_patches, embedding_dim]

        proj = self.projection(cls_token)
        proj = self.norm(proj)
        proj = F.normalize(proj, p=2, dim=1)

        if return_tokens:
            return proj, patch_tokens  # CLS embedding + raw tokens
        else:
            return proj

class ViTMultiHeadCLS(nn.Module):
    def __init__(self, base_model, embedding_dim=768,
                 head_names=('Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2'),
                 dropout=0.1, use_layernorm=True):
        # Remove fused and handle explicitely
        head_names = [h for h in head_names if h != 'fused']
        self.ordered_heads = [h for h in ['Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2', 'Q2', 'Q2_s0', 'Q2_s1', 'Q2_s2'] if h in head_names]
        super().__init__()
        self.base_model = base_model  # e.g., HuggingFace ViT backbone
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim),
            ) for name in head_names
        })
        self.norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()

        # Fusion projection head
        fused_dim = embedding_dim * len(head_names)
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward_single(self, pixel_values, head_key):
        """
        Forward pass for a single crop and its corresponding head.
        Args:
            pixel_values: Tensor [B, C, H, W]
            head_key: str, one of head_names
        Returns:
            L2-normalized embedding [B, embedding_dim]
        """
        outputs = self.base_model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        proj = self.heads[head_key](cls_token)
        proj = self.norm(proj)
        proj = F.normalize(proj, p=2, dim=1)
        return proj

    def forward(self, x_q1=None, x_q1s0=None, x_q1s1=None, x_q1s2=None, x_q2=None, x_q2s0=None, x_q2s1=None, x_q2s2=None):
        """
        Convenience forward for 4 crops in one call.
        Args:
            x_q1, x_s0, x_s1, x_s2: Tensors [B, C, H, W]
        Returns:
            global embedding, tuple of slice embeddings
        """
        
        res = []
        if x_q1 is not None and 'Q1' in self.ordered_heads: res.append(self.forward_single(x_q1, 'Q1'))
        if x_q1s0 is not None and 'Q1_s0' in self.ordered_heads: res.append(self.forward_single(x_q1s0, 'Q1_s0'))
        if x_q1s1 is not None and 'Q1_s1' in self.ordered_heads: res.append(self.forward_single(x_q1s1, 'Q1_s1'))
        if x_q1s2 is not None and 'Q1_s2' in self.ordered_heads: res.append(self.forward_single(x_q1s2, 'Q1_s2'))
        if x_q2 is not None and 'Q2' in self.ordered_heads: res.append(self.forward_single(x_q2, 'Q2'))
        if x_q2s0 is not None and 'Q2_s0' in self.ordered_heads: res.append(self.forward_single(x_q2s0, 'Q2_s0'))
        if x_q2s1 is not None and 'Q2_s1' in self.ordered_heads: res.append(self.forward_single(x_q2s1, 'Q2_s1'))
        if x_q2s2 is not None and 'Q2_s2' in self.ordered_heads: res.append(self.forward_single(x_q2s2, 'Q2_s2'))

        # Concatenate and project
        fused = torch.cat(res, dim=1)
        fused = self.fusion_head(fused)
        fused = F.normalize(fused, p=2, dim=1)
        res.append(fused)

        return res

def freeze_vit_layers(base_model, num_layers_to_freeze=6):
    """
    Freeze the first `num_layers_to_freeze` transformer blocks in a ViT model,
    but keep LayerNorm parameters trainable.
    """
    for name, param in base_model.named_parameters():
        # Match transformer blocks like layer.0, layer.1, ...
        if name.startswith("layer."):
            # Extract layer index
            parts = name.split('.')
            try:
                layer_num = int(parts[1])  # after 'layer'
            except ValueError:
                layer_num = -1

            if 0 <= layer_num < num_layers_to_freeze:
                # Keep LayerNorm trainable
                if ("norm" in name.lower()):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

def freeze_convnext_layers(base_model, num_blocks_to_freeze=9):
    """
    Freeze approximately `num_blocks_to_freeze` ConvNeXt blocks in HuggingFace DINOv3 ConvNeXt.
    Keeps LayerNorm trainable for stability.
    """
    frozen_blocks = 0
    for stage_idx, stage in enumerate(base_model.stages):
        # Each stage likely has .layers (list of blocks)
        if hasattr(stage, "layers"):
            for block_idx, block in enumerate(stage.layers):
                if frozen_blocks < num_blocks_to_freeze:
                    for name, param in block.named_parameters():
                        if "norm" in name.lower():
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    frozen_blocks += 1
                else:
                    return  # Stop once enough blocks are frozen

def count_trainable_params(model):
    """
    Prints the number of trainable parameters in the given model.

    Args:
        model (nn.Module): The PyTorch model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.2f}% trainable)")

def load_model(config, device = 'cuda'):
    if config['network']=='vit_dino' or config['network']=='vit_ensamble_decoupled' or config['network']=='vit_complete_img':
        local_model_path = "./vit_base"  # or absolute path if needed
    if config['network']=='convnext_dino':
        local_model_path = "./convnext_base"  # or absolute path if needed
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    base_model = AutoModel.from_pretrained(local_model_path)

    # Freeze
    if config['freeze']:
        print('Before freeze: ')
        count_trainable_params(base_model)
        if config['network']=='vit_dino' or config['network']=='vit_ensamble_decoupled' or config['network']=='vit_complete_img': freeze_vit_layers(base_model, num_layers_to_freeze=6)
        elif config['network']=='convnext_dino': freeze_convnext_layers(base_model, num_blocks_to_freeze=25)
        print('After freeze: ')
        count_trainable_params(base_model)

    # Projection head
    if config['network']=='vit_dino' or config['network']=='vit_complete_img': model = ViTWithProjection(base_model, embedding_dim=config['embedding_dim']).to(device)
    elif config['network']=='convnext_dino': model = ConvnextWithProjection(base_model, embedding_dim=config['embedding_dim']).to(device)
    elif config['network']=='vit_ensamble_decoupled': model = ViTMultiHeadCLS(base_model, embedding_dim=config['embedding_dim'], head_names=config['heads']).to(device)
        
    loss_dict = {}
    for loss in config['losses'].keys():
        if loss == 'arcface':
            loss_dict[loss] = ArcFaceLoss(
            num_classes=config['num_classes'],              # same as your head
            embedding_size=config['embedding_dim'],         # equivalent to embedding_dim
            margin=config['arcface_m'],                     
            scale=config['arcface_s']                       
            ).to(device)
        elif loss == 'triplet':
            triplet_loss = losses.TripletMarginLoss(margin=1.0).to(device)
            triplet_loss.reducer = reducers.AvgNonZeroReducer()
            loss_dict[loss] = triplet_loss
        elif loss == 'smoothap':
            smoothaploss = losses.SmoothAPLoss().to(device)
            smoothaploss.reducer = reducers.AvgNonZeroReducer()
            loss_dict[loss] = smoothaploss

    params = list(model.parameters())
    if 'arcface' in loss_dict.keys(): params += list(loss_dict['arcface'].parameters())
    return model, loss_dict, params, processor

# --- Vizualization helpers ---
def plot_top_matches(
    images,
    labels,
    similarity_matrix,
    embeddings,
    save_folder,
    queries_per_sheet=10,
    matches_per_query=5,
    per_query_AP=None,
    plot_indices=None,
):
    """
    Visualizes query + top matches with an overlayed ViT-style patch grid (commented),
    scaled to the original image dimensions (no resizing for visualization).

    Args:
        images: list/array of images in HxWxC format (or tensors convertible to that).
        labels: array-like labels per image.
        similarity_matrix: (N, N) similarities; higher = more similar (assumed cosine).
        embeddings: (N, D) embeddings, used for L2 distance display.
        save_folder: output folder for sheets.
        queries_per_sheet: rows per saved figure.
        matches_per_query: number of matches to show per query (columns after the query).
        per_query_AP: optional array of AP values per query index.
        plot_indices: optional list of indices to use as queries.
    """
    if plot_indices is None:
        plot_indices = list(range(len(images)))

    num_queries = len(plot_indices)
    for sheet_idx in range(0, num_queries, queries_per_sheet):
        fig, axes = plt.subplots(
            queries_per_sheet,
            matches_per_query + 1,
            figsize=(2 * (matches_per_query + 1), 1 * queries_per_sheet),
        )

        # If queries_per_sheet == 1, axes will be 1D; normalize to 2D indexing
        if queries_per_sheet == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, idx in enumerate(range(sheet_idx, min(sheet_idx + queries_per_sheet, num_queries))):
            query_idx = plot_indices[idx]
            sims = similarity_matrix[query_idx]

            # Sort by descending similarity and remove the query itself
            sorted_idx = np.argsort(-sims)
            sorted_idx = sorted_idx[sorted_idx != query_idx][:matches_per_query]

            # ---- Query image ----
            q_img = images[query_idx]
            # If torch tensor CHW, convert to HWC
            if hasattr(q_img, "shape") and len(q_img.shape) == 3 and q_img.shape[0] in (1, 3) and q_img.shape[0] != q_img.shape[-1]:
                try:
                    q_img = q_img.permute(1, 2, 0).cpu().numpy()
                except Exception:
                    pass  # if not torch tensor, ignore

            axes[row, 0].imshow(q_img)
            ap_text = f"{per_query_AP[query_idx]:.2f}" if per_query_AP is not None else ""
            axes[row, 0].set_title(f"Q:{labels[query_idx]} | AP:{ap_text}", fontsize=8)
            axes[row, 0].axis('off')

            # ---- Matches ----
            for col, match_idx in enumerate(sorted_idx, start=1):
                m_img = images[match_idx]
                # If torch tensor CHW, convert to HWC
                if hasattr(m_img, "shape") and len(m_img.shape) == 3 and m_img.shape[0] in (1, 3) and m_img.shape[0] != m_img.shape[-1]:
                    try:
                        m_img = m_img.permute(1, 2, 0).cpu().numpy()
                    except Exception:
                        pass

                axes[row, col].imshow(m_img)

                # --- Similarity & distance metrics ---
                cos_sim = float(similarity_matrix[query_idx, match_idx])
                l2_dist = float(np.linalg.norm(embeddings[query_idx] - embeddings[match_idx]))

                axes[row, col].set_title(
                    f"ID:{labels[match_idx]} (#{col})\nCos:{cos_sim:.3f} | L2:{l2_dist:.3f}",
                    fontsize=8
                )
                axes[row, col].axis('off')

                # Green rectangle for true matches
                if labels[match_idx] == labels[query_idx]:
                    h, w = m_img.shape[:2]
                    rect = patches.Rectangle(
                        (0, 0), w, h, linewidth=2.0, edgecolor='green', facecolor='none', zorder=6
                    )
                    axes[row, col].add_patch(rect)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"sheet_{sheet_idx // queries_per_sheet}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

def plot_confusion_matrices(similarity: np.ndarray,
                      labels: np.ndarray,
                      out_dir: Path,
                      head: str,
                      mode: str = "hardk",         # "hard1" | "hardk" | "soft"
                      top_k = 'same_id',           # int for fixed-k, or 'same_id' for k_i = n_sameID(i)-1
                      normalize_rows: bool = True):
    """
    Build and save an ID-to-ID confusion matrix from a pairwise similarity matrix.

    Args:
        similarity (np.ndarray): NxN similarity (higher = more similar).
        labels (np.ndarray): length-N array of IDs (integers or strings).
        out_dir (Path): where to save CSVs and heatmap.
        head (str): used in filenames, e.g., 'Q1'/'fused'.
        mode (str): 
            - 'hard1' : each query votes for its top-1 neighbor's ID (leave-one-out).
                        Row sum = (#samples of that row ID).
            - 'hardk' : each query votes for top-k neighbors' IDs (excl. self).
                        If top_k='same_id', per-query k_i = (#samples with same label as the query) - 1.
                        Else if top_k is int>=1, fixed k. 
                        Row sum = sum_i k_i (typically = n_row * k for fixed k).
            - 'soft'  : for each query, accumulate sum of similarities to each ID (excl. self),
                        then (optionally) row-normalize to get a distribution.
        top_k: int or 'same_id' (only used for mode='hardk').
        normalize_rows (bool): if True, row-normalize before saving the *_norm.csv and heatmap.

    Saves:
        - '{head}_confusion_counts.csv' : raw counts (float for 'soft')
        - '{head}_confusion_norm.csv'   : row-normalized matrix
        - '{head}_confusion_heatmap.png': heatmap of normalized matrix

    Returns:
        unique_ids (np.ndarray), cm_norm (np.ndarray), cm_counts (np.ndarray)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels)
    N = similarity.shape[0]
    if labels.shape[0] != N or similarity.shape[1] != N:
        raise ValueError("similarity must be NxN and labels must be length N")

    unique_ids = np.unique(labels)
    U = len(unique_ids)
    id_to_idx = {idv: i for i, idv in enumerate(unique_ids)}

    # We use float64 so 'soft' mode can store similarity mass, and 'hard' modes store integer-like floats.
    cm_counts = np.zeros((U, U), dtype=np.float64)

    # Precompute per-ID index lists (needed by 'soft', and for 'same_id' k)
    id_to_indices = {idv: np.where(labels == idv)[0] for idv in unique_ids}

    if mode == "hard1":
        # Each query contributes exactly one vote to the ID of its best neighbor (excluding self).
        for i in range(N):
            sim_row = similarity[i].copy()
            sim_row[i] = -np.inf  # exclude self
            j = int(np.argmax(sim_row))
            r = id_to_idx[labels[i]]
            c = id_to_idx[labels[j]]
            cm_counts[r, c] += 1.0

    elif mode == "hardk":
        # Each query contributes votes to top-k neighbors (excluding self).
        # k can be fixed int or adaptive ('same_id' => k_i = n_sameID(i)-1).
        for i in range(N):
            sim_row = similarity[i].copy()
            sim_row[i] = -np.inf  # exclude self
            r = id_to_idx[labels[i]]

            if top_k == 'same_id':
                n_same = id_to_indices[labels[i]].size
                k_i = max(0, min(n_same - 1, N - 1))
            else:
                if not isinstance(top_k, int) or top_k < 1:
                    raise ValueError("For mode='hardk', top_k must be int>=1 or 'same_id'.")
                k_i = min(top_k, N - 1)

            if k_i == 0:
                # Edge case: only one sample for this ID → no positives to consider.
                continue

            # Take indices of top k_i neighbors (order within top-k not important)
            # np.argpartition is O(N) average, avoids full sort.
            part_idx = np.argpartition(-sim_row, k_i-1)[:k_i]
            # Increment votes for their IDs
            for j in part_idx:
                c = id_to_idx[labels[j]]
                cm_counts[r, c] += 1.0

    elif mode == "soft":
        # Each query spreads its similarity mass to each ID (excluding self).
        # After row-normalization, each row becomes a probability distribution over predicted IDs.
        for i in range(N):
            sim_row = similarity[i]
            true_id = labels[i]
            r = id_to_idx[true_id]
            for t, idv in enumerate(unique_ids):
                idxs = id_to_indices[idv]
                if idv == true_id:
                    # exclude the query itself if it belongs to this ID
                    idxs = idxs[idxs != i]
                if idxs.size > 0:
                    cm_counts[r, t] += float(sim_row[idxs].sum())
    else:
        raise ValueError("mode must be one of: 'hard1', 'hardk', 'soft'.")

    # Row-normalize (safe for zero-rows)
    if normalize_rows:
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.divide(cm_counts, row_sums, where=row_sums != 0)
            cm_norm[row_sums[:, 0] == 0] = 0
    else:
        cm_norm = cm_counts.copy()

    # Save CSVs with ID headers
    df_counts = pd.DataFrame(cm_counts, index=unique_ids, columns=unique_ids)
    df_norm = pd.DataFrame(cm_norm, index=unique_ids, columns=unique_ids)
    df_counts.to_csv(out_dir / f"{head}_confusion_counts.csv")
    df_norm.to_csv(out_dir / f"{head}_confusion_norm.csv")

    # Save heatmap image (normalized)
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_norm, cmap="viridis", interpolation="nearest", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Row-normalized proportion")

        ax.set_title(f"ID→ID Confusion ({head}, mode={mode}, top_k={top_k})")
        ax.set_xlabel("Predicted ID")
        ax.set_ylabel("True ID")

        max_ticks = 50
        if len(unique_ids) <= max_ticks:
            ax.set_xticks(np.arange(len(unique_ids)))
            ax.set_yticks(np.arange(len(unique_ids)))
            ax.set_xticklabels(unique_ids, rotation=90)
            ax.set_yticklabels(unique_ids)
        else:
            tick_idx = np.linspace(0, len(unique_ids) - 1, num=min(20, len(unique_ids)), dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels(unique_ids[tick_idx], rotation=90)
            ax.set_yticklabels(unique_ids[tick_idx])

        fig.tight_layout()
        fig.savefig(out_dir / f"{head}_confusion_heatmap.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save confusion heatmap for head '{head}': {e}")

    return unique_ids, cm_norm, cm_counts

def plot_tsne(
    embeddings, labels, per_query_AP, save_path,
    draw_hulls=True, draw_labels=True
):
    """
    Creates a t-SNE plot using precomputed embeddings and labels.
    Draws convex hulls and annotates each class with its ID and average AP.
    
    Args:
        embeddings (np.ndarray): Embedding vectors (N x D).
        labels (np.ndarray): Class labels for each embedding.
        per_query_AP (list): Average precision per query (aligned with queries).
        save_path (str): Path to save the plot.
        draw_hulls (bool): Whether to draw convex hulls.
        draw_labels (bool): Whether to draw class IDs and AP.
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare plot
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)

    # Group points by class
    class_points = defaultdict(list)
    for point, label in zip(embeddings_2d, labels):
        class_points[int(label)].append(point)

    # Compute average AP per class (safe)
    unique_labels = sorted(set(int(l) for l in labels))
    class_ap_map = {}
    for label in unique_labels:
        aps = [ap for ap, lbl in zip(per_query_AP, labels[:len(per_query_AP)]) if lbl == label]
        class_ap_map[label] = np.mean(aps) if aps else 0.0  # fallback if no AP

    # Draw hulls and labels
    if draw_hulls or draw_labels:
        for label, points in class_points.items():
            points = np.array(points)
            hull_color = scatter.to_rgba(label)

            # Draw convex hull
            if draw_hulls and len(points) >= 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color=hull_color, alpha=0.8, linewidth=1.5)

            # Draw label with background
            if draw_labels:
                centroid = points.mean(axis=0)
                text_label = f"{label}\nAP:{class_ap_map.get(label, 0.0):.2f}"
                plt.text(
                    centroid[0], centroid[1], text_label,
                    color='black',
                    fontsize=9, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor=hull_color, alpha=0.2, edgecolor='none', boxstyle='round,pad=0.3')
                )

    plt.colorbar(scatter, label='Class ID')
    plt.title("t-SNE with Convex Hulls and Average AP per Class")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ t-SNE plot saved to {save_path}")

# --- Validation ---
def compute_map(similarity_matrix, labels):
    'Compute mean Average Precision (mAP) given similarity matrix and labels'
    aps = []
    per_query_AP = []
    for i in range(len(labels)):
        query_label = labels[i]
        sims = similarity_matrix[i]
        sorted_idx = np.argsort(-sims)
        sorted_idx = sorted_idx[sorted_idx != i]  # Remove self
        relevant = (labels[sorted_idx] == query_label).astype(int)
        if relevant.sum() == 0:
            per_query_AP.append(0.0)
            continue
        cum_prec = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (cum_prec * relevant).sum() / relevant.sum()
        aps.append(ap)
        per_query_AP.append(ap)
    return (np.mean(aps) if aps else 0.0), per_query_AP

def calculate_validation_performance(
    model, dataloader, save_folder, device='cuda',
    queries_per_sheet=15, matches_per_query=5, plot_ids="all",
    save_map_to_disk=None, processor=None,
    network: str = 'vit_ensamble_decoupled',
    heads=('Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2', 'fused')
):
    """
    Calculate validation performance (mAP) for all heads in the model.

    Returns (dicts for all heads present):
        mAP_dict:           Dict[str, float]
        embeddings_dict:    Dict[str, np.ndarray]   # per head: (N x D)
        labels_np:          np.ndarray              # (N,)
        per_query_AP_dict:  Dict[str, np.ndarray]   # per head: (N,)
    """
    assert processor is not None, "processor is required"
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    model.to(device)

    heads = list(heads)
    emb_buf = {h: [] for h in heads}
    labels_buf = []

    # Decide which heads we will plot per network
    if network == 'vit_ensamble_decoupled':
        heads_to_plot = [h for h in heads if not h.endswith('fused')]
    else:
        heads_to_plot = heads

    images_buf = {h: [] for h in heads_to_plot}  # no plots for 'fused'

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            # ---------- VIT DINO ----------
            if network == 'vit_dino' or network == 'vit_complete_img' or network == 'convnext_dino':
                if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                    continue
                g_orig, anchor_labels = batch

                # Processor -> pixel values
                inputs = processor(images=g_orig, return_tensors="pt")
                g_pixels = inputs["pixel_values"].to(device)

                # Forward pass
                g_emb = model(g_pixels)
                for h in heads:
                    emb_buf[h].append(g_emb.detach().cpu())

                # Labels
                labels_buf.extend(anchor_labels.detach().cpu().tolist())

                # Plot images: only global (Q1)
                if 'Q1' in images_buf:
                    images_buf['Q1'].extend([img for img in g_orig])
                if 'Q2' in images_buf:
                    images_buf['Q2'].extend([img for img in g_orig])
                if 'complete_img' in images_buf:
                    images_buf['complete_img'].extend([img for img in g_orig])
                if 'dorsal_fin' in images_buf:
                    images_buf['dorsal_fin'].extend([img for img in g_orig])
                if 'head' in images_buf:
                    images_buf['head'].extend([img for img in g_orig])
                if 'tail_fin' in images_buf:
                    images_buf['tail_fin'].extend([img for img in g_orig])
                if 'adi_fin' in images_buf:
                    images_buf['adi_fin'].extend([img for img in g_orig])

            # ---------- DECOUPLED ----------
            elif network == 'vit_ensamble_decoupled':
                vit_decoupled_q1 = 'Q1' in heads and len(heads) == 5
                vit_decoupled_q2 = 'Q2' in heads and len(heads) == 5

                # Expect (g, q1s0, q1s1, q1s2, labels)
                if not isinstance(batch, (tuple, list)):
                    continue

                #g, q1s0, q1s1, q1s2, anchor_labels = batch
                crops = batch[:-2]
                anchor_labels = batch[-1]
                ordered_heads = [h for h in ['Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2', 'Q2', 'Q2_s0', 'Q2_s1', 'Q2_s2'] if h in heads]
                for (crop, head_key) in zip(crops, ordered_heads): 
                    if head_key in images_buf:
                        images_buf[head_key].extend([img for img in crop])
                    if head_key in heads:
                        pix = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
                        emb = model.forward_single(pix, head_key=head_key)
                        emb_buf[head_key].append(emb.detach().cpu())
                

                if 'fused' in heads:
                    arg = [processor(images=crop, return_tensors="pt")["pixel_values"].to(device) for crop in crops]
                    if vit_decoupled_q1:
                        res = model(x_q1 = arg[0], x_q1s0=arg[1], x_q1s1=arg[2], x_q1s2=arg[3])
                        fused = res[-1]
                    elif vit_decoupled_q2:
                        res = model(x_q2 = arg[0], x_q2s0=arg[1], x_q2s1=arg[2], x_q2s2=arg[3])
                        fused = res[-1]

                    emb_buf['fused'].append(fused.detach().cpu())

                labels_buf.extend(anchor_labels.detach().cpu().tolist())

            else:
                raise ValueError(f"Unknown network type: {network}")

    if len(labels_buf) == 0:
        print("No valid samples found in dataloader.")
        return {}, {}, np.array([]), {}

    # Stack to numpy per head
    embeddings_dict = {}
    for h, parts in emb_buf.items():
        if len(parts) > 0:
            embeddings_dict[h] = torch.cat(parts, dim=0).numpy()

    labels_np = np.array(labels_buf)

    # Compute metrics per head
    mAP_dict = {}
    per_query_AP_dict = {}
    sim_mats = {}
    for head, E in embeddings_dict.items():
        S = cosine_similarity(E)
        mAP, per_query_AP = compute_map(S, labels_np)
        mAP_dict[head] = mAP
        per_query_AP_dict[head] = per_query_AP
        sim_mats[head] = S

    # Plot selection
    if plot_ids != "all":
        plot_indices = [i for i, lbl in enumerate(labels_np) if lbl in plot_ids]
    else:
        plot_indices = list(range(len(labels_np)))

    # Plot per-head based on network rule; never plot 'fused'
    if len(plot_indices) > 0:
        for head in heads_to_plot:
            if head not in embeddings_dict or head not in images_buf:
                continue
            save_sub = save_folder / Path('Results plot for ' + head)
            plot_top_matches(
                images_buf[head], labels_np, sim_mats[head], embeddings_dict[head],
                save_sub, queries_per_sheet, matches_per_query,
                per_query_AP_dict[head], plot_indices,
            )

    # Optional mAP dump
    if save_map_to_disk is not None:
        map_file = os.path.join(save_folder, save_map_to_disk + ".txt")
        with open(map_file, "w") as f:
            for head in mAP_dict.keys():
                f.write(f"{head}: {mAP_dict[head]:.4f}\n")

    # Console summary
    for head in mAP_dict.keys():
        print(f"[{network}] {head} mAP: {mAP_dict[head]:.4f}")

    return mAP_dict, embeddings_dict, labels_np, per_query_AP_dict, S

def evaluate_performance(
    model, test_loader_noaug, test_loader_aug, config, processor,
    epoch_metrics_file, difficulty, epoch, loss, device='cuda',
    eval_type="post", network='vit_ensamble_decoupled',
    heads=('Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2', 'fused')
):
    'Evaluate model performance and save plots and metrics.'

    if eval_type == 'pre':
        plot_path = config['analysis_path'] / Path('pre_plots')
    elif eval_type == 'post':
        plot_path = config['analysis_path'] / Path('post_plots')
    else:
        plot_path = config['analysis_path'] / Path('during_plots')

    os.makedirs(plot_path, exist_ok=True)
    model.eval()
    model.to(device)

    heads = list(heads)
    plot_ids = 'all' if eval_type in ['pre', 'post'] else []

    with open(epoch_metrics_file, "a") as f:
        f.write(f"{difficulty} Epoch {epoch}:\n")

        # --- RAW set ---
        mAP_raw, emb_raw, labels_raw, ap_raw, S = calculate_validation_performance(
            model=model,
            dataloader=test_loader_noaug,
            save_folder=plot_path,
            device=device,
            queries_per_sheet=15,
            matches_per_query=5,
            plot_ids=plot_ids,
            save_map_to_disk=None,
            processor=processor,
            network=network,
            heads=heads
        )
        for h in heads:
            if h in mAP_raw:
                f.write(f"raw_{h}: Loss={loss:.4f}, mAP={mAP_raw[h]:.4f}\n")

        
        # --- Pick a single head by priority, then do TSNE + Confusion Matrix ---
        priority = ['fused', 'Q1', 'Q2', 'complete_img', 'dorsal_fin', 'head', 'tail_fin', 'adi_fin']
        chosen = next((h for h in priority if h in emb_raw and labels_raw.size > 0), None)

        if chosen is not None:
            # TSNE
            plot_tsne(
                emb_raw[chosen], labels_raw, ap_raw[chosen],
                save_path=plot_path / Path(f"TSNE_{chosen}_epoch{epoch}_raw.png")
            )
            mAP_raw_ret = mAP_raw.get(chosen, float('nan'))
            conf_dir = plot_path / Path("confusion_matrices") / Path(f"CM_{chosen}_epoch{epoch}_raw.png")
            plot_confusion_matrices(S, labels_raw, conf_dir, chosen)
        else:
            mAP_raw_ret = float('nan')

        # --- AUGMENTED set ---
        mAP_aug, emb_aug, labels_aug, ap_aug, S = calculate_validation_performance(
            model=model,
            dataloader=test_loader_aug,
            save_folder=plot_path,
            device=device,
            queries_per_sheet=15,
            matches_per_query=5,
            plot_ids=[],  # no augmented plots (mirrors your logic)
            save_map_to_disk=None,
            processor=processor,
            network=network,
            heads=heads
        )

        for h in heads:
            if h in mAP_aug:
                f.write(f"aug_{h}: Loss={loss:.4f}, mAP={mAP_aug[h]:.4f}\n")

        # --- Pick a single head by priority, then do TSNE + Confusion Matrix ---
        priority = ['fused', 'Q1', 'Q2', 'complete_img', 'dorsal_fin', 'head', 'tail_fin', 'adi_fin']
        chosen = next((h for h in priority if h in emb_aug and labels_aug.size > 0), None)

        if chosen is not None:
            # TSNE
            plot_tsne(
                emb_aug[chosen], labels_aug, ap_aug[chosen],
                save_path=plot_path / Path(f"TSNE_{chosen}_epoch{epoch}_aug.png")
            )
            mAP_aug_ret = mAP_aug.get(chosen, float('nan'))
            conf_dir = plot_path / Path("confusion_matrices") / Path(f"CM_{chosen}_epoch{epoch}_aug.png")
            plot_confusion_matrices(S, labels_aug, conf_dir, chosen)
        else:
            mAP_aug_ret = float('nan')

    return mAP_raw_ret, mAP_aug_ret

# --- I/O ---
def save_ckpt(path, epoch, best_mAP, model, optimizer, scheduler):
    # Convert defaultdicts to normal dicts for pickling
    torch.save({
        'epoch': epoch,
        'best_mAP': best_mAP,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, device='cpu', embedding_dim=768):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Restore model weights
    model.load_state_dict(ckpt['model_state'])

    # Restore optimizer and scheduler if provided
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    return ckpt['epoch'], ckpt['best_mAP']
