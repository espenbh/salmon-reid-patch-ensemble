import yaml
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random
import os
import json

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, miners
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from training_helpers import *
from transformers import AutoModel, AutoImageProcessor
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning import miners, losses, distances
from hypll.manifolds.poincare_ball import PoincareBall, Curvature

config = {
'data_root': Path('/cluster/home/espebh/salmon_reid_ICIP/data/reid/'),
'training_folders': ['analysis1', 'analysis2', 'analysis3', 'analysis4', 'analysis5', 'analysis6', 'analysis7', 'analysis8', 'analysis9', 'analysis10', 'analysis11', 'analysis12', 'analysis13', 'analysis14'],
'test_folders': ['analysis15'],
'out_root': Path('/cluster/home/espebh/salmon_reid_ICIP/data/hyperbolic_regularization/'),
'run_name': 'ViT',
'batch_size': 128, 
'epochs': 20,
'curriculum_schedule': {1: 'easy', 5: 'medium', 7: 'hard'}, # 0-1: easiest, 1-4: easy ... 5 medium
'network': 'vit_ensamble_decoupled', # convnext_dino, vit_dino, vit_ensamble_decoupled, vit_ensamble_singleforward, vit_complete_img
'imgs_to_load': ['Q2', 'Q2_s0', 'Q2_s1', 'Q2_s2'], # input images
'heads': ['Q2', 'Q2_s0', 'Q2_s1', 'Q2_s2', 'fused'],   
#'network': 'vit_dino', # convnext_dino, vit_dino, vit_ensamble_decoupled, vit_complete_img
#'imgs_to_load': ['Q2_s0'], # input images
#'heads': ['Q2_s0'], 
    
'embedding_dim': 768, #1024, 768,
'losses': {'multisim': 1.0},#{'multisim': 1.0, 'smoothap': 1.0, 'arcface': 0.1}, # triplet, smoothap, arcface, multisim, proto, info_nce
'freeze': True,
'T0_gain': 1/16,
'T_mult': 2,
'lr': 1e-4,
'loss_weight_decay': 0.02,
'arcface_m': 0.3,
'arcface_s': 30,
'num_classes': 924, #915 #924
'crop_slices_to_mask': True,
'slice_split_idces': [0.3, 0.7],

}

ensamble_networks = ['vit_ensamble_decoupled']
single_patch_networks = ['vit_dino', 'convnext_dino', 'vit_complete_img']

config = create_analysis_path(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['device'] = device

# Model
model, loss_dict, params, processor = load_model(config, device)
loss_dict['multisim'] = None

# Training hyperparameters
optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['loss_weight_decay'])
train_loader = get_dataloader(config, difficulty='easiest', shuffle=True, loader_type='train', load_raw_images=True, num_IDs=8, imgs_to_load=config['imgs_to_load'], samples_per_ID=config['batch_size']//8)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_loader)*config['T0_gain']), T_mult=config['T_mult'])
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# Init training hyperparameters
batch_loss_file = os.path.join(config['analysis_path'], "batch_losses.txt")
epoch_metrics_file = os.path.join(config['analysis_path'], "epoch_metrics.txt")
lr_log_file = os.path.join(config['analysis_path'], "lr_log_file.txt")

# Testloaders
test_loader_noaug = get_dataloader(config, difficulty='easiest', shuffle=False, loader_type='test', imgs_to_load=config['imgs_to_load'], load_raw_images=True)
test_loader_aug = get_dataloader(config, difficulty='hard', shuffle=False, loader_type='test', imgs_to_load=config['imgs_to_load'], load_raw_images=True)

# Initialize
difficulty = 'easiest'
best_mAP = -1.0

# Pre training mAP
#_, _ = evaluate_performance(model, test_loader_noaug, test_loader_aug, config, processor, epoch_metrics_file, difficulty, -1, -1, device, eval_type='pre', network = config['network'], heads = config['heads'])

# training loop
for epoch in range(config['epochs']):
    if epoch in config['curriculum_schedule'].keys():
         difficulty = config['curriculum_schedule'][epoch]
    with open(batch_loss_file, "a") as f:
        f.write(f"{difficulty}\n")

    if difficulty == 'easiest':
        if 'multisim' in loss_dict.keys():
            miner = miners.MultiSimilarityMiner(epsilon=0.1)  # epsilon controls margin for mining
            loss_dict['multisim'] = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        num_IDs = 8

    elif difficulty == 'easy':
        if 'multisim' in loss_dict.keys():
            miner = miners.MultiSimilarityMiner(epsilon=0.08)
            loss_dict['multisim'] = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        num_IDs = 16

    elif difficulty == 'medium':
        if 'multisim' in loss_dict.keys():
            miner = miners.MultiSimilarityMiner(epsilon=0.07)
            loss_dict['multisim'] = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        num_IDs = 16

    elif difficulty == 'hard':
        if 'multisim' in loss_dict.keys():
            miner = miners.MultiSimilarityMiner(epsilon=0.05)
            loss_dict['multisim'] = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        num_IDs = 32

    if 'multisim' in loss_dict.keys(): loss_dict['multisim'].reducer = reducers.AvgNonZeroReducer()
    samples_per_ID = config['batch_size']//num_IDs # Batch size is always divisible by 32, and at least 64

    train_loader = get_dataloader(config, difficulty=difficulty, shuffle=True, loader_type='train', load_raw_images=True, num_IDs=num_IDs, imgs_to_load=config['imgs_to_load'], samples_per_ID=samples_per_ID)
    model.train()
    if 'arcface' in loss_dict.keys(): loss_dict['arcface'].train()

    total_loss = 0.0
    cnt = 0
    for batch in train_loader:
        if batch is None:
            with open(batch_loss_file, "a") as f:
                f.write(f"Skipping batch...\n")
            continue
        
        # ---format inputs
        vit_decoupled_q1 = config['network'] == 'vit_ensamble_decoupled' and 'Q1' in config['heads'] and len(config['heads']) == 5
        vit_decoupled_q2 = config['network'] == 'vit_ensamble_decoupled' and 'Q2' in config['heads'] and len(config['heads']) == 5
        vit_decoupled_q1q2 = config['network'] == 'vit_ensamble_decoupled' and 'Q1' in config['heads'] and 'Q2' in config['heads'] and len(config['heads']) == 11
        if vit_decoupled_q1:
            g, q1s0, q1s1, q1s2, labels = batch
            g, q1s0, q1s1, q1s2 = [processor(images=i, return_tensors="pt") for i in [g, q1s0, q1s1, q1s2]]
        elif vit_decoupled_q2:
            g, q2s0, q2s1, q2s2, labels = batch
            g, q2s0, q2s1, q2s2 = [processor(images=i, return_tensors="pt") for i in [g, q2s0, q2s1, q2s2]]
        elif vit_decoupled_q1q2:
            gq1, q1s0, q1s1, q1s2, gq2, q2s0, q2s1, q2s2, labels = batch
            gq1, q1s0, q1s1, q1s2, gq2, q2s0, q2s1, q2s2 = [processor(images=i, return_tensors="pt") for i in [gq1, q1s0, q1s1, q1s2, gq2, q2s0, q2s1, q2s2]]
        elif config['network'] == 'vit_ensamble_singleforward':
            g_orig, token_dict, labels = batch
            g = processor(images=g_orig, return_tensors="pt")
        elif config['network'] in single_patch_networks:
            g_orig, labels = batch
            g = processor(images=g_orig, return_tensors="pt")
        labels = labels.long().to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # ---Forward pass---
            if vit_decoupled_q1: euc_res = model(x_q1 = g["pixel_values"].to(device), x_q1s0 = q1s0["pixel_values"].to(device), x_q1s1 = q1s1["pixel_values"].to(device), x_q1s2 = q1s2["pixel_values"].to(device))
            elif vit_decoupled_q2: euc_res = model(x_q2 = g["pixel_values"].to(device), x_q2s0 = q2s0["pixel_values"].to(device), x_q2s1 = q2s1["pixel_values"].to(device), x_q2s2 = q2s2["pixel_values"].to(device))
            elif vit_decoupled_q1q2: euc_res = model(x_q1 = gq1["pixel_values"].to(device), x_q1s0 = q1s0["pixel_values"].to(device), x_q1s1 = q1s1["pixel_values"].to(device), x_q1s2 = q1s2["pixel_values"].to(device), x_q2 = gq2["pixel_values"].to(device), x_q2s0 = q2s0["pixel_values"].to(device), x_q2s1 = q2s1["pixel_values"].to(device), x_q2s2 = q2s2["pixel_values"].to(device))
            elif config['network'] in single_patch_networks: euc_res = model(g["pixel_values"].to(device))

            if isinstance(euc_res, (list, tuple)):
                euc_embeds = {k: v for k, v in zip(config['heads'], euc_res)}
            elif torch.is_tensor(euc_res):  # or isinstance(res, torch.Tensor)
                euc_embeds = {k: v for k, v in zip(config['heads'], [euc_res])}
            else:
                raise TypeError(f"Unexpected type for euc_res: {type(euc_res)}")
            
            miner_key = 'fused' if config['network'] in ensamble_networks else config['heads'][0]
            if 'triplet' in loss_dict.keys(): a_idx, p_idx, n_idx = miner(euc_embeds[miner_key], labels)
            elif 'multisim' in loss_dict.keys(): pairs = miner(euc_embeds[miner_key], labels)

            if vit_decoupled_q1q2:
                if 'multisim' in loss_dict.keys():
                    multisim_q1_pairs = miner(euc_embeds['Q1_fused'], labels)
                    multisim_q2_pairs = miner(euc_embeds['Q2_fused'], labels)

        # --------- Euclidean losses ---------
        temp_loss_logger = '######### cnt ' + str(cnt) + ' #########\n'
        combined_loss = torch.zeros((), device=device)

        for loss_name in loss_dict.keys():
            for key, emb in euc_embeds.items():
                if loss_name == 'triplet':
                    if a_idx.numel() == 0:
                        temp_loss_logger += f"{key}-Triplet: {0:.6f}, "
                        continue
                    part_loss = loss_dict['triplet'](emb, labels, (a_idx, p_idx, n_idx))
                    temp_loss_logger += f"{key}-Triplet: {part_loss.item() * config['losses']['triplet']:.6f}, "
                    combined_loss += part_loss * config['losses']['triplet']

                elif loss_name == 'smoothap':
                    part_loss = loss_dict['smoothap'](emb, labels)
                    temp_loss_logger += f"{key}-SmoothAP: {part_loss.item() * config['losses']['smoothap']:.6f}, "
                    combined_loss += part_loss * config['losses']['smoothap']

                elif loss_name == 'arcface':
                    part_loss = loss_dict['arcface'](emb, labels)
                    temp_loss_logger += f"{key}-Arcface: {part_loss.item() * config['losses']['arcface']:.6f}, "
                    combined_loss += part_loss * config['losses']['arcface']

                elif loss_name == 'multisim':
                    if vit_decoupled_q1q2:
                        if 'Q1' in key:
                            part_loss = loss_dict['multisim'](emb, labels, multisim_q1_pairs)
                            print(key)
                        elif 'Q2' in key: part_loss = loss_dict['multisim'](emb, labels, multisim_q2_pairs)
                        else: part_loss = loss_dict['multisim'](emb, labels, pairs)
                        temp_loss_logger += f"{key}-Multisim: {part_loss.item() * config['losses']['multisim']:.6f}, "
                        combined_loss += part_loss * config['losses']['multisim']
                    else:
                        part_loss = loss_dict['multisim'](emb, labels, pairs)
                        temp_loss_logger += f"{key}-Multisim: {part_loss.item() * config['losses']['multisim']:.6f}, "
                        combined_loss += part_loss * config['losses']['multisim']
            temp_loss_logger += '\n'
            
        # ---Update steps---
        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        print(combined_loss)
        
        total_loss += combined_loss.item()

        # ---Save batch loss---
        with open(batch_loss_file, "a") as f:
            f.write(temp_loss_logger + f"Total: {combined_loss.item():.6f}, " + "\n")

        # ---Save learning rate---
        with open(lr_log_file, "a") as f:
            f.write(f"{optimizer.param_groups[0]['lr']:.6f}\n")
        cnt = cnt + 1

    avg_loss = total_loss / max(cnt, 1)
    print(f"Epoch [{epoch}/{config['epochs']}] Loss: {avg_loss:.4f}")

    if epoch < config['epochs']-1:
        mAP_unaug, mAP_aug = evaluate_performance(model, test_loader_noaug, test_loader_aug, config, processor, epoch_metrics_file, difficulty, epoch, avg_loss, device, eval_type='during', network = config['network'], heads = config['heads'])
    
        # Save best
        if mAP_unaug > best_mAP:
            best_mAP = mAP_unaug
            save_ckpt(os.path.join(config['analysis_path'], "best_ckpt.pth"),
                    epoch, best_mAP, model, optimizer, scheduler)

    # Save last
    save_ckpt(os.path.join(config['analysis_path'], "last_ckpt.pth"),
            epoch, best_mAP, model, optimizer, scheduler)

# Post training mAP
_, _ = evaluate_performance(model, test_loader_noaug, test_loader_aug, config, processor, epoch_metrics_file, difficulty, epoch, avg_loss, device, eval_type='post', network = config['network'], heads = config['heads'])
