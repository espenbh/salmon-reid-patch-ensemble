import torch
import json
from pathlib import Path
from training_helpers import *
from test_helpers import *
import os

# analysis15_loader: queries
# analysis16_loader: gallery
# Define test run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_path = Path('/cluster/home/espebh/salmon_reid_ICIP/data/trained_networks/')
key_folder_map = {'Q1': 'analysis3', 'Q2': 'analysis4', 'head': 'analysis7', 'dorsal_fin': 'analysis8', 'complete_img': 'analysis2'}

# Build configs
parts_cfg = {}
for ensamble_key, analysis_folder in key_folder_map.items():
    config = json.load(open(network_path / analysis_folder / 'config.json'))
    if ensamble_key.startswith('Q1'):
        config_joint = config.copy()

    config['data_root'] = Path(config['data_root'])
    config['out_root'] = Path(config['out_root'])
    config['device'] = str(device)

    model, _, _, processor = load_model(config, device=device)
    model_checkpoint = torch.load(network_path / analysis_folder / 'best_ckpt.pth', map_location=device, weights_only=False)
    model.load_state_dict(model_checkpoint['model_state'])
    model.eval()
    parts_cfg[ensamble_key] = {'model': model, 'processor': processor}
    
config_joint["network"] = "test_ensamble_vit" 
config_joint['imgs_to_load'] = ['Q1', 'Q1_s0', 'Q1_s1', 'Q1_s2', 'Q2', 'Q2_s0', 'Q2_s1', 'Q2_s2', 'head', 'dorsal_fin', 'complete_img']
config_joint["data_root"] = Path(config_joint["data_root"])
config_joint["out_root"]  = Path(config_joint['out_root'])
config_joint["device"]    = str(device)

query_config = config_joint.copy()
gallery_config = config_joint.copy()
gallery_config['test_folders'] = ['analysis16']

# Data loaders
query_loader = get_dataloader(query_config, difficulty='easiest', shuffle=False, loader_type='test', imgs_to_load=query_config['imgs_to_load'], load_raw_images=True)
gallery_loader = get_dataloader(gallery_config, difficulty='easiest', shuffle=False, loader_type='test', imgs_to_load=gallery_config['imgs_to_load'], load_raw_images=True)

query_files, query_ids = get_file_names_and_labels(config['data_root'], ['analysis15'])
gallery_files, gallery_ids = get_file_names_and_labels(config['data_root'], ['analysis16'])
match_files_and_ids = {'query_files': query_files, 'query_ids': query_ids, 'gallery_files': gallery_files, 'gallery_ids': gallery_ids}

calculate_cross_q1_and_ensemble_plots_fused(
    parts_cfg,                    # dict: 'Q1','Q2','head','dorsal_fin' -> {'model':..., 'processor':...}
    query_loader,                 # 12 slots: (g_q1,q1s0,q1s1,q1s2, g_q2,q2s0,q2s1,q2s2, head, dorsal, complete, labels)
    gallery_loader,               # same 12 slots from anoidmatch_xlsx_pathther folder
    config['out_root'] / Path('test_single_convnext_with_val_l075'),
    device=device,
    queries_per_sheet=15, matches_per_query=6,
    plot_q_indices=None,          # optional subset of query indices to visualize
    lambda_mix=0.75, rrf_k=20,
    group_ids=True, strict_global=False,
    idmatch_xlsx_path = '/cluster/home/espebh/salmon_reid_nov25/data/reid/a15_a16_idmatch_with_traj_IDs.xlsx',
    match_files_and_ids = match_files_and_ids,
    quarter_models = 'single',
    holdout_patches = [],
    plot_images = True,
    )

# Run test
#os.makedirs(config['out_root'] / Path('lambda_ablation'), exist_ok = True)
#for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
#    calculate_cross_q1_and_ensemble_plots_fused(
#        parts_cfg,                    # dict: 'Q1','Q2','head','dorsal_fin' -> {'model':..., 'processor':...}
#        query_loader,                 # 12 slots: (g_q1,q1s0,q1s1,q1s2, g_q2,q2s0,q2s1,q2s2, head, dorsal, complete, labels)
#        gallery_loader,               # same 12 slots from anoidmatch_xlsx_pathther folder
#        config['out_root'] / Path('lambda_ablation') / Path('test_results_lambda' + str(i)),
#        device=device,
#        queries_per_sheet=15, matches_per_query=6,
#        plot_q_indices=None,          # optional subset of query indices to visualize
#        lambda_mix=i, rrf_k=20,
#        group_ids=True, strict_global=False,
#        idmatch_xlsx_path = '/cluster/home/espebh/salmon_reid_nov25/data/reid/a15_a16_idmatch_with_traj_IDs.xlsx',
#        match_files_and_ids = match_files_and_ids,
#        quarter_models = 'sliced',
#        holdout_patches = [],
#        plot_images = False,
#    )

#os.makedirs(config['out_root'] / Path('holdout_ablation'), exist_ok = True)
#for i in ['head', 'Q1', 'Q2', 'dorsal_fin', '']:
#    calculate_cross_q1_and_ensemble_plots_fused(
#        parts_cfg,                    # dict: 'Q1','Q2','head','dorsal_fin' -> {'model':..., 'processor':...}
#        query_loader,                 # 12 slots: (g_q1,q1s0,q1s1,q1s2, g_q2,q2s0,q2s1,q2s2, head, dorsal, complete, labels)
#        gallery_loader,               # same 12 slots from anoidmatch_xlsx_pathther folder
#        config['out_root'] / Path('holdout_ablation') / Path('test_results_holdout_' + str(i)),
#        device=device,
#        queries_per_sheet=15, matches_per_query=6,
#        plot_q_indices=None,          # optional subset of query indices to visualize
#        lambda_mix=0.75, rrf_k=20,
#        group_ids=True, strict_global=False,
#        idmatch_xlsx_path = '/cluster/home/espebh/salmon_reid_nov25/data/reid/a15_a16_idmatch_with_traj_IDs.xlsx',
#        match_files_and_ids = match_files_and_ids,
#        quarter_models = 'sliced',
#        holdout_patches = [i],
#        plot_images = False,
#    )

#os.makedirs(config['out_root'] / Path('tau_ablation'), exist_ok = True)
#for i in [0.2, 0.5, 0.7, 1.0, 2.0, 5.0]:
#    calculate_cross_q1_and_ensemble_plots_fused(
#        parts_cfg,                    # dict: 'Q1','Q2','head','dorsal_fin' -> {'model':..., 'processor':...}
#        query_loader,                 # 12 slots: (g_q1,q1s0,q1s1,q1s2, g_q2,q2s0,q2s1,q2s2, head, dorsal, complete, labels)
#        gallery_loader,               # same 12 slots from anoidmatch_xlsx_pathther folder
#        config['out_root'] / Path('tau_ablation') / Path('test_results_tau_' + str(i)),
#        device=device,
#        queries_per_sheet=15, matches_per_query=6,
#        plot_q_indices=None,          # optional subset of query indices to visualize
#        lambda_mix=0.75, rrf_k=20,
#        tau_by_part = {k:i for k in key_folder_map.keys()},
#        group_ids=True, strict_global=False,
#        idmatch_xlsx_path = '/cluster/home/espebh/salmon_reid_nov25/data/reid/a15_a16_idmatch_with_traj_IDs.xlsx',
#        match_files_and_ids = match_files_and_ids,
#        quarter_models = 'sliced',
#        holdout_patches = [],
#        plot_images = False,
#    )
    
#os.makedirs(config['out_root'] / Path('k_ablation'), exist_ok = True)
#for i in [1, 10, 20, 30, 60, 100, 150, 200, 300, 500]:#[150, 200, 300, 500]:
#    calculate_cross_q1_and_ensemble_plots_fused(
#        parts_cfg,                    # dict: 'Q1','Q2','head','dorsal_fin' -> {'model':..., 'processor':...}
#        query_loader,                 # 12 slots: (g_q1,q1s0,q1s1,q1s2, g_q2,q2s0,q2s1,q2s2, head, dorsal, complete, labels)
#        gallery_loader,               # same 12 slots from anoidmatch_xlsx_pathther folder
#        config['out_root'] / Path('k_ablation') / Path('test_results_k_' + str(i)),
#        device=device,
#        queries_per_sheet=15, matches_per_query=6,
#        plot_q_indices=None,          # optional subset of query indices to visualize
#        lambda_mix=0.75, rrf_k=i,
#        group_ids=True, strict_global=False,
#        idmatch_xlsx_path = '/cluster/home/espebh/salmon_reid_nov25/data/reid/a15_a16_idmatch_with_traj_IDs.xlsx',
#        match_files_and_ids = match_files_and_ids,
#        quarter_models = 'sliced',
#        holdout_patches = [],
#        plot_images = False,
#    )