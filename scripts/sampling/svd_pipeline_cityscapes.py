import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from sgm.modules.diffusionmodules.sampling import SingleStepDiffusionSampler
from sgm.modules.diffusionmodules.sampling_utils import to_d
from sgm.util import append_dims, get_modulate_timestep_frames
import sys
sys.path.append('~/VideoSegCode/')
from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from scripts.sampling.feature_extraction import feature_extraction_main
from scripts.sampling.process_output import get_seg_map_main

from tqdm import tqdm
import argparse

import time


def sample(
    input_video_path: str,
    input_gt_path: str,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 14,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 17,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    save_all_features: bool = False,
    mask_source: str = "kmeans",
    modulate_block_idx: str = "8",
    modulate_timestep: str = "15",
    modulate_schedule: str = "constant",
    modulate_lambda_start: float = 100.0,
    modulate_lambda_end: float = 100.0,
    modulate_layer_type: str = "spatial,temporal",
    modulate_attn_type: str = "self_attn",
    num_masks: int = 10,
    modulate_timestep_frames_schedule: str = "constant",
    is_injected_features: bool = False,
    is_xt_mask: bool = False,
    is_global_ref: bool = False,
    feature_folder: str = "features_outputs_VSPW",
    sub_seq_start_idx: int = 0,
    delete_feature_maps: bool = False,
):

    
    def load_feature_masks(masks_path, mask_id, num_frames=14, feature_timestep="24", mask_source="kmeans", modulate_block_idx=8,
                           base_height=8, base_width=8, frame_name_list=None):
        sub_masks = []
        
        for frame_id in range(num_frames):
            if frame_name_list is not None:
                frame_name = frame_name_list[frame_id]
            else:
                frame_name = frame_id
            if mask_source == "kmeans":
                sub_dir_name = f"kmeans_time_{feature_timestep}_frame_{frame_name}"
                sub_dir_path = os.path.join(masks_path, sub_dir_name)
                sub_mask_path = os.path.join(sub_dir_path, f"mask_{mask_id}.png")
                sub_mask = Image.open(sub_mask_path)
            elif mask_source == "knn":
                sub_dir_name = f"knn_time_{feature_timestep}_frame_{frame_name}"
                sub_dir_path = os.path.join(masks_path, sub_dir_name)
                sub_mask_path = os.path.join(sub_dir_path, f"mask_{mask_id}.png")
                sub_mask = Image.open(sub_mask_path)
            elif mask_source == "sam":
                sub_dir_name = "sam_frame_{}".format(frame_name)
                sub_dir_path = os.path.join(masks_path, sub_dir_name)
                sub_mask_path = os.path.join(sub_dir_path, f"mask_{mask_id}.png")
                sub_mask = Image.open(sub_mask_path)
                sub_mask = sub_mask.resize((base_width * 2, base_height * 2), Image.LANCZOS)
                sub_mask = sub_mask.point(lambda p: p > 128 and 255)            
            width, height = sub_mask.size
            
            if modulate_block_idx in [0, 1, 2]:
                resize_height = base_height
                resize_width = base_width
            elif modulate_block_idx in [3, 4, 5]:
                resize_height = base_width * 2
                resize_width = base_height * 2
            elif modulate_block_idx in [6, 7, 8]:
                resize_height = base_height * 4
                resize_width = base_width * 4
            elif modulate_block_idx in [9, 10, 11]:
                resize_height = base_height * 8
                resize_width = base_width * 8
            
            sub_mask = sub_mask.resize((resize_width, resize_height))
            sub_mask = np.array(sub_mask) / 255.0
            sub_mask = rearrange(sub_mask, "h w -> (h w)")

            sub_mask = torch.from_numpy(sub_mask).to(device)  # dim: [hw] (1 dim tensor)
            sub_masks.append(sub_mask)
            
        return sub_masks
    
    def ddim_sampler_callback(xt, i):
        save_feature_maps_callback(i, xt=xt)
        # save_sampled_img(xt, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block", xt=None):
        if not global_ref_step:
            block_idx = 0

            for block in blocks:
                if not save_all_features and block_idx < 0:
                    block_idx += 1
                    continue

                if len(block) > 1 and "SpatialVideoTransformer" in str(type(block[1])):
                    save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_spatial_self_attn_k_time_{i}")
                    save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_spatial_self_attn_q_time_{i}")
                    save_feature_map(block[1].transformer_blocks[0].attn2.k, f"{feature_type}_{block_idx}_spatial_cross_attn_k_time_{i}")
                    save_feature_map(block[1].transformer_blocks[0].attn2.q, f"{feature_type}_{block_idx}_spatial_cross_attn_q_time_{i}")
                    save_feature_map(block[1].time_stack[0].attn1.k, f"{feature_type}_{block_idx}_temporal_self_attn_k_time_{i}")
                    save_feature_map(block[1].time_stack[0].attn1.q, f"{feature_type}_{block_idx}_temporal_self_attn_q_time_{i}")
                    save_feature_map(block[1].time_stack[0].attn2.k, f"{feature_type}_{block_idx}_temporal_cross_attn_k_time_{i}")
                    save_feature_map(block[1].time_stack[0].attn2.q, f"{feature_type}_{block_idx}_temporal_cross_attn_q_time_{i}")
                block_idx += 1

            if xt is not None:
                save_feature_map(xt, f"xt_time_{i}")
        else:
            block_idx = 8
            save_feature_map(blocks[block_idx][1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_spatial_self_attn_q_time_{i}_ref")


    def save_feature_maps_callback(i, xt=None):
        # save_feature_maps(model.model.diffusion_model.input_blocks, i, "input_block")
        if not global_ref_step:
            if i >= 14:
                save_feature_maps(model.model.diffusion_model.output_blocks , i, "output_block", xt=xt)
        else:
            if i == 24:
                save_feature_maps(model.model.diffusion_model.output_blocks , i, "output_block", xt=xt)

    def save_feature_map(feature_map, filename):
        if is_modulate_cross_attn:
            save_folder = os.path.join(feature_maps_path, f"mask_{mask_id}_lambda_{modulate_lambda_start_pn}")
        else:
            save_folder = feature_maps_path
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{filename}.pt")
        try:
            torch.save(feature_map, save_path)
        except:
            print(f"Failed to save feature map to {save_path}")
        del feature_map
    
    
    def sample_video(latent, is_modulate_cross_attn, modulate_params, t_start=14,
                     mask_id=None, is_save_video=False, callback_to_use=None, uc_list=None,
                     ori_h=None, ori_w=None, output_folder=None, frame_name_list=None, is_xt_mask=False,
                     feature_height=None, feature_width=None):
        samples_z = model.sampler(denoiser, latent.clone(), cond=c, uc=uc, img_callback=callback_to_use, 
                                          is_modulate=is_modulate_cross_attn, modulate_params=modulate_params,
                                          uc_list=uc_list, t_start=t_start, is_xt_mask=is_xt_mask,
                                          feature_height=feature_height, feature_width=feature_width)
        
        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        base_count = len(glob(os.path.join(output_folder, "*.mp4")))
        video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
        if is_modulate_cross_attn:
            modulate_lambda_start = modulate_params["modulate_lambda_start"]
            frames_path = os.path.join(output_folder, f"{base_count:06d}_l_{modulate_lambda_start}_mask_{mask_id}")
        else:
            frames_path = os.path.join(output_folder, f"{base_count:06d}")
        os.makedirs(frames_path, exist_ok=True)
        
        vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        
        for frame_id, frame in enumerate(vid):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if ori_h is not None and ori_w is not None:
                frame = cv2.resize(frame, (ori_w, ori_h))
            # save frame into an image
            if frame_name_list is not None:
                frame_name = frame_name_list[frame_id]
            else:
                frame_name = frame_id
            cv2.imwrite(os.path.join(frames_path, f"{frame_name}.png"), frame)
            
        if is_save_video:
            
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MP4V"),
                fps_id + 1,
                (samples.shape[-1], samples.shape[-2]),
            )

            for frame_id, frame in enumerate(vid):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()
            print("saved video to {}".format(video_path))
    
        return base_count
        
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    batch_size = 15
    video_frames_path = [f for f in os.listdir(input_video_path) if f.endswith(".png") or f.endswith(".jpg")]
    video_frames_path = sorted(video_frames_path, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))
    all_sub_seq_path = []
    sub_seq_path = []
    prev_seq_id = int(video_frames_path[0].split("_")[1])
    prev_frame_id = int(video_frames_path[0].split("_")[2])
    for video_frame_path in video_frames_path:
        seq_id = int(video_frame_path.split("_")[1])
        frame_id = int(video_frame_path.split("_")[2])
        if seq_id != prev_seq_id:
            all_sub_seq_path.append(sub_seq_path)
            sub_seq_path = []
            prev_seq_id = seq_id
            prev_frame_id = frame_id
            sub_seq_path.append(os.path.join(input_video_path, video_frame_path))
        else:
            if frame_id - prev_frame_id > 1:
                all_sub_seq_path.append(sub_seq_path)
                sub_seq_path = []
                prev_frame_id = frame_id
                sub_seq_path.append(os.path.join(input_video_path, video_frame_path))
            else:
                prev_frame_id = frame_id
                sub_seq_path.append(os.path.join(input_video_path, video_frame_path))
    
    all_sub_seq_path = all_sub_seq_path[sub_seq_start_idx:]
    
    
    t_start = int(modulate_timestep) - 1
    
    modulate_timestep = [int(timestep) for timestep in modulate_timestep.split(",") if timestep]
    modulate_block_idx = [int(idx) for idx in modulate_block_idx.split(",") if idx]
    modulate_layer_type = [layer_type for layer_type in modulate_layer_type.split(",") if layer_type]
    modulate_attn_type = [attn_type for attn_type in modulate_attn_type.split(",") if attn_type]
    modulate_timestep_frames = {}
    
    for sub_seq_id, sub_seq_path in enumerate(all_sub_seq_path):
        print(f"Processing sub sequence {sub_seq_id}: length {len(sub_seq_path)}")
        frame_img_list = []
        frame_name_list = []
        frame_gt_name_list = []
        frame_gt_img_list = []
        
        if len(sub_seq_path) < 14:
            sub_seq_path = sub_seq_path + [sub_seq_path[-1]] * (14 - len(sub_seq_path))  # duplicate the last frame to make up to 14 frames
            
        found_gt = False
        for frame_id, frame in enumerate(sub_seq_path):
            assert os.path.isfile(frame), f"Frame {frame} does not exist"
            frame_name = frame.split("/")[-1].split(".")[0]
            frame_img = Image.open(frame)
            if frame_img.mode == "RGBA":
                frame_img = frame_img.convert("RGB")
            ori_w, ori_h = frame_img.size

            width, height = map(lambda x: x - x % 64, (ori_w, ori_h))
            # resize the images from [2048, 1024] to [512, 256]
            # width = width // 4
            # height = height // 4
            # frame_img = frame_img.resize((width, height), Image.LANCZOS)
            # center crop the frame_img
            frame_img = frame_img.crop((width // 4, height // 4, width // 4 + 512, height // 4 + 256))
            frame_name_list.append(frame_name)
            frame_img_list.append(frame_img)
            
            if not found_gt and os.path.exists(os.path.join(input_gt_path, frame_name.replace("leftImg8bit", "gtFine_labelIds") + ".png")):
                gt_frame_name = frame_name
                gt_idx = frame_id
                found_gt = True
                
        gt_end_idx = min(gt_idx + num_frames, len(frame_img_list))
        gt_start_idx = max(0, gt_end_idx - batch_size)
        
        frame_gt_name_list = frame_name_list[gt_start_idx:gt_end_idx]
        frame_gt_img_list = frame_img_list[gt_start_idx:gt_end_idx]

        ref_mask = None
        ref_feature_map = None
        ref_unique_labels = None
        # process frame_img_list with batch_size
        num_batches = len(frame_img_list) // batch_size + 1
        if is_global_ref:
            global_ref_step = True
        else:
            global_ref_step = False
        
        batch_start_idx = -1
        
            
        for batch_id in range(batch_start_idx, num_batches):
            if batch_id == -1:
                if len(frame_gt_name_list) > 0:
                    frame_name_list_batch = frame_gt_name_list
                    frame_img_list_batch = frame_gt_img_list
                else:
                    continue
            else:
                start_idx = batch_id * batch_size
                end_idx = min((batch_id + 1) * batch_size, len(frame_img_list))
                if end_idx == len(frame_img_list):
                    start_idx = end_idx - num_frames
                    if start_idx < 0:
                        start_idx = 0
                print(f"start_idx: {start_idx}, end_idx: {end_idx}")
                frame_name_list_batch = frame_name_list[start_idx:end_idx]
                frame_img_list_batch = frame_img_list[start_idx:end_idx]
            
            for param in model.model.parameters():
                param.requires_grad = False
                
            seed_everything(seed)
                
            # convert list of images to tensor
            input_video_tensor = torch.stack([2.0 * ToTensor()(frame) - 1.0 for frame in frame_img_list_batch], axis=0) # [f c h w]
            input_video_tensor = input_video_tensor.to(device)  # [-1, 1]
            latent_video = model.encode_first_stage(input_video_tensor)
            
            # latent_video.shape: [f c h w]
            image = input_video_tensor[0].unsqueeze(0) # first frame as condition image, [1, c, h, w]
            
            H, W = image.shape[2:]
            assert image.shape[1] == 3            
            F = 8
            C = 4
            shape = (num_frames, C, H // F, W // F)
    
                
            value_dict = {}
            value_dict["motion_bucket_id"] = motion_bucket_id
            value_dict["fps_id"] = fps_id
            value_dict["cond_aug"] = cond_aug
            value_dict["cond_frames_without_noise"] = image
            value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
            value_dict["cond_aug"] = cond_aug

            with torch.no_grad():
                with torch.autocast(device):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=[
                            "cond_frames",
                            "cond_frames_without_noise",
                        ],
                    )

                    for k in ["crossattn", "concat"]:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)


                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(
                        2, num_frames
                    ).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c, is_modulate_step=False, is_injected_step=False, 
                                    modulate_params=None):
                        return model.denoiser(
                            model.model, input, sigma, c, 
                            is_modulate_step=is_modulate_step, 
                            is_injected_step=is_injected_step,
                            modulate_params=modulate_params,
                            **additional_model_inputs
                        )
                        
            with torch.no_grad():
                with torch.autocast(device):                
                    callback_to_use = ddim_sampler_callback
                    
                    # Step 1: add noise to the latent
                    latent = model.sampler.add_noise(latent_video, cond=c, uc=uc, num_steps=num_steps, noise_level=t_start)
                    # Step 2. Extract video features
                    is_modulate_cross_attn = False
                    
                    
                    feature_maps_path = os.path.join(feature_folder, f'{exp_name}/feature_maps')
                    os.makedirs(feature_maps_path, exist_ok=True)
                    
                    inversion_output_folder = os.path.join(feature_folder, f"{exp_name}/inversion_output")
        
                    sample_video(latent, False, None, mask_id=None, is_save_video=False, t_start=t_start,
                                        callback_to_use=ddim_sampler_callback, uc_list=None, 
                                            output_folder=inversion_output_folder, frame_name_list=frame_name_list_batch)
                    
                    if is_global_ref and global_ref_step:
                        global_ref_step = False
                        continue
                    
                    # Step 3. Extract low-res masks
                    from scripts.sampling.feature_extraction import match_concept_embed
                    concept_embed_path = "features_outputs/concept_embed/avg_embed_4.pt"
                    
                    # mode = "kmeans_masks"
                    # mode = "match_concept_embed"
                    mode = "match_gt_mask"
                    num_clusters = num_masks
                    block_name = "output_block_8"
                    experiment_name = exp_name
                    fit_experiments = exp_name
                    feature_types = "spatial_self_attn_q"
                    feature_height = H // (F * 2)
                    feature_width = W // (F * 2)
                    input_mask_path = input_video_path.replace("origin", "mask")
                    if batch_id == -1:
                        gt_mask_name = gt_frame_name.replace("leftImg8bit", "gtFine_labelIds")
                        gt_mask_path = os.path.join(input_gt_path, f"{gt_mask_name}.png")
                    else:
                        gt_mask_path = None
                    unique_labels, ref_mask, ref_feature_map = feature_extraction_main(mode, num_clusters, t_start, block_name, experiment_name, fit_experiments, feature_types, 
                                            feature_height, feature_width, selected_timestep, concept_embed_path=concept_embed_path,
                                            frame_name_list=frame_name_list_batch, base_folder=feature_folder, 
                                            ref_mask=ref_mask, ref_feature_map=ref_feature_map, ref_unique_labels=ref_unique_labels,
                                            gt_mask_path=gt_mask_path, is_global_ref=is_global_ref)
                    if batch_id == batch_start_idx:
                        ref_unique_labels = unique_labels
                    
                    print(f"per batch unique_labels: {unique_labels}")
                    feature_masks_folder = os.path.join(feature_folder, f"{exp_name}/{mode}/{block_name}_{feature_types}_masks_{num_masks}")
                    assert feature_masks_folder is not None, "feature_masks_folder should not be None"
                    assert os.path.isdir(feature_masks_folder), f"{feature_masks_folder} should be a folder"
                    assert len(modulate_block_idx) > 0, "modulate_block_idx should not be empty"
                    
                    
                    # Step 4. Modulation
                    is_modulate_cross_attn = True
                    
                    modulated_output_folder = os.path.join(feature_folder, f"{exp_name}/modulated_output")
                    os.makedirs(modulated_output_folder, exist_ok=True)
                    
                    if is_injected_features:
                        injected_block_types = ["output"]
                        injected_feature_types = [
                            # "spatial_cross_attn_k", "spatial_cross_attn_q", 
                            # "spatial_selfattn_k", "spatial_self_attn_q",
                            "temporal_cross_attn_k", "temporal_cross_attn_q", 
                            "temporal_self_attn_k", "temporal_self_attn_q"
                            # "temporal_cross_attn_v",
                            # "temporal_self_attn_v",
                        ]
                        input_block_indices = [3, 4, 5, 6, 7, 8, 10, 11]
                        output_block_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                        
                    else:
                        injected_block_types = None
                        injected_feature_types = None
                        input_block_indices = None
                        output_block_indices = None
                    
                        
                    for pn in range(2):
                        if pn % 2 == 0:
                            modulate_lambda_start_pn = modulate_lambda_start
                            modulate_lambda_end_pn = modulate_lambda_end
                        else:
                            modulate_lambda_start_pn = -modulate_lambda_start
                            modulate_lambda_end_pn = -modulate_lambda_end
                            
                        for mask_id in tqdm(ref_unique_labels, desc="modulate mask"):
                            
                            feature_masks = load_feature_masks(feature_masks_folder, mask_id, num_frames=num_frames,
                                                            feature_timestep=selected_timestep,
                                                                mask_source=mask_source, 
                                                                modulate_block_idx=modulate_block_idx[0],
                                                                base_height=H//(F * 8), 
                                                                base_width=W//(F * 8),
                                                                frame_name_list=frame_name_list_batch)
                            # if mask_id != 8:
                            #     modulate_layer_type_tmp = modulate_layer_type
                            # else:
                            #     modulate_layer_type_tmp = ["spatial"]
                            modulate_params = {
                                "feature_masks": feature_masks,
                                "modulate_block_idx": modulate_block_idx,
                                "modulate_layer_type": modulate_layer_type, # modulate_layer_type_tmp, 
                                "modulate_attn_type": modulate_attn_type,
                                "modulate_timestep": modulate_timestep,
                                "modulate_schedule": modulate_schedule,
                                "modulate_lambda_start": modulate_lambda_start_pn,
                                "modulate_lambda_end": modulate_lambda_end_pn,
                                "num_frames": num_frames,
                                "modulate_uc": True,
                                "is_injected_features": is_injected_features,
                                "injected_feature_types": injected_feature_types,
                                "injected_block_types": injected_block_types,
                                "input_block_indices": input_block_indices,
                                "output_block_indices": output_block_indices,
                                "feature_folder": feature_folder,
                                "exp_name": exp_name,
                                "injected_features_group": {},
                                # "modulate_block_frames": {3: [10, 11, 12, 13],5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
                                # "modulate_layer_frames": {"temporal": [7, 8, 9, 10, 11, 12, 13], "spatial": [0, 1, 2, 3, 4, 5, 6]},
                                "modulate_layer_frames": {},
                                "modulate_block_frames": {},
                                "modulate_timestep_frames": modulate_timestep_frames,
                                "modulate_lambda_layers": {"spatial": 10.0, "temporal": 10.0},
                            }
                            
                            base_count = sample_video(latent, is_modulate_cross_attn, modulate_params, t_start=t_start,
                                                mask_id=mask_id, is_save_video=False, output_folder=modulated_output_folder,
                                                    callback_to_use=None, uc_list=None, ori_h=ori_h, ori_w=ori_w, frame_name_list=frame_name_list_batch,
                                                    is_xt_mask=is_xt_mask, feature_height=feature_height, feature_width=feature_width)
    
                    # Step 5: generate masks
                    get_seg_map_main(exp_name, base_count, modulate_lambda_start, num_masks, num_frames, filter_difference=False, filter_s=1.0,
                            resize_height=(H // (F * 2)), resize_width=(H // (F * 2)), unique_labels=ref_unique_labels, base_folder=feature_folder, 
                            frame_name_list=frame_name_list_batch, mask_folder=feature_masks_folder, feature_timestep=selected_timestep)  
                    get_seg_map_main(exp_name, base_count, modulate_lambda_start, num_masks, num_frames, filter_difference=True, filter_s=0.9,
                            resize_height=(H // (F * 2)), resize_width=(H // (F * 2)), unique_labels=ref_unique_labels, base_folder=feature_folder,
                            frame_name_list=frame_name_list_batch, mask_folder=feature_masks_folder, feature_timestep=selected_timestep)         
                    get_seg_map_main(exp_name, base_count, modulate_lambda_start, num_masks, num_frames, filter_difference=True, filter_s=0.7,
                            resize_height=(H // (F * 2)), resize_width=(H // (F * 2)), unique_labels=ref_unique_labels, base_folder=feature_folder,
                            frame_name_list=frame_name_list_batch, mask_folder=feature_masks_folder, feature_timestep=selected_timestep) 
                    
                    

        del latent, input_video_tensor, latent_video
        
        if delete_feature_maps:
            files_to_delete = os.listdir(feature_maps_path)
            for file in files_to_delete:
                os.remove(os.path.join(feature_maps_path, file))        
    
def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
        
    filter = DeepFloydDataFiltering(verbose=False, device=device)
    
    return model, filter
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 


if __name__ == "__main__":
    # Fire(sample)
    # change it to argparser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=str, default="dataset/cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/val", help="path to the input dataset")
    parser.add_argument("--split_file_path", type=str, default=None, help="path to the split file")
    parser.add_argument("--dataset_gt_path", type=str, default="dataset/cityscapes/gtFine/val", help="path to the mask file")
    parser.add_argument("--num_steps", type=int, default=25, help="number of steps")
    parser.add_argument("--num_frames", type=int, default=14, help="number of frames")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--seed", type=int, default=1, help="seed for sampling")
    parser.add_argument("--motion_bucket_id", type=int, default=127, help="motion bucket id")
    parser.add_argument("--cond_aug", type=float, default=0.02, help="condition augmentation")
    parser.add_argument("--save_all_features", default=False, action="store_true", help="whether to save all features")
    parser.add_argument("--mask_source", type=str, default="kmeans", help="mask source")
    parser.add_argument("--modulate_block_idx", type=str, default="8", help="selected block idx")
    parser.add_argument("--modulate_timestep", type=str, default="15", help="selected modulate timestep")
    parser.add_argument("--modulate_schedule", type=str, default="constant", help="modulate lambda schedule")
    parser.add_argument("--modulate_lambda_start", type=float, default=100.0, help="modulate lambda start")
    parser.add_argument("--modulate_lambda_end", type=float, default=100.0, help="modulate lambda end")
    parser.add_argument("--num_masks", type=int, default=20, help="number of masks to use")
    parser.add_argument("--is_injected_features", default=False, action="store_true", help="whether to use injected features")
    parser.add_argument("--modulate_layer_type", type=str, default="spatial,temporal", help="modulate layer type")
    parser.add_argument("--modulate_attn_type", type=str, default="self_attn", help="modulate attention type")
    parser.add_argument("--modulate_timestep_frames_schedule", type=str, default="constant", help="modulate timestep frames schedule")
    parser.add_argument("--is_global_ref", default=False, action="store_true", help="whether to use global reference sequence")
    parser.add_argument("--feature_folder", type=str, default="features_outputs_VSPW", help="feature folder path")
    parser.add_argument("--exp_start_idx", type=int, default=0, help="experiment start index")
    parser.add_argument("--num_exp", type=int, default=100, help="number of experiments to run")
    parser.add_argument("--sub_seq_start_idx", type=int, default=0, help="sub sequence start index")
    parser.add_argument("--delete_feature_maps", default=False, action="store_true", help="whether to delete feature maps to save space")
    
    args = parser.parse_args()
    
    num_frames = default(args.num_frames, 14)
    num_steps = default(args.num_steps, 25)
    model_config = "scripts/sampling/configs/svd.yaml"
    device = args.device
    
    model, filter = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
        )
    model.en_and_decode_n_samples_a_time = 1
    
    dataset_path = args.dataset_path
    dataset_gt_path = args.dataset_gt_path
    exp_name_list = os.listdir(dataset_path)
    
    if args.exp_start_idx + args.num_exp > len(exp_name_list):
        args.num_exp = len(exp_name_list) - args.exp_start_idx
    exp_name_list = exp_name_list[args.exp_start_idx:args.exp_start_idx + args.num_exp]
    
    print("We start from exp:", exp_name_list[0])

    for exp_name in tqdm(exp_name_list, desc="num_exps"):
        input_video_path = os.path.join(dataset_path, exp_name)
        input_gt_path = os.path.join(dataset_gt_path, exp_name)
        try:
            sample(
                input_video_path=input_video_path,
                input_gt_path=input_gt_path,
                seed=args.seed,
                num_frames=num_frames,
                num_steps=num_steps,
                motion_bucket_id=args.motion_bucket_id,
                cond_aug=args.cond_aug,
                save_all_features=args.save_all_features,
                mask_source=args.mask_source,
                modulate_block_idx=args.modulate_block_idx,
                modulate_timestep=args.modulate_timestep,
                modulate_schedule=args.modulate_schedule,
                modulate_lambda_start=args.modulate_lambda_start,
                modulate_lambda_end=args.modulate_lambda_end,
                modulate_layer_type=args.modulate_layer_type,
                modulate_attn_type=args.modulate_attn_type,
                modulate_timestep_frames_schedule=args.modulate_timestep_frames_schedule,
                is_injected_features=args.is_injected_features,
                num_masks=args.num_masks,
                is_xt_mask=True,
                is_global_ref=args.is_global_ref,
                feature_folder=args.feature_folder,
                sub_seq_start_idx=args.sub_seq_start_idx,
                delete_feature_maps=args.delete_feature_maps,)
        except Exception as e:
            print(f"Failed to sample video {exp_name}: {e}")
            continue