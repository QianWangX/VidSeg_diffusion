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
from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
    
from scripts.sampling.feature_extraction import feature_extraction_main
from scripts.sampling.process_output import get_seg_map_main

from tqdm import tqdm
import argparse


def sample(
    input_image_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    input_video_path: Optional[str] = None,
    exp_name: str = "default",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 14,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 17,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    save_all_features: bool = False,
    modulate_block_idx: str = "8",
    modulate_timestep: str = "17",
    feature_timestep: str = "24",
    modulate_schedule: str = "constant",
    modulate_lambda_start: float = 50.0,
    modulate_lambda_end: float = 50.0,
    modulate_layer_type: str = "spatial,temporal",
    modulate_attn_type: str = "self_attn",
    num_masks: int = 10,
    modulate_timestep_frames_schedule: str = "constant",
    gt_mask_path: Optional[str] = None,
    feature_extraction_mode: str = "kmeans_masks",
    is_injected_features: bool = False,
    is_latent_blending: bool = False,
    feature_folder: str = "features_outputs",
    inversion_type: str = "add_noise",
    is_refine_mask: bool = False,
    is_aggre_attn: bool = False,
):

    
    def load_feature_masks(masks_path, mask_id, num_frames=14, feature_timestep="24", modulate_block_idx=8,
                           base_height=8, base_width=8, frame_name_list=None):
        sub_masks = []
        
        for frame_id in range(num_frames):
            if frame_name_list is not None:
                frame_name = frame_name_list[frame_id]
            else:
                frame_name = frame_id
                
                
            sub_dir_name = f"kmeans_time_{feature_timestep}_frame_{frame_name}"
            sub_dir_path = os.path.join(masks_path, sub_dir_name)
            sub_mask_path = os.path.join(sub_dir_path, f"mask_{mask_id}.png")
            sub_mask = Image.open(sub_mask_path)         
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
    
    feature_maps_path = os.path.join(feature_folder, f'{exp_name}/feature_maps')
    os.makedirs(feature_maps_path, exist_ok=True)
    
    def ddim_sampler_callback(xt, i):
            save_feature_maps_callback(i, xt=xt)
        # save_sampled_img(xt, i, predicted_samples_path)

    def save_feature_maps(blocks, i, feature_type="input_block", xt=None):
        block_idx = 0
        for block in blocks:

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


    def save_feature_maps_callback(i, xt=None):
        if i >= t_start:
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
                     ori_h=None, ori_w=None, output_folder=None, frame_name_list=None, is_latent_blending=False,
                     feature_height=None, feature_width=None):
        samples_z = model.sampler(denoiser, latent.clone(), cond=c, uc=uc, img_callback=callback_to_use, 
                                          is_modulate=is_modulate_cross_attn, modulate_params=modulate_params,
                                          uc_list=uc_list, t_start=t_start, is_latent_blending=is_latent_blending,
                                          feature_height=feature_height, feature_width=feature_width, 
                                          model=model)
        
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

    num_frames = default(num_frames, 14)
    num_steps = default(num_steps, 25)
    output_folder = default(output_folder, "outputs/modulate_video_sample/svd/{}".format(exp_name))
    model_config = "configs/inference/svd.yaml"

    os.makedirs(output_folder, exist_ok=True)
    
    if feature_extraction_mode == "match_gt_mask":
        assert gt_mask_path is not None, "gt_mask_path should not be None when mode is match_gt_mask"


    print("Input video is provided. Input image will be ignored. \
            Num_frames will be overridden by the number of frames in the video.")
    video_frames_path = [f for f in os.listdir(input_video_path) if f.endswith(".png") or f.endswith(".jpg")]
    video_frames_path = sorted(video_frames_path, key=lambda x: int(x.split(".")[0]))
    video_frames_path = [os.path.join(input_video_path, f) for f in video_frames_path]
    
    frame_img_list = []
    frame_name_list = []
    for frame_id, frame in enumerate(video_frames_path):
        assert os.path.isfile(frame), f"Frame {frame} does not exist"
        frame_name = frame.split("/")[-1].split(".")[0]
        frame_img = Image.open(frame)
        if frame_img.mode == "RGBA":
            frame_img = frame_img.convert("RGB")
        ori_w, ori_h = frame_img.size
        if frame_id == 0:
            print(f"Image size: {ori_h}x{ori_w}")

        if ori_h % 64 != 0 or ori_w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (ori_w, ori_h))
            frame_img = frame_img.resize((width, height))

        frame_img_list.append(frame_img)
        frame_name_list.append(frame_name)
        
    num_frames = len(frame_img_list)
    
    ref_mask = None
    ref_feature_map = None
    ref_unique_labels = None
    
    if inversion_type == "add_noise":
        t_start = int(modulate_timestep)
    elif inversion_type == "inversion":
        t_start = 0
    
    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    model.en_and_decode_n_samples_a_time = decoding_t
    
    for param in model.model.parameters():
        param.requires_grad = False
        
    seed_everything(seed)
    
    # convert list of images to tensor
    input_video_tensor = torch.stack([2.0 * ToTensor()(frame) - 1.0 for frame in frame_img_list], axis=0) # [f c h w]
    input_video_tensor = input_video_tensor.to(device)  # [-1, 1]
    latent_video = model.encode_first_stage(input_video_tensor)
    
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

            latent = torch.randn(shape, device=device)

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
            # Step 1: add noise to the latent
            if inversion_type == "add_noise":
                latent = model.sampler.add_noise(latent_video, cond=c, uc=uc, num_steps=num_steps, noise_level=t_start)
            elif inversion_type == "inversion":
                latent, _ = model.sampler.inversion(denoiser, latent_video, cond=c, uc=uc, num_steps=num_steps)
            else:
                raise ValueError(f"Invalid inversion type {inversion_type}")
            # Step 2. Extract video features
            is_modulate_cross_attn = False
            
            uc_list = None           
                
            feature_maps_path = os.path.join(feature_folder, f'{exp_name}/feature_maps')
            os.makedirs(feature_maps_path, exist_ok=True)
            
            inversion_output_folder = os.path.join(feature_folder, f"{exp_name}/inversion_output")

            sample_video(latent, False, None, mask_id=None, is_save_video=False, t_start=t_start,
                                    callback_to_use=ddim_sampler_callback, uc_list=None, 
                                    output_folder=inversion_output_folder, frame_name_list=frame_name_list)
            
            # Step 3. Extract low-res masks

            num_clusters = num_masks
            if is_aggre_attn:
                block_name = "output_block_8,output_block_7,output_block_6"
            else:
                block_name = "output_block_8"
            experiment_name = exp_name
            fit_experiments = exp_name
            feature_types = "spatial_self_attn_q"
            feature_height = H // (F * 2)
            feature_width = W // (F * 2)
            
            
            unique_labels, ref_mask, ref_feature_map = feature_extraction_main(feature_extraction_mode, num_clusters, t_start, block_name, experiment_name, fit_experiments, feature_types, 
                                    feature_height, feature_width, feature_timestep,
                                    frame_name_list=frame_name_list, base_folder=feature_folder, 
                                    ref_mask=ref_mask, ref_feature_map=ref_feature_map, ref_unique_labels=ref_unique_labels,
                                    gt_mask_path=gt_mask_path)

            ref_unique_labels = unique_labels
                
            print(f"per batch unique_labels: {unique_labels}")
            block_name_list = block_name.split(',')
            if len(block_name_list) == 1:
                block_name_str = block_name_list[0]
            else:
                block_name_str = '_'.join(block_name_list)
            feature_masks_folder = os.path.join(feature_folder, f"{exp_name}/{feature_extraction_mode}/{block_name_str}_{feature_types}_masks_{num_masks}")
            assert feature_masks_folder is not None, "feature_masks_folder should not be None"
            assert os.path.isdir(feature_masks_folder), f"{feature_masks_folder} should be a folder"
            assert len(modulate_block_idx) > 0, "modulate_block_idx should not be empty"
            
            if is_refine_mask:
                correct_mode = "correct_low_res_mask"
                correct_block_name = "output_block_7"
                _, ref_mask, _ = feature_extraction_main(correct_mode, num_clusters, t_start, correct_block_name, experiment_name, fit_experiments, feature_types, 
                                        feature_height, feature_width, feature_timestep, 
                                        frame_name_list=frame_name_list, base_folder=feature_folder, num_frames=num_frames,
                                        ref_mask=ref_mask, ref_feature_map=ref_feature_map, ref_unique_labels=ref_unique_labels,
                                        gt_mask_path=gt_mask_path, mask_folder=feature_masks_folder)
                
                feature_masks_folder = os.path.join(feature_folder, f"{exp_name}/{feature_extraction_mode}/{block_name_str}_{feature_types}_masks_{num_masks}_corrected")
                assert feature_masks_folder is not None, "feature_masks_folder should not be None"
                assert os.path.isdir(feature_masks_folder), f"{feature_masks_folder} should be a folder"

            # Step 4. Modulation
            is_modulate_cross_attn = True
            
            modulated_output_folder = os.path.join(feature_folder, f"{exp_name}/modulated_output")
            os.makedirs(modulated_output_folder, exist_ok=True)
            
            if is_modulate_cross_attn and is_injected_features:
                injected_block_types = ["output"]
                injected_feature_types = [
                    "temporal_cross_attn_k", "temporal_cross_attn_q", 
                        "temporal_self_attn_k", "temporal_self_attn_q"
                ]
                input_block_indices = [1, 2, 4, 5, 7, 8, 10, 11]
                output_block_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                
            else:
                injected_block_types = None
                injected_feature_types = None
                input_block_indices = None
                output_block_indices = None
                
            if input_video_path is None:
                t_start = 0
            else:
                t_start = 14
                
            
            if is_modulate_cross_attn:
                modulate_block_idx = [int(idx) for idx in modulate_block_idx.split(",") if idx]
                modulate_layer_type = [layer_type for layer_type in modulate_layer_type.split(",") if layer_type]
                modulate_attn_type = [attn_type for attn_type in modulate_attn_type.split(",") if attn_type]
                modulate_timestep_frames = get_modulate_timestep_frames(start_timestep=20, end_timestep=15,
                                                                        num_frames=num_frames, schedule=modulate_timestep_frames_schedule)
                modulate_timestep = [int(timestep) for timestep in modulate_timestep.split(",") if timestep]
                for pn in range(2):
                    if pn % 2 == 0:
                        modulate_lambda_start_pn = modulate_lambda_start
                        modulate_lambda_end_pn = modulate_lambda_end
                    else:
                        modulate_lambda_start_pn = -modulate_lambda_start
                        modulate_lambda_end_pn = -modulate_lambda_end
                        
                    for mask_id in tqdm(range(num_masks), desc="modulate mask"):
                        assert feature_masks_folder is not None, "feature_masks_folder should not be None"
                        assert os.path.isdir(feature_masks_folder), "feature_masks_folder should be a folder"
                        assert len(modulate_block_idx) > 0, "modulate_block_idx should not be empty"
                        
                        feature_masks = load_feature_masks(feature_masks_folder, mask_id, num_frames=num_frames,
                                                           feature_timestep=feature_timestep,
                                                            modulate_block_idx=modulate_block_idx[0],
                                                            base_height=H//(F * 8), 
                                                            base_width=W//(F * 8),
                                                            frame_name_list=frame_name_list,)
                        
                        modulate_params = {
                            "feature_masks": feature_masks,
                            "modulate_block_idx": modulate_block_idx,
                            "modulate_layer_type": modulate_layer_type, 
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
                            "modulate_layer_frames": {},
                            "modulate_block_frames": {},
                            "modulate_timestep_frames": modulate_timestep_frames,
                            "modulate_lambda_layers": {},
                            "latent_mask_start": min(modulate_timestep),
                            "latent_mask_end": num_steps,
                        }
                        if modulate_block_idx[0] in [0, 1, 2]:
                            modulate_height = H // (F * 8)
                            modulate_width = W // (F * 8)
                        elif modulate_block_idx[0] in [3, 4, 5]:
                            modulate_height = H // (F * 4)
                            modulate_width = W // (F * 4)
                        elif modulate_block_idx[0] in [6, 7, 8]:
                            modulate_height = H // (F * 2)
                            modulate_width = W // (F * 2)
                        elif modulate_block_idx[0] in [9, 10, 11]:
                            modulate_height = H // F
                            modulate_width = W // F
                        
                        base_count = sample_video(latent, is_modulate_cross_attn, modulate_params, t_start=t_start,
                                               mask_id=mask_id, is_save_video=False, output_folder=modulated_output_folder,
                                                callback_to_use=None, uc_list=None, ori_h=ori_h, ori_w=ori_w, frame_name_list=frame_name_list,
                                                is_latent_blending=is_latent_blending, feature_height=modulate_height, feature_width=modulate_width)
                base_count = 0
                # Step 5: generate masks
                get_seg_map_main(exp_name, base_count, modulate_lambda_start, num_masks, num_frames, filter_difference=False, filter_s=1.0,
                        resize_height=(H // (F * 2)), resize_width=(H // (F * 2)), unique_labels=ref_unique_labels, base_folder=feature_folder, 
                        frame_name_list=frame_name_list, mask_folder=feature_masks_folder, feature_timestep=feature_timestep)  
                get_seg_map_main(exp_name, base_count, modulate_lambda_start, num_masks, num_frames, filter_difference=True, filter_s=0.7,
                        resize_height=(H // (F * 2)), resize_width=(H // (F * 2)), unique_labels=ref_unique_labels, base_folder=feature_folder,
                        frame_name_list=frame_name_list, mask_folder=feature_masks_folder, feature_timestep=feature_timestep) 
                

                del latent, input_video_tensor, latent_video
                
                torch.cuda.empty_cache()
                
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
    
    # make all the parser into one-line code
    parser.add_argument("--input_video_path", type=str, default=None, help="path to the input video which needs to be inverted")
    parser.add_argument("--exp_name", type=str, default="default", help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed for sampling")
    parser.add_argument("--motion_bucket_id", type=int, default=127, help="motion bucket id")
    parser.add_argument("--cond_aug", type=float, default=0.02, help="condition augmentation")
    parser.add_argument("--modulate_block_idx", type=str, default="8", help="selected block idx")
    parser.add_argument("--modulate_timestep", type=str, default="17", help="selected modulate timestep")
    parser.add_argument("--feature_timestep", type=str, default="24", help="selected timestep to extract features")
    parser.add_argument("--modulate_schedule", type=str, default="constant", help="modulate lambda schedule")
    parser.add_argument("--modulate_lambda_start", type=float, default=50.0, help="modulate lambda start")
    parser.add_argument("--modulate_lambda_end", type=float, default=50.0, help="modulate lambda end")
    parser.add_argument("--num_masks", type=int, default=10, help="number of masks to use")
    parser.add_argument("--feature_extraction_mode", type=str, default="kmeans_masks", help="feature extraction mode")
    parser.add_argument("--gt_mask_path", type=str, default=None, help="path to the ground truth mask of the first frame. Only needed \
        when feature_extraction_mode is match_gt_mask")
    parser.add_argument("--is_injected_features", default=False, action="store_true", help="whether to use injected features")
    parser.add_argument("--modulate_layer_type", type=str, default="spatial", help="modulate layer type")
    parser.add_argument("--modulate_attn_type", type=str, default="cross_attn", help="modulate attention type")
    parser.add_argument("--modulate_timestep_frames_schedule", type=str, default="constant", help="modulate timestep frames schedule")
    parser.add_argument("--disable_latent_blending", default=False, action="store_true", help="whether to disable latent blending")
    parser.add_argument("--is_refine_mask", default=False, action="store_true", help="whether to correct the mask")
    parser.add_argument("--is_aggre_attn", default=False, action="store_true", help="whether to use multiple attention maps for kmeans mask extraction")
  
    args = parser.parse_args()
    
    if args.disable_latent_blending:
        is_latent_blending = False
    else:
        is_latent_blending = True
    print(f"Is latent blending: {is_latent_blending}")
    print(f"Is multi attn: {args.is_aggre_attn}")
    

    sample(
        input_video_path=args.input_video_path,
        exp_name=args.exp_name,
        seed=args.seed,
        motion_bucket_id=args.motion_bucket_id,
        cond_aug=args.cond_aug,
        modulate_block_idx=args.modulate_block_idx,
        modulate_timestep=args.modulate_timestep,
        feature_timestep=args.feature_timestep,
        modulate_schedule=args.modulate_schedule,
        modulate_lambda_start=args.modulate_lambda_start,
        modulate_lambda_end=args.modulate_lambda_end,
        modulate_layer_type=args.modulate_layer_type,
        modulate_attn_type=args.modulate_attn_type,
        modulate_timestep_frames_schedule=args.modulate_timestep_frames_schedule,
        feature_extraction_mode=args.feature_extraction_mode,
        gt_mask_path=args.gt_mask_path,
        is_injected_features=args.is_injected_features,
        num_masks=args.num_masks,
        is_latent_blending=is_latent_blending,
        is_refine_mask=args.is_refine_mask,
        is_aggre_attn=args.is_aggre_attn)