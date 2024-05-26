import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2

def compute_difference(modulated_map1_path, modulated_map2_path, output_folder, output_folder_vis, i):
    feature_map1 = np.array(Image.open(modulated_map1_path))
    feature_map2 = np.array(Image.open(modulated_map2_path))
     
    # compute the euclidean distance between two images over the color channels
    difference = np.sqrt(np.sum((feature_map1 - feature_map2) ** 2, axis=2))
    # apply gaussian filter in opencv
    difference = cv2.GaussianBlur(difference, (5, 5), 3)
    difference_max = difference.max()
    
    difference_image = Image.fromarray(difference).convert("L")
    difference_image.save(os.path.join(output_folder, f"{i}.jpg"))
    
    # if the pixel value is greater than 10, set it to 255; otherwise, set it to 0
    
    difference_normal = difference / difference.max() * 255
    
    difference_vis = Image.fromarray(difference_normal).convert("L")
    difference_vis.save(os.path.join(output_folder_vis, f"{i}.jpg"))
    
    return difference, difference_normal

def filter_difference_map(difference_map, mask_image, filter_s=0.5):
    # for the region outside of mask on difference_map, make a weight of filter_s
    
    height, width = difference_map.shape
    mask = np.array(mask_image.resize((width, height), Image.LANCZOS)) / 255.0
    
    difference_map = difference_map * mask + filter_s * difference_map * (1 - mask)
    
    return difference_map

# process difference maps across all the masks and frames
def generate_difference_map(exp_name, basecount, modulate_lambda, num_masks, num_frames, unique_labels=None, base_folder=None,
                            frame_name_list=None):
    if base_folder is None:
        base_folder = f"outputs"
        modulated_map_folder = f"outputs/modulate_video_sample/svd/{exp_name}"
    else:
        modulated_map_folder = os.path.join(base_folder, f'{exp_name}/modulated_output')

    output_folder = os.path.join(base_folder, f'{exp_name}/difference_map/original_map/') 
    output_folder_vis = os.path.join(base_folder, f'{exp_name}/difference_map/vis_map/') 

    mask_iterator = unique_labels if unique_labels is not None else range(num_masks)

    for i in mask_iterator:
        modulated_map1_folder = os.path.join(modulated_map_folder, f"{basecount:06d}_l_{modulate_lambda}_mask_{i}")
        modulated_map2_folder = os.path.join(modulated_map_folder, f"{basecount:06d}_l_{-modulate_lambda}_mask_{i}")
        output_folder_mask = os.path.join(output_folder, f"{basecount:06d}_l_{modulate_lambda}_mask_{i}")
        output_folder_vis_mask = os.path.join(output_folder_vis, f"{basecount:06d}_l_{modulate_lambda}_mask_{i}")
        os.makedirs(output_folder_mask, exist_ok=True)
        os.makedirs(output_folder_vis_mask, exist_ok=True)
        
        for frame_id in range(num_frames):
            if frame_name_list is not None:
                frame_name = frame_name_list[frame_id]
            else:
                frame_name = frame_id
            modulated_map1_path = os.path.join(modulated_map1_folder, f"{frame_name}.png")
            modulated_map2_path = os.path.join(modulated_map2_folder, f"{frame_name}.png")
            compute_difference(modulated_map1_path, modulated_map2_path, output_folder_mask, output_folder_vis_mask, frame_name)
            
            
            
# compute the final segmentation maps
def get_seg_map_main(exp_name, basecount, modulate_lambda, num_masks, num_frames, filter_difference, filter_s=0.7,
                     resize_height=28, resize_width=52, unique_labels=None, 
                     base_folder=None, mask_folder=None, frame_name_list=None, feature_timestep="24",
                     is_smooth=False, batch_id=None, color_map_path=None, color_map_mapping="order"):
    
    generate_difference_map(exp_name, basecount, modulate_lambda, num_masks, num_frames, unique_labels=unique_labels,
                            base_folder=base_folder, frame_name_list=frame_name_list, )
    
    
    if base_folder is None:
        base_folder = f"outputs"
    
    difference_map_folder = os.path.join(base_folder, f'{exp_name}/difference_map/original_map/') 

    if filter_difference:
        segmentation_map_folder = os.path.join(base_folder, f'{exp_name}/segmentation_map_f_{filter_s}')
    else:
        segmentation_map_folder = os.path.join(base_folder, f'{exp_name}/segmentation_map') 
    segmentation_map_folder = os.path.join(segmentation_map_folder, f"{basecount:06d}_l_{modulate_lambda}")
    os.makedirs(segmentation_map_folder, exist_ok=True)

    if mask_folder is None:
        mask_folder = f"features_outputs/kmeans_masks/{exp_name}/output_block_8_spatial_self_attn_q_masks_{num_masks}"
    
    if "kmeans" or "match_concept_embed" in mask_folder:
        mask_source = "kmeans"
    elif "sam" in mask_folder:
        mask_source = "sam"
    elif "knn" in mask_folder:
        mask_source = "knn"
    
    if color_map_path is None:    
        color_map_path = "scripts/util/color_map_soft.txt"
    color_map = np.loadtxt(color_map_path, delimiter=',')
    
    mask_iterator = unique_labels if unique_labels is not None else range(num_masks)
    
    all_frames_difference_maps = []
    for i in mask_iterator:  
        all_difference_maps = []  
        for frame_id in range(num_frames):
            seg_map = np.zeros((512, 512))            
            if frame_name_list is not None:
                frame_name = frame_name_list[frame_id]
            else:
                frame_name = frame_id  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            difference_map_path = os.path.join(difference_map_folder, f"{basecount:06d}_l_{modulate_lambda}_mask_{i}", f"{frame_name}.jpg")
            difference_map = np.array(Image.open(difference_map_path))
            # normalize the difference map to [0, 1]
            difference_map = difference_map / (np.max(difference_map) + 1e-5)
            if filter_difference:
                if mask_source == "kmeans":
                    mask_path = os.path.join(mask_folder, f'kmeans_time_{feature_timestep}_frame_{frame_name}', f"mask_{i}.png")
                    mask_image = Image.open(mask_path)
                elif mask_source == "knn":
                    mask_path = os.path.join(mask_folder, f'knn_time_{feature_timestep}_frame_{frame_name}', f"mask_{i}.png")
                    mask_image = Image.open(mask_path)
                elif mask_source == "sam":
                    mask_path = os.path.join(mask_folder, f'sam_frame_{frame_name}', f"mask_{i}.png")
                    mask_image = Image.open(mask_path)
                    mask_image = mask_image.resize((base_width, base_height), Image.LANCZOS)
                difference_map = filter_difference_map(difference_map, mask_image, filter_s=filter_s)
            all_difference_maps.append(difference_map)  
        all_frames_difference_maps.append(all_difference_maps) 

    for frame_id in range(num_frames):
        if frame_name_list is not None:
            frame_name = frame_name_list[frame_id]
        else:
            frame_name = frame_id  
        all_difference_maps = []    
        for i in range(len(mask_iterator)):
            all_difference_maps.append(all_frames_difference_maps[i][frame_id])
            
                                     
        seg_map = np.argmax(np.array(all_difference_maps), axis=0)
        # map the seg_map to the color map
        if filter_difference:
            segmentation_map_folder_raw = os.path.join(base_folder, f'{exp_name}/segmentation_map_raw_f_{filter_s}')
        else:
            segmentation_map_folder_raw = os.path.join(base_folder, f'{exp_name}/segmentation_map_raw') 
        segmentation_map_folder_raw = os.path.join(segmentation_map_folder_raw, f"{basecount:06d}_l_{modulate_lambda}")
        os.makedirs(segmentation_map_folder_raw, exist_ok=True)
        # map the seg_map based on the mask_iterator
        seg_map_raw = mask_iterator[seg_map]
        seg_map_raw = Image.fromarray(seg_map_raw.astype(np.uint8))
        seg_map_raw.save(os.path.join(segmentation_map_folder_raw, f"{frame_name}.png"))
        if color_map_mapping == "order":
            seg_map_color = color_map[seg_map]
        else:
            seg_map_color = color_map[seg_map_raw]
        seg_map_color = Image.fromarray(seg_map_color.astype(np.uint8))
        seg_map_color.save(os.path.join(segmentation_map_folder, f"{frame_name}.jpg"))
        