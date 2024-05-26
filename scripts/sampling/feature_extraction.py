import argparse, os
import torch
from einops import rearrange
import torch.nn.functional
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.manifold import TSNE
# from tslearn.clustering import TimeSeriesKMeans
# from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm
from PIL import Image
from math import sqrt
import torch.nn as nn
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import einops
import copy
from collections import Counter
import time
import shutil
import sys
# from scripts.util.gms_matcher import GmsMatcher

            
def save_inidividual_masks_kmeans(feature_maps, selected_timestep, output_folder, 
                                  num_frames=14, num_clusters=10,
                                  feature_height=16, feature_width=16, attn_type="spatial",
                                  temporal_feature_maps=None,
                                  is_feature_reassign=False, frame_name_list=None):
    feature_maps = feature_maps.cpu().numpy()
    
    # normalize the feature_maps on the last channel to [0, 1]
    if feature_maps.shape[-1] > 1:
        feature_maps = feature_maps / np.max(np.abs(feature_maps), axis=-1, keepdims=True)
    # feature_maps = (feature_maps - np.min(feature_maps, axis=-1, keepdims=True)) / (np.max(feature_maps, axis=-1, keepdims=True) - np.min(feature_maps, axis=-1, keepdims=True))
              
    h = feature_height
    w = feature_width
    if attn_type == "spatial" or attn_type == "features":
        feature_maps_split = feature_maps[num_frames:]  # [f, hw, c]
        feature_maps_split_fit = rearrange(feature_maps_split, 'b t c -> (b t) c')
    elif attn_type == "temporal":
        # feature_maps = feature_maps / np.max(np.abs(feature_maps), axis=1, keepdims=True) # normalize temporal dim
        hq_sq = feature_maps.shape[0] // 2
        feature_maps_split = feature_maps[hq_sq:]  # [hw, f, c]
        feature_maps_split_fit = rearrange(feature_maps_split, 't b c -> (t b) c')
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # needs to set n_init to avoid random initialization

    kmeans.fit(feature_maps_split_fit)
    cluster_labels = kmeans.predict(feature_maps_split_fit)
    cluster_labels = np.reshape(cluster_labels, (num_frames, h, w))
    
    output_folder_name = output_folder.split("/")[-1]
    output_folder_name = output_folder_name + f"_masks_{num_clusters}"
    output_folder = output_folder.replace(output_folder.split("/")[-1], output_folder_name)

    num_iterations = feature_maps_split.shape[0]
    # for spatial, num_iterations = num_frames;
    # for temporal, num_iterations = hw.
    
    if attn_type == "spatial":
        all_kmeans_maps = []
        for i in range(num_iterations):  
            if frame_name_list is not None:
                frame_name = frame_name_list[i]
            else:
                frame_name = i
            output_folder_mask = os.path.join(output_folder, f"kmeans_time_{selected_timestep}_frame_{frame_name}")
            os.makedirs(output_folder_mask, exist_ok=True)
            feature_maps_split_transform = feature_maps_split[i]
            kmeans_map = cluster_labels[i]
            all_kmeans_maps.append(kmeans_map)
            
            for label in range(num_clusters):
                # Create a mask for the current label
                mask = np.where(kmeans_map == label, 255, 0).astype(np.uint8)
                # Convert the mask to a PIL Image
                mask_image = Image.fromarray(mask)
                # Save the mask image
                mask_image.save(os.path.join(output_folder_mask, f'mask_{label}.png'))
                
    elif attn_type == "temporal":
        kmeans_map = []
        for i in range(num_iterations):
            feature_maps_split_transform = feature_maps_split[i]
            # kmeans.fit(feature_maps_split_transform)  # for temporal, each frame is processed independently
            cluster_labels = kmeans.predict(feature_maps_split_transform)
            kmeans_map.append(cluster_labels)
        
        kmeans_map = np.array(kmeans_map)
        for i in range(num_frames):
            if frame_name_list is not None:
                frame_name = frame_name_list[i]
            else:
                frame_name = i
            output_folder_mask = os.path.join(output_folder, f"kmeans_time_{selected_timestep}_frame_{fframe_name}")
            os.makedirs(output_folder_mask, exist_ok=True)
            frame_mask = np.reshape(kmeans_map[:, i], (h, w))
            for label in range(num_clusters):
            # Create a mask for the current label
                mask = np.where(frame_mask == label, 255, 0).astype(np.uint8)
                # Convert the mask to a PIL Image
                mask_image = Image.fromarray(mask)
                # Save the mask image
                mask_image.save(os.path.join(output_folder_mask, f'mask_{label}.png'))
                
    unique_labels = np.arange(num_clusters)
    return unique_labels
                

def feature_matching_iterative(feature_maps, output_folder=None, input_coordinate=(14, 20), overlay_images_folder=None,
                                  feature_height=16, feature_width=16, attn_type="spatial", num_frames=14,
                                  top_k=1):
    if type(feature_maps) == torch.Tensor:
        feature_maps = feature_maps.cpu().numpy()
        
    if type(input_coordinate[0]) == float and input_coordinate[0] < 1:
        input_coordinate = (int(input_coordinate[0] * feature_height), int(input_coordinate[1] * feature_width))
    
    if output_folder is not None:
        plt.figure()
        overlay_image = Image.open(os.path.join(overlay_images_folder, f"frame_0.png"))
        plt.imshow(overlay_image, alpha=0.5, cmap="jet")
        plt.scatter(input_coordinate[1], input_coordinate[0], c="r", s=100)
        plt.axis("off")
        plt.savefig(os.path.join(output_folder, f"query_feature_point_0.png"), bbox_inches='tight')
    
    
    for query_frame_idx in range(num_frames - 1):
        if attn_type == "spatial":
            src_features = feature_maps[num_frames + query_frame_idx].reshape(feature_height, feature_width, -1)
            # trg_features should be all the features but the query frame
            trg_features = feature_maps[num_frames + query_frame_idx + 1][None, ...]
            trg_features = trg_features.transpose(0, 2, 1)  # [f, c, hw]
        elif attn_type == "temporal":
            hq_sq = feature_maps.shape[0] // 2
            src_features = feature_maps[hq_sq:, query_frame_idx].reshape(feature_height, feature_width, -1)
            trg_features = feature_maps[hq_sq:, query_frame_idx + 1][None, ...]  # [hw, f, c]
            trg_features = trg_features.transpose(1, 2, 0)  # [f, c, hw]
            trg_features = trg_features / np.linalg.norm(trg_features, axis=0, keepdims=True)
        
        src_features = src_features[input_coordinate[0], input_coordinate[1]][None, ...] # [1, c]
        
        src_features = src_features / np.linalg.norm(src_features, axis=1, keepdims=True)
        trg_features = trg_features / np.linalg.norm(trg_features, axis=1, keepdims=True)
        cos_map = einops.einsum(src_features, trg_features, "i c, f c s -> f s")
        cos_map = cos_map.reshape(1, feature_height, feature_width)  # only 1 frame at a time
        
        top_indices = np.argpartition(cos_map[0].ravel(), -top_k)[-top_k:]
        max_yx = np.unravel_index(top_indices, cos_map[0].shape)
        input_coordinate = (max_yx[0][-1], max_yx[1][-1])
        
        if output_folder is not None:
            heatmap = cos_map[0]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            plt.figure()
            plt.imshow(heatmap, alpha=0.5, cmap="jet")
            plt.scatter(max_yx[1], max_yx[0], c="r", s=100)
            plt.axis("off")
            plt.savefig(os.path.join(output_folder, f"feature_matching_{query_frame_idx + 1}.png"), bbox_inches='tight')
            
            plt.figure()
            overlay_image = Image.open(os.path.join(overlay_images_folder, f"frame_{query_frame_idx + 1}.png"))
            plt.imshow(overlay_image, alpha=0.5, cmap="jet")
            plt.scatter(max_yx[1], max_yx[0], c="r", s=100)
            plt.axis("off")
            plt.savefig(os.path.join(output_folder, f"query_feature_point_{query_frame_idx + 1}.png"), bbox_inches='tight')
        
            plt.close("all")
            
def dense_feature_matching_iterative(feature_maps, output_folder=None, input_coordinate_list=[], overlay_images_folder=None,
                                  feature_height=16, feature_width=16, attn_type="spatial", num_frames=14,
                                  upsample_height=None, upsample_width=None,
                                  top_k=1, device="cuda", use_aux=False, feature_matching_method="nn", is_backtracing=False):
    
    if type(feature_maps) == torch.Tensor:
        feature_maps = feature_maps.to(device)
        
    for i, input_coordinate in enumerate(input_coordinate_list):
        if input_coordinate[0] < 1:
            if upsample_height is not None and upsample_width is not None:
                input_coordinate = (int(input_coordinate[0] * upsample_height), int(input_coordinate[1] * upsample_width))
            else:
                input_coordinate = (int(input_coordinate[0] * feature_height), int(input_coordinate[1] * feature_width))
            input_coordinate_list[i] = input_coordinate
            
    input_h_list = [input_coordinate[0] for input_coordinate in input_coordinate_list]
    input_w_list = [input_coordinate[1] for input_coordinate in input_coordinate_list]
    
    num_query_points = len(input_h_list)
    batch_size = 500
    num_batches = num_query_points // batch_size + 1
    assert len(input_w_list) == num_query_points
    
    all_h_list = [input_h_list]
    all_w_list = [input_w_list]
        
    if output_folder is not None:
        plt.figure()
        try:
            overlay_image = Image.open(os.path.join(overlay_images_folder, f"0.png"))
        except:
            overlay_image = Image.open(os.path.join(overlay_images_folder, f"frame_0.png"))
        if upsample_height is not None and upsample_width is not None:
            overlay_image = overlay_image.resize((upsample_width, upsample_height), Image.BILINEAR)
        plt.imshow(overlay_image, alpha=0.5, cmap="jet")
        plt.scatter(input_w_list, input_h_list, c="r", s=100)
        plt.axis("off")
        plt.savefig(os.path.join(output_folder, f"query_feature_point_0.png"), bbox_inches='tight')
    
    start_time = time.time()
    
    for query_frame_idx in tqdm(range(num_frames - 1)):
        if attn_type == "spatial":
            if not is_backtracing:
                src_features = feature_maps[num_frames + query_frame_idx].reshape(feature_height, feature_width, -1)
                trg_features = feature_maps[num_frames + query_frame_idx + 1][None, ...]
            else:
                src_features = feature_maps[num_frames + query_frame_idx + 1].reshape(feature_height, feature_width, -1)
                trg_features = feature_maps[num_frames + query_frame_idx][None, ...]
            trg_features = trg_features.permute(0, 2, 1)  # [f, c, hw]
            aux_features = feature_maps[num_frames][None, ...]
            aux_features = aux_features.permute(0, 2, 1)  # [f, c, hw]
        elif attn_type == "temporal":
            hq_sq = feature_maps.shape[0] // 2
            if not is_backtracing:
                src_features = feature_maps[hq_sq:, query_frame_idx].reshape(feature_height, feature_width, -1)
                trg_features = feature_maps[hq_sq:, query_frame_idx + 1][None, ...]  # [hw, f, c]
            else:
                src_features = feature_maps[hq_sq:, query_frame_idx + 1].reshape(feature_height, feature_width, -1)
                trg_features = feature_maps[hq_sq:, query_frame_idx][None, ...]  # [hw, f, c]
            trg_features = trg_features.permute(1, 2, 0)  # [f, c, hw]
            trg_features = trg_features / torch.norm(trg_features, dim=0, keepdim=True)
            aux_features = feature_maps[hq_sq:, 0][None, ...]
            aux_features = aux_features.permute(1, 2, 0)  # [f, c, hw]
            aux_features = aux_features / torch.norm(aux_features, dim=0, keepdim=True)
            
        if upsample_height is not None and upsample_width is not None:
            
            src_features = src_features.permute(2, 0, 1).unsqueeze(0)  # [1, c, h, w]
            src_features = nn.Upsample(size=(upsample_height, upsample_width), mode='bilinear', align_corners=False)(src_features)
            src_features = src_features.permute(0, 2, 3, 1).squeeze()  # [h, w, c]
            
            trg_features = trg_features.reshape(1, -1, feature_height, feature_width)  # [f, c, h, w]
            trg_features = nn.Upsample(size=(upsample_height, upsample_width), mode='bilinear', align_corners=False)(trg_features)
            trg_features = trg_features.reshape(1, -1, upsample_height * upsample_width)  # [f, c, hw]
            
            aux_features = aux_features.reshape(1, -1, feature_height, feature_width)  # [f, c, h, w]
            aux_features = nn.Upsample(size=(upsample_height, upsample_width), mode='bilinear', align_corners=False)(aux_features)
            aux_features = aux_features.reshape(1, -1, upsample_height * upsample_width)  # [f, c, hw]
            
        output_h_list = []
        output_w_list = []
        
        for b in range(num_batches):
            
            if (b + 1) * batch_size > num_query_points:
                batch_h_list = input_h_list[b * batch_size:]
                batch_w_list = input_w_list[b * batch_size:]
                num_points_batch = len(batch_h_list)
            else:
                batch_h_list = input_h_list[b * batch_size: (b + 1) * batch_size]
                batch_w_list = input_w_list[b * batch_size: (b + 1) * batch_size]
                num_points_batch = batch_size
            src_features_batch = src_features[batch_h_list, batch_w_list] # [i, c]
            
            src_features_batch = src_features_batch / torch.norm(src_features_batch, dim=1, keepdim=True)
            trg_features = trg_features / torch.norm(trg_features, dim=1, keepdim=True)
            aux_features = aux_features / torch.norm(aux_features, dim=1, keepdim=True)
            
            if feature_matching_method == "nn":
                cos_map = einops.einsum(src_features_batch, trg_features, "i c, f c s -> i f s")  # for every query point there is a cos_map
                cos_map_aux = einops.einsum(src_features_batch, aux_features, "i c, f c s -> i f s")

                if upsample_height is not None and upsample_width is not None:
                    cos_map = cos_map.reshape(num_points_batch, upsample_height, upsample_width)
                    cos_map_aux = cos_map_aux.reshape(num_points_batch, upsample_height, upsample_width)
                else:
                    cos_map = cos_map.reshape(num_points_batch, feature_height, feature_width)  # only 1 frame at a time
                    cos_map_aux = cos_map_aux.reshape(num_points_batch, feature_height, feature_width)
                    
                cos_map = cos_map.cpu().numpy()
                cos_map_aux = cos_map_aux.cpu().numpy()
                
                if use_aux:
                    cos_map = query_frame_idx / (query_frame_idx + 1) * cos_map + 1 / (query_frame_idx + 1) * cos_map_aux
                for i in range(num_points_batch):
                    top_indices = np.argpartition(cos_map[i].ravel(), -top_k)[-top_k:]
                    max_yx = np.unravel_index(top_indices, cos_map[i].shape)
                    output_h_list.append(int(max_yx[0][-1]))
                    output_w_list.append(int(max_yx[1][-1]))
           
        if not is_backtracing:
            input_h_list = output_h_list.copy()
            input_w_list = output_w_list.copy()
        
        if output_folder is not None:
            plt.figure()
            try:
                overlay_image = Image.open(os.path.join(overlay_images_folder, f"{query_frame_idx + 1}.png"))
            except:
                overlay_image = Image.open(os.path.join(overlay_images_folder, f"frame_{query_frame_idx + 1}.png"))
            if upsample_height is not None and upsample_width is not None:
                overlay_image = overlay_image.resize((upsample_width, upsample_height), Image.BILINEAR)
            plt.imshow(overlay_image, alpha=0.5, cmap="jet")
            plt.scatter(output_w_list, output_h_list, c="r", s=100)
            plt.axis("off")
            plt.savefig(os.path.join(output_folder, f"query_feature_point_{query_frame_idx + 1}.png"), bbox_inches='tight')
        
            plt.close("all")
            
        all_h_list.append(output_h_list)
        all_w_list.append(output_w_list)

    print("dense feature matching iterative time taken:", time.time() - start_time)
    if overlay_images_folder is not None and output_folder is not None:
        draw_correpondences(all_h_list, all_w_list, overlay_images_folder, output_folder)
            
    return all_h_list, all_w_list

            
def dense_tracking(feature_maps, mask_path=None, output_folder=None, overlay_images_folder=None,
                                  feature_height=16, feature_width=16, attn_type="spatial", num_frames=14,
                                  top_k=1, device="cuda", use_aux=True, is_backtracing=False):
    if mask_path is not None:
        mask_image = Image.open(mask_path).resize((feature_width, feature_height))
        image_w, image_h = mask_image.size
        mask_np = np.array(mask_image)
        # get all the coordinates where the values are 255
        input_h_list, input_w_list = np.where(mask_np == 255)
    else:
        # use all the points in the feature map
        input_h_list, input_w_list = torch.meshgrid(torch.arange(feature_height), torch.arange(feature_width))
        input_h_list = input_h_list.flatten().to(device)
        input_w_list = input_w_list.flatten().to(device)
        image_w = feature_width
        image_h = feature_height
    input_h_list = [h / image_h for h in input_h_list]
    input_w_list = [w / image_w for w in input_w_list]
    
    print("number of query points:", len(input_h_list))
    
    input_coordinate_list = list(zip(input_h_list, input_w_list))
    
    all_h_list, all_w_list = dense_feature_matching_iterative(feature_maps, output_folder=output_folder, 
                                     input_coordinate_list=input_coordinate_list, 
                                     overlay_images_folder=overlay_images_folder,
                                  feature_height=feature_height, feature_width=feature_width, 
                                  # upsample_height=image_h, upsample_width=image_w,
                                  attn_type=attn_type, num_frames=num_frames,
                                  top_k=top_k, device=device, use_aux=use_aux, is_backtracing=is_backtracing)
    
    if output_folder is not None:        
        for frame_id in range(num_frames - 1):
            new_mask_np = np.zeros((image_h, image_w))
            new_mask_np[all_h_list[frame_id], all_w_list[frame_id]] = 255
            new_mask_image = Image.fromarray(new_mask_np).convert("L")
            new_mask_image.save(os.path.join(output_folder, f"new_mask_{frame_id + 1}.png"))
            
    return all_h_list, all_w_list


def correct_low_res_mask(feature_maps, mask_folder, output_folder=None, num_clusters=10, overlay_images_folder=None,
                                  feature_height=16, feature_width=16, attn_type="spatial", num_frames=14, timestep=24,
                                  top_k=1, anchor_label_method="common", frame_name_list=None, ref_unique_labels=None, spatial_filter=True):
    all_h_list, all_w_list = dense_tracking(feature_maps, mask_path=None, output_folder=None,
                                            overlay_images_folder=overlay_images_folder,
                                            feature_height=feature_height, feature_width=feature_width,
                                            attn_type=attn_type, num_frames=num_frames, top_k=top_k)
    all_h_np = np.array(all_h_list)
    all_w_np = np.array(all_w_list)
    num_points = all_h_np.shape[1]
    anchor_id = 0
    
    ori_seg_map_list = []
    for i in range(num_frames):
        if frame_name_list is not None:
            frame_name = frame_name_list[i]
        else:
            frame_name = i
        seg_map = generate_aggregate_mask(mask_folder, 24, num_clusters, frame_name, feature_height, feature_width,
                                          labels=ref_unique_labels)  
        ori_seg_map_list.append(seg_map)
        
    ori_seg_map_list = np.array(ori_seg_map_list)  # [f, h, w]
    new_seg_map_list = copy.deepcopy(ori_seg_map_list)
    
    if spatial_filter:
        spatial_threshold = 1
        for p in range(num_points):
            trj_h = all_h_np[:, p]
            trj_w = all_w_np[:, p]
            num_frames = len(trj_h)
            for frame_id in range(1, num_frames):
                if trj_h[frame_id] - trj_h[frame_id - 1] > spatial_threshold or trj_w[frame_id] - trj_w[frame_id - 1] > spatial_threshold:
                    all_h_np[:, p] = -1
                    all_w_np[:, p] = -1
                    break
            
        cols_with_nan = np.any(all_h_np == -1, axis=0)
        all_h_np = all_h_np[:, ~cols_with_nan]
        all_w_np = all_w_np[:, ~cols_with_nan]
                
        num_points = all_h_np.shape[1]
        print("filter_num_points", num_points)
    
    for p in range(num_points):
        trj_h = all_h_np[:, p]
        trj_w = all_w_np[:, p]
        if anchor_label_method == "common":
            all_labels = []
            for frame_id in range(num_frames):
                all_labels.append(ori_seg_map_list[frame_id, trj_h[frame_id], trj_w[frame_id]])
            counts = Counter(all_labels)
            most_common_label = counts.most_common(1)[0][0]
            for frame_id in range(num_frames):
                new_seg_map_list[frame_id, trj_h[frame_id], trj_w[frame_id]] = most_common_label
        elif anchor_label_method == "first":
            anchor_label = ori_seg_map_list[0, trj_h[0], trj_w[0]]
            for frame_id in range(1, num_frames):
                new_seg_map_list[frame_id, trj_h[frame_id], trj_w[frame_id]] = anchor_label
        
        
    if output_folder is not None:
        for i in range(num_frames):
            plt.figure()
            rgb_image = convert_label_to_rgb(new_seg_map_list[i])
            rgb_image = Image.fromarray(rgb_image)
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(os.path.join(output_folder, f"corrected_mask_{i}.png"), bbox_inches='tight')
            
            plt.figure()
            rgb_image = convert_label_to_rgb(ori_seg_map_list[i])
            rgb_image = Image.fromarray(rgb_image)
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(os.path.join(output_folder, f"original_mask_{i}.png"), bbox_inches='tight')
            
            plt.close("all")
        
    output_binary_mask_folder_name = mask_folder.split("/")[-1] + "_corrected"
    output_binary_mask_folder = mask_folder.replace(mask_folder.split("/")[-1], output_binary_mask_folder_name)
    os.makedirs(output_binary_mask_folder, exist_ok=True)
    
    
    for i in range(num_frames):
        if frame_name_list is not None:
            frame_name = frame_name_list[i]
        else:
            frame_name = i
        output_binary_mask_folder_frame = os.path.join(output_binary_mask_folder, f"kmeans_time_{timestep}_frame_{frame_name}")
        os.makedirs(output_binary_mask_folder_frame, exist_ok=True)
        generate_binary_mask(new_seg_map_list[i], output_binary_mask_folder_frame, ref_unique_labels)

    ref_mask = new_seg_map_list.reshape(-1)
    return ref_unique_labels, ref_mask, None
        
        
def draw_correpondences(all_h_list, all_w_list, overlay_images_folder, output_folder):
    num_frames = len(all_h_list)
    all_images = []
    for frame_id in range(num_frames):
        try:
            overlay_image = cv2.imread(os.path.join(overlay_images_folder, f"frame_{frame_id}.png"))
        except:
            overlay_image = cv2.imread(os.path.join(overlay_images_folder, f"{frame_id}.png"))
        all_images.append(overlay_image)
            
    all_correspondences = []
    for frame_id in range(num_frames):
        h_list = all_h_list[frame_id]
        w_list = all_w_list[frame_id]
        all_correspondences.append(list(zip(h_list, w_list)))
        
    # create a blank canvas to display all frames side by side
    canvas = np.zeros((all_images[0].shape[0], all_images[0].shape[1] * num_frames, 3), dtype=np.uint8)
    # place each frame onto the canvas
    for i, image in enumerate(all_images):
        canvas[:, i * image.shape[1]: (i + 1) * image.shape[1]] = image
        
    # draw line between the correspondences
    num_points = len(all_correspondences[0])
    # rand_idx = np.random.choice(num_points, 30, replace=False)
    for p in range(num_points):
        for frame_id in range(1, num_frames):
            h1, w1 = all_correspondences[frame_id - 1][p]
            h2, w2 = all_correspondences[frame_id][p]
            w1 += (frame_id - 1) * all_images[0].shape[1]
            w2 += frame_id * all_images[0].shape[1]
            cv2.line(canvas, (int(w1), int(h1)), (int(w2), int(h2)), (0, 255, 0), 1)
            
    cv2.imwrite(os.path.join(output_folder, "correspondences.png"), canvas)
    
    
def generate_aggregate_mask(mask_folder, timestep, num_masks, frame_id, resize_height, resize_width,
                            labels=None):
    seg_map = np.zeros((resize_height, resize_width))
    all_masks = []
    
    if labels is None:
        iterator = range(num_masks)
    else:
        iterator = labels
    
    for i in iterator: 
        mask_path = os.path.join(mask_folder, f"kmeans_time_{timestep}_frame_{frame_id}", f"mask_{i}.png")
        mask_image = Image.open(mask_path).resize((resize_width, resize_height))
        mask = np.array(mask_image)
        all_masks.append(mask)
    
    seg_map = np.argmax(all_masks, axis=0)
    
    if labels is not None:
        seg_map = labels[seg_map]
    
    return seg_map

def generate_binary_mask(aggregate_mask, output_folder, labels=None):
    if labels is None:
        unique_labels = np.unique(aggregate_mask)
        for i, label in enumerate(unique_labels):
            mask = np.where(aggregate_mask == label, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask).convert("L")
            mask_image.save(os.path.join(output_folder, f"mask_{label}.png"))
    else:
        for i in labels:
            mask = np.where(aggregate_mask == i, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask).convert("L")
            i = int(i)
            mask_image.save(os.path.join(output_folder, f"mask_{i}.png"))
            
    
def convert_label_to_rgb(clustered_tensor):
    color_map = np.loadtxt("scripts/util/color_map_soft.txt", dtype=np.uint8, delimiter=',')
    # Create an empty image array
    rgb_image = np.zeros((clustered_tensor.shape[0], clustered_tensor.shape[1], 3), dtype=np.uint8)
    rgb_image = color_map[clustered_tensor]

    return rgb_image

def match_gt_mask(feature_maps, gt_mask_path, feature_height, feature_width, output_folder, num_masks, 
                        selected_timestep=24, frame_name_list=None, ref_mask=None, ref_feature_map=None,
                        ref_unique_labels=None, use_gt_mask=False):
    feature_maps = feature_maps.cpu().numpy()
    num_frames = feature_maps.shape[0] // 2
    feature_maps = feature_maps[num_frames:]
    num_clusters = num_masks
    
    if feature_maps.shape[-1] > 1:
        feature_maps = feature_maps / np.max(np.abs(feature_maps), axis=-1, keepdims=True)
    # feature_maps = (feature_maps - np.min(feature_maps, axis=-1, keepdims=True)) / (np.max(feature_maps, axis=-1, keepdims=True) - np.min(feature_maps, axis=-1, keepdims=True))
              
    h = feature_height
    w = feature_width
    feature_maps_split_fit = rearrange(feature_maps, 'b t c -> (b t) c')
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # needs to set n_init to avoid random initialization

    output_folder_name = output_folder.split("/")[-1] + f"_masks_{num_masks}"
    output_folder = output_folder.replace(output_folder.split("/")[-1], output_folder_name)

    if ref_mask is None:
        feature_maps_split_fit = rearrange(feature_maps, 'b t c -> (b t) c')
        
        kmeans.fit(feature_maps_split_fit)

        cluster_labels = kmeans.predict(feature_maps[0])
        cluster_labels = np.reshape(cluster_labels, (h, w))
        cluster_labels_image = Image.fromarray(convert_label_to_rgb(cluster_labels))
        os.makedirs(output_folder, exist_ok=True)
        cluster_labels_image.save(os.path.join(output_folder, "kmeans_cluster_labels.png"))
        cluster_labels = np.reshape(cluster_labels, (h * w))
        fake_mask = cluster_labels
        
        if gt_mask_path is not None:
            mask_image = Image.open(gt_mask_path).resize((feature_width, feature_height), Image.NEAREST)
            mask_np = np.array(mask_image)
            mask_np = mask_np.flatten()
            labels = np.unique(mask_np)
            
        else:
            mask_np = cluster_labels
        
        if not use_gt_mask:
            ref_mask = np.zeros((h * w)).astype(int)
            for fake_label in np.unique(fake_mask):
                sub_gt_mask = mask_np[fake_mask == fake_label]
                values, counts = np.unique(sub_gt_mask, return_counts=True)
                gt_label = values[np.argmax(counts)]
                ref_mask[fake_mask == fake_label] = gt_label
        else:
            assert gt_mask_path is not None
            ref_mask = mask_np
            
        ref_feature_map = feature_maps[0]
        
        
    if ref_unique_labels is None:
        ref_unique_labels = np.unique(ref_mask)
        
    unique_labels = np.unique(ref_mask)
    
    knn = KNeighborsClassifier(n_neighbors=4)
    
    knn.fit(ref_feature_map, ref_mask)
    
    cluster_labels = knn.predict(feature_maps_split_fit)
    cluster_labels = np.reshape(cluster_labels, (num_frames, h * w))

    num_iterations = feature_maps.shape[0]
    
    all_new_masks = []    
    for frame_id in range(num_iterations):  
        new_mask = cluster_labels[frame_id]
        all_new_masks.append(new_mask)
        new_mask = np.reshape(new_mask, (h, w))
        frame_unique_labels = np.unique(new_mask)
        if frame_name_list is not None:
            frame_name = frame_name_list[frame_id]
        else:
            frame_name = frame_id
        output_frame_folder = os.path.join(output_folder, f"kmeans_time_{selected_timestep}_frame_{frame_name}")
        if not os.path.exists(output_frame_folder):
            os.makedirs(output_frame_folder)
        else:
            shutil.rmtree(output_frame_folder)           # Removes all the subdirectories!
            os.makedirs(output_frame_folder)
        # ref_unique_labels will be only used to generate the binary mask
        # in case a certain class is missing from the current frame
        # an empty mask will be generated
        generate_binary_mask(new_mask, output_frame_folder, ref_unique_labels)
        
    
    ref_mask = np.concatenate(all_new_masks, axis=0)
    ref_feature_map = feature_maps_split_fit

        
    return unique_labels, ref_mask, ref_feature_map


def load_experiments_features(feature_maps_paths, blocks, feature_type, t, frame_id=None):
    
    if frame_id is None:
        feature_maps = []
        for i, feature_maps_path in enumerate(feature_maps_paths):
            if "attn" in feature_type:
                feature_map = torch.load(os.path.join(feature_maps_path, f"{blocks}_{feature_type}_time_{t}.pt"))
            else:
                feature_map = \
                    torch.load(os.path.join(feature_maps_path, f"{blocks}_{feature_type}_time_{t}.pt"))
                if len(feature_map.shape) == 4:
                    feature_map = rearrange(feature_map, 'b c h w -> b (h w) c')
            feature_maps.append(feature_map)

    else:
        feature_maps = []
        if frame_id is not None:
            for i, feature_maps_path in enumerate(feature_maps_paths):
                feature_map = torch.load(os.path.join(feature_maps_path, f"{blocks}_{feature_type}_time_{t}_frame_{frame_id}.pt"))
                feature_maps.append(feature_map)
            

    return feature_maps
                    
def feature_extraction_main(mode, num_clusters, t_start, block_name, experiment_name, fit_experiments, feature_types, feature_height, feature_width, selected_timestep,
                            frame_name_list=None, base_folder=None, 
                            ref_mask=None, ref_feature_map=None, ref_unique_labels=None, gt_mask_path=None,
                            num_frames=None, mask_folder=None, use_gt_mask=False):
    if base_folder is None:
        exp_path_root = "features_outputs"
    else:
        exp_path_root = base_folder
    selected_timestep = [int(timestep) for timestep in selected_timestep.split(",") if timestep]
    fit_experiments = [item for item in fit_experiments.split(",") if item]
    transform_experiments = fit_experiments

    if num_frames is None:
        num_frames = 14
        
    total_steps = 25
    time_range = np.arange(t_start, total_steps, 1)
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

    block_name = block_name.split(",")
    if len(block_name) == 1:
        block_name = block_name[0]

    
    print(f"visualizing features experiments: block - {block_name}; experiment name: {experiment_name}")

    transform_feature_maps_paths = []
    for experiment in transform_experiments:
        transform_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))

    fit_feature_maps_paths = []
    for experiment in fit_experiments:
        fit_feature_maps_paths.append(os.path.join(exp_path_root, experiment, "feature_maps"))
    

    feature_types = [item for item in feature_types.split(",") if item]
    feature_pca_paths = {}
    
    if mode == "kmeans_masks":
        pca_folder_path = os.path.join(exp_path_root, experiment_name, "kmeans_masks")
        os.makedirs(pca_folder_path, exist_ok=True)
    elif mode == "correct_low_res_mask":
        pca_folder_path = os.path.join(exp_path_root, experiment_name, "correct_low_res_mask")
        os.makedirs(pca_folder_path, exist_ok=True)
    elif mode == "match_gt_mask":
        pca_folder_path = os.path.join(exp_path_root, experiment_name, "match_gt_mask")
        os.makedirs(pca_folder_path, exist_ok=True)
    else:
        raise ValueError(f"mode {mode} not supported")

    for feature_type in feature_types:
        if isinstance(block_name, list):
            block_str = '_'.join(block_name)
            feature_pca_path = os.path.join(pca_folder_path, f"{block_str}_{feature_type}")
        else:
            feature_pca_path = os.path.join(pca_folder_path, f"{block_name}_{feature_type}")
        feature_pca_paths[feature_type] = feature_pca_path
        
    for t in selected_timestep:
        fit_features = []
        transform_features = []
        for feature_type in feature_types:
            if "temporal" in feature_type:
                attn_type = "temporal"
            elif "features" in feature_type:
                attn_type = "features"
            else:
                attn_type = "spatial"
            
            if isinstance(block_name, list):
                transform_features = []
                for sub_block_name in block_name:
                    features = load_experiments_features(transform_feature_maps_paths, sub_block_name, feature_type, t,)
                    features = torch.cat(features, dim=0)
                    transform_features.append(features)
                transform_features = torch.mean(torch.stack(transform_features), dim=0)
            else:
                transform_features = load_experiments_features(transform_feature_maps_paths, block_name, feature_type, t,)
                transform_features = torch.cat(transform_features, dim=0)
            
            
            temporal_features = None

                
            if mode == "kmeans_masks":
                unique_labels = save_inidividual_masks_kmeans(transform_features, 
                                            t, 
                                            feature_pca_paths[feature_type], 
                                            num_frames=num_frames,
                                            num_clusters=num_clusters,
                                            feature_height=feature_height,
                                            feature_width=feature_width,
                                            attn_type=attn_type,
                                            temporal_feature_maps=temporal_features,
                                            is_feature_reassign=False,
                                            frame_name_list=frame_name_list)
                    
            
            elif mode == "match_gt_mask":
                unique_labels, ref_mask, ref_feature_map = match_gt_mask(transform_features, 
                                    gt_mask_path=gt_mask_path,
                                    feature_height=feature_height,
                                    feature_width=feature_width,
                                    output_folder=feature_pca_paths[feature_type],
                                    num_masks=num_clusters,
                                    selected_timestep=t,
                                    frame_name_list=frame_name_list,
                                    ref_mask=ref_mask,
                                    ref_feature_map=ref_feature_map,
                                    ref_unique_labels=ref_unique_labels,
                                    use_gt_mask=use_gt_mask)
            
            elif mode == "correct_low_res_mask":
                unique_labels, ref_mask, ref_feature_map = correct_low_res_mask(transform_features, 
                                    mask_folder=mask_folder,
                                    overlay_images_folder=None,
                                    output_folder=None, # feature_pca_paths[feature_type], 
                                    feature_height=feature_height,
                                    feature_width=feature_width,
                                    attn_type=attn_type,
                                    timestep=t,
                                    num_frames=num_frames,
                                    frame_name_list=frame_name_list,
                                    ref_unique_labels=ref_unique_labels,)
            
    return unique_labels, ref_mask, ref_feature_map

if __name__ == "__main__":
    main()
