# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NVIDIA
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import pickle
import os
import numpy as np
from nuscenes.eval.common.utils import Quaternion
import json
from os import path as osp
from planning_utils import PlanningMetric
import torch
from tqdm import tqdm
import threading
import re

from PIL import Image, ImageDraw

def append_tangent_directions(traj):
    directions = []

    if np.linalg.norm(traj[0]) < 0.5:
        directions.append(0.0)
    else:
        directions.append(np.arctan2(traj[0][1], traj[0][0]))

    for i in range(1, len(traj)):
        vector = traj[i] - traj[i-1]
        # filter small vectors and keep its last angle
        if np.linalg.norm(vector) < 0.3:
            angle = directions[-1]
        else:
            angle = np.arctan2(vector[1], vector[0])
        directions.append(angle)
    directions = np.array(directions).reshape(-1, 1)
    traj_yaw = np.concatenate([traj, directions], axis=-1)
    return traj_yaw

def print_progress(current, total):
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.2f}%)", end="")

def visualize_bev(bev_seg, drivable_seg, ego_seg, gt_traj, pred_traj, obj_box_coll, out_of_drivable, data, vis_path):
    # print('bev_seg shape', bev_seg.shape,'drivable_seg shape',drivable_seg.shape, 'ego_seg shape', ego_seg.shape)
    # visualize the bev_seg and ego/gt trajectory
    vis_save_path = vis_path + 'vis/'
    if obj_box_coll.max().item() > 0 and not out_of_drivable:
        vis_save_path = vis_save_path + 'obj_collision/'
    elif obj_box_coll.max().item() == 0 and out_of_drivable:
        vis_save_path = vis_save_path + 'out_of_drivable/'
    elif obj_box_coll.max().item() > 0 and out_of_drivable:
        vis_save_path = vis_save_path + 'both_collision/'
    os.makedirs(vis_save_path, exist_ok=True)

    # draw all bev segs in different colors
    # has shape (7,1000,1000)
    bev_img = Image.fromarray((np.zeros_like(bev_seg[0])).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(bev_img)
    # draw each segment in a different color
    for j in range(0, bev_seg.shape[0]):
            mask = (bev_seg[j] > 0)

            mask_img = Image.fromarray((mask * 255).astype(np.uint8))

            color_layer = Image.new("RGB", bev_img.size, (255-j*(255//bev_seg.shape[0]),255- j*(255//bev_seg.shape[0]), 255- j*(255//bev_seg.shape[0])))

            bev_img = Image.composite(color_layer, bev_img, mask_img)

    for k in range(0, ego_seg.shape[0]):
        mask = (ego_seg[k] > 0)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        color_layer = Image.new("RGB", bev_img.size, (255 - k*(255//ego_seg.shape[0]), 0, 255 - k*(255//ego_seg.shape[0])))

        bev_img = Image.composite(color_layer, bev_img, mask_img)



    # draw drivable_seg on top
    drivable_img = Image.fromarray((drivable_seg*255).astype(np.uint8)).convert("RGB")
    bev_img = Image.blend(bev_img, drivable_img, alpha=0.3)

    # draw gt traj in greens
    draw = ImageDraw.Draw(bev_img)
    for j in range(gt_traj.shape[1]-1):
        x1 = int((gt_traj[0, j, 0] + 50 ) / 0.1)
        y1 = int((gt_traj[0, j, 1] + 50 ) / 0.1)
        x2 = int((gt_traj[0, j+1, 0]+ 50 ) / 0.1)
        y2 = int((gt_traj[0, j+1, 1]  + 50 ) / 0.1)
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=6)

    # draw pred traj in red
    for j in range(pred_traj.shape[1]-1):
        x1 = int((pred_traj[0, j, 0] + 50 ) / 0.1)
        y1 = int((pred_traj[0, j, 1] + 50 ) / 0.1 )
        x2 = int((pred_traj[0, j+1, 0]+ 50 ) / 0.1 )
        y2 = int((pred_traj[0, j+1, 1]  + 50 ) / 0.1 )
        draw.line((x1, y1, x2, y2), fill=(255, 128, 0), width=6)

    bev_img.save(osp.join(vis_save_path, f"{data['token']}_bev.png"))


def process_data(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric, visualize=False, vis_path='results_planning_only/'):
    ego_boxes = np.array([[ 0, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]]) 
    for i in range(start, end):
        try:
            data = key_infos['infos'][i]
            if data['token'] not in preds.keys():
                continue
            # print('pred traj', preds[data['token']].shape)
            pred_traj = preds[data['token']][:6, :]
            # print('pred traj after selection', pred_traj.shape)
            gt_traj, mask = data['gt_planning'], data['gt_planning_mask'][0]
            gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
            gt_agent_feats = np.concatenate([data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6], data['gt_fut_idx']], -1)
            bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

            e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
            e2g_t = data['ego2global_translation']
            drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)
            pred_traj_yaw = append_tangent_directions(pred_traj[..., :2])
            pred_traj_mask = np.concatenate([pred_traj_yaw[..., :2].reshape(1, -1), np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1), pred_traj_yaw[..., 2:].reshape(1, -1)], axis=-1)
            ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)
            
            pred_traj = torch.from_numpy(pred_traj).unsqueeze(0)
            gt_traj = torch.from_numpy(gt_traj[..., :2])

        
            print(f"Processing sample {i+1}/{end} - Token: {data['token']}" + '-'*80)
            # print('pred_traj', pred_traj, 'gt_traj', gt_traj,)

            fut_valid_flag = mask.all()
            future_second = 3
            if fut_valid_flag:
                with lock:
                    metric_dict['samples'] += 1

                    metric_dict['all_pred_traj'] = np.append(metric_dict['all_pred_traj'], pred_traj[:,:6,:].numpy(), axis=0)
                    metric_dict['all_gt_traj'] = np.append(metric_dict['all_gt_traj'], gt_traj[:,:6,:].numpy(), axis=0)
                    

                for i in range(future_second):
                    cur_time = (i+1)*2
                    ade = float(
                        sum(
                            np.sqrt(
                                (pred_traj[0, i, 0] - gt_traj[0, i, 0]) ** 2
                                + (pred_traj[0, i, 1] - gt_traj[0, i, 1]) ** 2
                            )
                            for i in range(cur_time)
                        )
                        / cur_time
                    )
                    metric_dict['l2_{}s'.format(i+1)] += ade

                    # print(f"Average Displacement Error (ADE) for {cur_time}s: {ade:.4f}")
                    
                    obj_coll, obj_box_coll = planning_metric.evaluate_coll(pred_traj[:, :cur_time], gt_traj[:, :cur_time], torch.from_numpy(bev_seg[1:]).unsqueeze(0))
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] += obj_box_coll.max().item()

                    
                    rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0 # Note: ego_seg[0:1] is used to get the first frame of the ego segmentation
                    out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1:cur_time+1] == 1)).sum() > 0


                    if out_of_drivable and not rec_out:
                        metric_dict['plan_boundary_{}s'.format(i+1)] += 1

                    if out_of_drivable or obj_box_coll.max().item() > 0:
                        print(f"Predicted Trajectory: {pred_traj.numpy()}"
                               + f"Ground Truth Trajectory: {gt_traj.numpy()}" 
                               + f"Object Collision for {cur_time}s: {obj_coll}, Box Collision: {obj_box_coll}" 
                               + f"Out of Drivable Area for {cur_time}s: {((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1:cur_time+1] == 1)).sum(axis=(-1,-2))}, Recovery: {rec_out} "
                            )
                    
                if visualize:
                    visualize_bev(bev_seg, drivable_seg, ego_seg, gt_traj.numpy(), pred_traj.numpy(), obj_box_coll.numpy(), out_of_drivable, data, vis_path)

            else:
                print('Token is skipped' )
            pbar.update(1)
        except Exception as e:
            print(f"EVALUATION_ERROR token={data.get('token', 'unknown')} error={e!r}", flush=True)
            pbar.update(1)


def main(args):
    pred_path = args.pred_path
    discrete_coords = args.discrete_coords

    if discrete_coords > 0:
        print(f"Using discrete coordinates with resolution {discrete_coords}")
    
    if pred_path[-1] != '/':
        pred_path += '/'



    print('The current pred_path', pred_path)
    anno_path = args.anno_path
    key_infos = pickle.load(open(osp.join(args.base_path, anno_path), 'rb'))
    preds = dict()
    for data in key_infos['infos']:
        if os.path.exists(pred_path+data['token']):
            with open(pred_path+data['token'],'r',encoding='utf8')as f:
                full_match = None
                pred_data = json.load(f)
                if 'pure' in pred_path or 'vis3dpos' in pred_path:
                    traj = pred_data[0]['A'][0]
                    full_match = re.search(r'\[\((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', traj)
                elif 'spacedrive'in pred_path:
                    traj = pred_data[0]['A']
                    print(traj)
                    try:
                        full_match = re.search(r'\[<POS_INDICATOR>\(([\d\.-]+, [\d\.-]+)\)\s*([\s\S]*<POS_INDICATOR>\([\d\.-]+, [\d\.-]+\)\s*)*', traj)
                    except Exception as e:
                        print(pred_data, traj, e)
                else :
                    traj = pred_data[0]['A']
                    try:
                        full_match = re.search(r'\[<tool_call>\(([\d\.-]+, [\d\.-]+)\)(, <tool_call>\([\d\.-]+, [\d\.-]+\))*', traj)
                    except Exception as e:
                        print(pred_data, traj, e)
                # print(full_match.group(0) if full_match else f"No match found, the trajectory is: {traj}")

                if full_match:
                    coordinates_matches = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full_match.group(0))
                    coordinates = [tuple(map(float, re.findall(r'-?\d+\.\d+', coord))) for coord in coordinates_matches]
                    coordinates_array = np.array(coordinates)

                    if discrete_coords > 0:
                        coordinates_array = np.floor(coordinates_array / discrete_coords) * discrete_coords
                    preds[data['token']] = coordinates_array


    metric_dict = {
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
        'plan_boundary_1s':0, 
        'plan_boundary_2s':0, 
        'plan_boundary_3s':0, 
        'l2_1s': 0,
        'l2_2s': 0,
        'l2_3s': 0,
        'samples':0,
        # create a empty tensor for storing all trajectories. Shape shoule be (num_samples, 6, 2)
        'all_pred_traj':np.empty((0, 6, 2), dtype=np.float32),
        'all_gt_traj':np.empty((0, 6, 2), dtype=np.float32),
    }

    num_threads = args.num_threads  
    total_data = len(key_infos['infos'])
    data_per_thread = total_data // num_threads
    threads = []
    lock = threading.Lock()
    pbar = tqdm(total=total_data)
    for i in range(num_threads):
        start = i * data_per_thread
        end = start + data_per_thread
        if i == num_threads - 1:
            end = total_data  
        thread = threading.Thread(target=process_data, args=(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric, False, pred_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    pbar.close()    
    for k in metric_dict:
        if k != "samples" and k != "all_pred_traj" and k != "all_gt_traj" and 'command' not in k:
            print('number of samples:', metric_dict["samples"])
            if 'plan' in k:
                metric_dict[k] = metric_dict[k] * 100
            print(f"""{k}: {metric_dict[k]/metric_dict["samples"]}""")

    all_pred_traj = metric_dict['all_pred_traj']
    all_gt_traj = metric_dict['all_gt_traj']
    # do same data distribution analysis based on each of the 6 way points
    for i in range(6):
        print(f"Analyzing waypoint {i+1}...", '-' * 30)
        pred_traj = all_pred_traj[:, i, :2]
        gt_traj = all_gt_traj[:, i, :2]

        # the distribution of the x and y coordinates
        print(f"Distribution of pred x coordinates for waypoint {i+1}: {np.mean(pred_traj[:, 0]):.4f} ± {np.std(pred_traj[:, 0]):.4f}")
        print(f"Distribution of gt x coordinates for waypoint {i+1}: {np.mean(gt_traj[:, 0]):.4f} ± {np.std(gt_traj[:, 0]):.4f}")

        print(f"Distribution of pred y coordinates for waypoint {i+1}: {np.mean(pred_traj[:, 1]):.4f} ± {np.std(pred_traj[:, 1]):.4f}")
        print(f"Distribution of gt y coordinates for waypoint {i+1}: {np.mean(gt_traj[:, 1]):.4f} ± {np.std(gt_traj[:, 1]):.4f}")

        # number of points in bin size of 0.1 for y, number of points in bin size of 1 for x for prediction
        y_bins = np.arange(np.min(pred_traj[:, 1]), np.max(pred_traj[:, 1]) + 0.1, 0.1)
        x_bins = np.arange(np.min(pred_traj[:, 0]), np.max(pred_traj[:, 0]) + 1, 1)
        y_hist, _ = np.histogram(pred_traj[:, 1], bins=y_bins)
        x_hist, _ = np.histogram(pred_traj[:, 0], bins=x_bins)
        print(f"Histogram of pred y coordinates for waypoint {i+1}: {y_hist}, {y_bins}")
        print(f"Histogram of pred x coordinates for waypoint {i+1}: {x_hist}, {x_bins}")

        # for gt
        y_bins_gt = np.arange(np.min(gt_traj[:, 1]), np.max(gt_traj[:, 1]) + 0.1, 0.1)
        x_bins_gt = np.arange(np.min(gt_traj[:, 0]), np.max(gt_traj[:, 0]) + 1, 1)
        y_hist_gt, _ = np.histogram(gt_traj[:, 1], bins=y_bins_gt)
        x_hist_gt, _ = np.histogram(gt_traj[:, 0], bins=x_bins_gt)
        print(f"Histogram of gt y coordinates for waypoint {i+1}: {y_hist_gt}, {y_bins_gt}")
        print(f"Histogram of gt x coordinates for waypoint {i+1}: {x_hist_gt}, {x_bins_gt}")

    # visualize all pred traj in a single plot and save it
    vis_path = './vis/'
    os.makedirs(vis_path, exist_ok=True)

    # draw all traj lines for prediction
    bev_img = Image.fromarray((np.zeros((1000, 1000))).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(bev_img)
    for i in range(all_pred_traj.shape[0]):
        for j in range(all_pred_traj.shape[1]-1):
            x1 = int((all_pred_traj[i, j, 0] + 50 ) / 0.1)
            y1 = int((all_pred_traj[i, j, 1] + 50 ) / 0.1)
            x2 = int((all_pred_traj[i, j+1, 0]+ 50 ) / 0.1)
            y2 = int((all_pred_traj[i, j+1, 1]  + 50 ) / 0.1)
            draw.line((x1, y1, x2, y2), fill=(255, 128, 0), width=2)
    bev_img.save(osp.join(vis_path, f"all_pred_traj.png"))

    # draw all traj lines for gt
    bev_img = Image.fromarray((np.zeros((1000, 1000))).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(bev_img)
    for i in range(all_gt_traj.shape[0]):
        for j in range(all_gt_traj.shape[1]-1):
            x1 = int((all_gt_traj[i, j, 0] + 50 ) / 0.1)
            y1 = int((all_gt_traj[i, j, 1] + 50 ) / 0.1)
            x2 = int((all_gt_traj[i, j+1, 0]+ 50 ) / 0.1)
            y2 = int((all_gt_traj[i, j+1, 1]  + 50 ) / 0.1)
            draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=2)
    bev_img.save(osp.join(vis_path, f"all_gt_traj.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('--base_path', type=str, default='../data/nuscenes/', help='Base path to the data.')
    parser.add_argument('--pred_path', type=str, default='results_planning_only/', help='Path to the prediction results.')
    parser.add_argument('--anno_path', type=str, default='nuscenes2d_ego_temporal_infos_val.pkl', help='Path to the annotation file.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use.')

    parser.add_argument('--discrete_coords', type=float, default=0, help='resolution of discrete coordinates.')

    args = parser.parse_args()
    
    planning_metric = PlanningMetric(args.base_path)
    main(args)
