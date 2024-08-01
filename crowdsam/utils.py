from typing import Any, Dict, Generator, ItemsView, List, Tuple
import cv2
import os
import logging
import functools
import time
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops 
from torchvision.ops.boxes import box_area

from PIL import Image
from matplotlib import pyplot as plt
from .coco_names import coco_classes
from itertools import product

import argparse
import yaml
# from reliability.Distributions import Weibull_Distribution,Mixture_Model
# from reliability.Fitters import Fit_Weibull_Mixture
#general utils
# data_meta = [dataset_path, n_class, categories]
data_meta = {'crowdhuman':["./datasets/crowdhuman", 1, {1:'person'}],
             'occhuman':["./datasets/OCHuman", 1, {1:'person'}],
             'coco_occ':["./datasets/coco", 80, coco_classes],
             'coco':["./datasets/occ_coco", 80, coco_classes], 
             }
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def convert_value(value):
    # Attempt to convert the value to the most appropriate data type
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
def modify_config(config_file, options):
    assert len(options) % 2 == 0
    keys = options[0::2]
    values = options[1::2]
    for key, value in zip(keys, values):
        keys = key.split('.')
        d = config_file
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = convert_value(value)
    return config_file

def visualize_result(image, result, class_names, save_path, conf_thresh=0.001, FP_ind = None, FN_ind = None):
    # plt.gca().set_title(f'gt boxes #GT {len(gt_boxes)}')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TP_ind = np.setdiff1d(np.arange(len(result['boxes'])), FP_ind)
    for i in range(len(result['boxes'])):
        box = result['boxes'][i]
        # mask = np.array(result['masks'][i])
        score =round(float(result["scores"][i]),3)
        class_id = result['categories'][i]
        if score  < conf_thresh:
            continue
        color = [255,255,0] #if i in FP_ind else  [255,255,0]
        if FP_ind is not None:
            if i in FP_ind:
                color = [0,0,255]
        class_name =class_names[class_id+1]
        image = draw_box(image, box,f"{class_name}:{score}",color=color)
        # image = draw_mask(image, mask, random_color=False)
    if FN_ind is not None:    
        for i in FN_ind:
            image = draw_box(image, result['gt_boxes'][i], color=[255,0,0])
    cv2.imwrite(save_path,image)
    
def generate_crop_boxes(
    crop_box: Tuple[int, ...], n_layers: int, overlap_ratio: float, 
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    sx0,sy0,sx1,sy1 = crop_box
    im_h, im_w = sy1-sy0, sx1-sx0
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append(crop_box)
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i  ) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i  ) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0 + sx0 , y0 + sy0, sx0 + min(x0 + crop_w, im_w), sy0 + min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

def resize_image(image, max_size):  
    # Maybe upscale image when image longest side is lower  than max_size
    # This modification is slightly better than only downscale (41.3->41.7 on midval)
    h,w = image.shape[:2]
    r = min(max_size /w , max_size/h)
    # r = min( r, 1) #not allow image to be scaled up
    h, w = (int(r*h), int(r*w))
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (w,h))
    elif isinstance(image, torch.Tensor):
        assert image.ndim==2 or image.ndim==3
        if image.ndim==2:
            image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (h,w))[0,0]
        elif image.ndim==3:
            image = F.interpolate(image.permute(2,0,1).unsqueeze(0), (h,w))[0].permute(1,2,0)
    return image, r


#general utils
def timestamp_to_datetime(ts):
    dt = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)), "%Y-%m-%d %H:%M:%S")
    return dt

@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(save_path, quiet=False):
    from loguru import logger
    import sys
    logger.remove()
    logger.add(f'{save_path}/{timestamp_to_datetime(time.time())}.log', format="{time}-{level}-{message}", filter="my_module",
                retention="10 days", level="DEBUG")
    logger.add(sys.stdout, format="{time}-{level}-{message}", filter="my_module", level="INFO")
    return logger

#box utils
def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int], downscale: float) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes/ downscale + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int], downscale:float) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points/ downscale + offset


def uncrop_masks(
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    w,h =  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    masks = F.interpolate(masks.unsqueeze(0).float(),(h,w))[0].bool()
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)

def apply_box_offsets(boxes, box_delta):
    # boxes_pred[:,:2] = boxes_pred[:,:2] + box_delta[:,:2]*boxes_pred[:,2:].detach()
    box_xy = boxes[:,:2] + box_delta[:,:2]*boxes[:,2:]
    box_wh = boxes[:,2:] * torch.exp(box_delta[:,2:]) 
    boxes = torch.cat([box_xy, box_wh], dim=-1)
    boxes = box_cxcywh_to_xyxy(boxes)
    return boxes
    
def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], downscale, atol: float = 20.0
) -> torch.Tensor:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box, downscale).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)


def box_xyxy_to_cxcywh(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    x0, y0, x1, y1 = bbox.unbind(-1)
    new_bbox = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(new_bbox, dim=-1)
def clustering_matched_points(coords, num_cluster):
    assert num_cluster >=1
    if len(coords)==0:
        return torch.zeros(0,2)
    num_cluster = min(len(coords), num_cluster)
    y_pred = KMeans(n_clusters= num_cluster, n_init='auto', random_state=9, ).fit_predict(coords)
    centroids = []
    for i in range(y_pred.max()+1):
        centroid = coords[ y_pred==i].float().mean(dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids,dim=0)
    return centroids

def sigmoid_focal_loss(
    preds,
    targets,
    weight=None,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    avg_factor: int = None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        avg_factor (int): Average factor that is used to average
            the loss. Default: None.

    Returns:
        torch.Tensor: The computed sigmoid focal loss with the reduction option applied.
    """
    preds = preds.float()
    targets = targets.float()
    p = torch.sigmoid(preds)
    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # if weight is not None:
    #     assert weight.ndim == loss.ndim

    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss.sum(dim=-1).mean()


def composite_clustering(coords, num_prompts, device):
    prompt_coords_list = []
    for num_prompt in num_prompts:
        #padding here seems does not help
        prompt_coords_i = clustering_matched_points(coords, num_prompt).to(device)
        prompt_coords_list.append(prompt_coords_i)
    return prompt_coords_list

#########drawing utils
def draw_point(point, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([30, 144, 255])
    image = cv2.circle(image, (int(point[0]), int(point[1])), 2, [255,0,0])
    # mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255).astype('uint8')
    # masked_image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)
    return image
def draw_mask(image,mask, random_color=False):
    mask = np.array(mask, dtype=np.int32)
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0) * 255
    else:
        color = np.array([30, 144, 255])
    h,w = mask.shape
    mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1) ).astype('uint8')
    masked_image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)
    return masked_image
def draw_box(image, box, score=None, color=[255,255,0] ):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    if score is not None:
        cv2.putText(image, str(score), (int(box[0]),int(box[1])),color=color,fontScale=font_scale, fontFace=font, thickness=thickness)
        
        # ax.text(box[0],box[1], str(round(score,3)), color='green')
    cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), color=color)
    return image
    
    # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))   
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box,  ax, color='green', score=None, ):
    if score != None:
        ax.text(box[0],box[1], str(round(score,3)), color='green')
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))    

def load_img_and_annotation(dataset_path, annots, dataset, id=0):
    img_meta = annots['images'][id]
    if dataset=='crowdhuman':
        img_path = os.path.join(dataset_path,'Images', img_meta['file_name'])
    elif dataset == 'coco':
        img_path = os.path.join(dataset_path,'val2017', img_meta['file_name'])
    elif dataset == 'coco_occ':
        img_path = os.path.join(dataset_path,'occ2017', img_meta['file_name'].split('/')[-1])
    elif dataset == 'occhuman':
        img_path = os.path.join(dataset_path,'images', img_meta['file_name'])
    elif dataset == 'mineapple':
        img_path = os.path.join(dataset_path,'images', img_meta['file_name'])
    else:
        raise NotImplementedError
    #load image
    image_cv = cv2.imread(img_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    bboxes = np.array([ annot['bbox'] for annot in annots['annotations'] if annot['image_id'] ==img_meta['id']])
    bboxes[...,2:] += bboxes[...,:2]
    img_id = img_meta['id']    
    return image_cv, bboxes, img_id 

def is_validbox(box):
    #[x_1,y_1,x_2,y_2]
    return box[2] > box[0] and box[3] > box[1]

def select_box(boxes, mode ='big'):
    wh = boxes[:,2:] - boxes[:,:2]
    area =wh[:,0] * wh[:,1]
    if mode == 'big':
        ind = area.argmax()
        return int(ind)
    elif mode == 'small':
        ind = area.argmin()
        return int(ind)
    elif mode =='random':
        ind = int(np.random.choice(np.arange(len(area)), 1))
    else:
        raise NotImplementedError


def coords2mask(coords):
    #N, 2 (x,y)
    width,height = coords.max(dim=0)[0]
    mask = torch.zeros(int(height),int(width))
    mask[coords[:,1], coords[:,0]] = True
    import pdb;pdb.set_trace()
    return mask
def mask2coord(mask):
    coords= mask.nonzero()
    return coords

def mask_iou_nms(boxes, scores, mask_preds, threshold):  
    """  
    boxes: 边界框的坐标，形状为 (N, 4)，N 是边界框的数量。坐标按照 (y_min, x_min, y_max, x_max) 格式给出。  
    scores: 每个边界框的得分，形状为 (N,)。  
    mask_preds: 每个边界框对应的掩码预测，形状为 (N, H, W)，其中 H 和 W 是掩码的高度和宽度。  
    threshold: IoU 阈值，用于决定哪些边界框应该被抑制。  
    """  
    if mask_preds.numel() == 0:
        return []
    # 首先，对边界框和得分进行排序，以便于选择得分最高的边界框  
    mask_preds = F.interpolate(mask_preds.float().unsqueeze(0), (150, 150))[0].bool()
    
    sorted_indices = np.argsort(-scores).tolist()
    sorted_boxes = boxes[sorted_indices]  
    sorted_scores = scores[sorted_indices]  
    sorted_masks = mask_preds[sorted_indices]  
    # 初始化保留的边界框和掩码  
    keep = []  
    # 对每个边界框进行循环  
    for i in range(sorted_boxes.shape[0]):  
        if len(keep) == 0:
            keep.append(sorted_indices[i])
            continue
        # 如果这个边界框的得分低于当前保留的边界框的得分，或者它们的 IoU 大于阈值，那么抑制这个边界框  
        # iou = box_iou(sorted_boxes[i].unsqueeze(0),boxes[keep])
        # if  torch.any(iou >=0.7):  
        #     continue  
        # if  torch.all(iou <0.3):  
        #     keep.append(sorted_indices[i])  
        #     continue 
        
        if  torch.any(coverage(sorted_masks[i].unsqueeze(0), mask_preds[keep]) > threshold):
            continue
        else:
            keep.append(sorted_indices[i]) 
        # 否则，将这个边界框添加到保留的列表中  
    # 将保留的边界框和掩码返回  
    return np.array(keep)

def coverage(mask1, mask2):  
    """  
    计算两个掩码的 coverage  
    """  
    intersection =(mask1 * mask2).sum([-1,-2])  
    coverage_1 = intersection/ mask1.sum([-1,-2])
    coverage_2 = intersection/ mask2.sum([-1,-2])
    # union = torch.logical_or(mask1, mask2).sum()  
    # iou = intersection / union  
    return torch.maximum(coverage_1, coverage_2)

def mask_iou(mask1, mask2):  
    """  
    计算两个掩码的 IoU。  
    """  
    intersection = torch.logical_and(mask1, mask2).sum([-1,-2])  
    union = torch.logical_or(mask1, mask2).sum([-1,-2])  
    iou = intersection / union  
    return iou

#evaluation utils
def evaluate_boxes(pred_boxes: np.ndarray, pred_scores:  np.ndarray, gt_boxes:  np.ndarray, iou_thresh: float):
    assert len(pred_scores) >= 0
    assert len(pred_boxes) == len(pred_scores)
    assert len(gt_boxes) >= 0
    if len(pred_boxes) == 0:
        return 0, 0, [], []
    pred_boxes = torch.tensor(pred_boxes)
    pred_scores = torch.tensor(pred_scores)
    gt_boxes = torch.tensor(gt_boxes)
    # Sort the predictions by scores
    _, ind = pred_scores.sort(descending=True)
    pred_boxes = pred_boxes[ind]
    pred_scores = pred_scores[ind]
    match_mask = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    iou = ops.box_iou(pred_boxes, gt_boxes)  # num_pred, num_gt
    prec = []
    TP = 0
    FP = 0
    FP_list = []

    for i in range(iou.shape[0]):
        matched = False
        for j in range(iou.shape[1]):
            if iou[i, j] > iou_thresh and not match_mask[j]:
                match_mask[j] = True
                TP += 1
                prec.append(TP / (TP + FP))
                matched = True
                break
        if not matched:
            FP += 1
            FP_list.append(ind[i].item())  # Use item() to convert Tensor to int

    if len(gt_boxes) > 0:
        precision = sum(prec) / len(gt_boxes) if prec else 0
        recall = TP / len(gt_boxes)
    else:
        precision = 0
        recall = 0
    
    FN_list = (~match_mask).nonzero(as_tuple=False).flatten().tolist()
    return precision, recall, FP_list, FN_list

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    #suppose inputs dim to be [1,3,H,W]
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss#.sum() / num_masks
def mIoU(inputs, targets):
    #inputs: B,3,H,W
    #targets: B,1,H,W
    mask_bin = (inputs > 0).float()
    mask_bin = mask_bin.flatten(2) #B,3,HW
    targets = targets.flatten(2) #B,1,HW
    intersection =  (mask_bin * targets).sum(-1) # B, 3
    union = mask_bin.sum(-1) + targets.sum(-1) - intersection
    return intersection/union
def box_iou(boxes1, boxes2) :
    """Modified from ``torchvision.ops.box_iou``

    Return both intersection-over-union (Jaccard index) and union between
    two sets of boxes.

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        Tuple: A tuple of NxM matrix, with shape `(torch.Tensor[N, M], torch.Tensor[N, M])`,
        containing the pairwise IoU and union values
        for every element in boxes1 and boxes2.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The input boxes should be in (x0, y0, x1, y1) format

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        torch.Tensor: a NxM pairwise matrix containing the pairwise Generalized IoU
        for every element in boxes1 and boxes2.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)

def average_metric(result, key):
    metrics = [item[key] for item in result]
    return round(float(sum(metrics) / len(metrics)),3)
#load img and annotations

def mask_to_rle_numpy(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Convert tensor to numpy array for processing
    tensor_np = tensor.cpu().numpy()
    
    # Put in fortran order and flatten h,w
    b, h, w = tensor_np.shape
    tensor_np = np.asfortranarray(tensor_np).reshape(b, -1)

    # Compute change indices
    diff = tensor_np[:, 1:] != tensor_np[:, :-1]
    change_indices = np.nonzero(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[1][change_indices[0] == i]
        cur_idxs = np.concatenate([
            np.array([0], dtype=cur_idxs.dtype),
            cur_idxs + 1,
            np.array([h * w], dtype=cur_idxs.dtype)
        ])
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor_np[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        out.append({"size": [h, w], "counts": counts})
    return out