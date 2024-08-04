import os
import json
import itertools 
import tqdm
from PIL import Image
import cv2
import numpy as np
import torch
from loguru import logger

import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from segment_anything_cs import sam_model_registry, SamPredictor
from segment_anything_cs.utils.amg import batched_mask_to_box, remove_small_regions
import crowdsam.utils as utils


class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, dataset_root, annot_path, transform):
        self.dataset_root = dataset_root
        self.transform = transform
        img_dir = 'Images'        
        annots = json.load(open( annot_path))
        annotations = annots['annotations']
        images = annots['images']
        self.image_ids = [img['id'] for img in images]
        self.boxes = {}
        for annot in annotations:
            image_id = int(annot['image_id'])
            if image_id not in self.boxes.keys():
                self.boxes[image_id] = []
            self.boxes[image_id].append(annot['bbox'])
        
        self.image_files = [os.path.join(dataset_root, img_dir,img['file_name']) for img in images]    
    def __getitem__(self, item):
        img = Image.open(self.image_files[item])
        w,h = img.size
        boxes = torch.tensor(self.boxes[item])
        boxes = boxes / torch.tensor([w,h,w,h]).unsqueeze(0)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        return img, boxes
    def __len__(self):
        return len(self.image_files)
    
def collate_fn(data):
    images, boxes= zip(*data)
    return images, boxes

@torch.no_grad()
def cache_feature(train_dataloader, sam, max_steps=100, feat_size=40, patch_size=14, debug=False):
    """
    Caches image embeddings for the SAM model from the training data.

    Args:
    - train_dataloader (DataLoader): DataLoader providing the training data.
    - sam (SamPredictor): The SAM model instance.
    - max_steps (int): Maximum number of steps to cache embeddings.
    - feat_size (int): Feature size (default 40).
    - patch_size (int): Patch size (default 14).
    - debug (bool): If true, enables debugging mode.

    Returns:
    - cache (list): List of cached features including image embeddings, DINO features, target boxes, image dimensions, and masks.
    """
    # Initialize dataloader iterator
    dataloader_iter = iter(train_dataloader)
    logger.info('Start caching SAM\'s image embeddings for training.. (This will take several seconds) ')
    cache = []

    for step in tqdm.tqdm(range(0, max_steps)):
        # Get the next batch of data
        data = next(dataloader_iter)
        imgs, target_boxes = data

        # Select one image for training
        image = imgs[0]
        image_np = np.array(image)
        target_boxes = target_boxes[0] 
        img_height, img_width = image_np.shape[:2]

        # Scale the target boxes to the image dimensions
        scale = torch.tensor([img_width, img_height, img_width, img_height]).unsqueeze(0)
        target_boxes = target_boxes * scale

        # Set the image in the SAM model
        sam.set_image(image_np)

        # Apply transformations to the target boxes
        prompt_boxes = sam.transform.apply_boxes(target_boxes.numpy(), sam.original_size)
        prompt_boxes = torch.tensor(prompt_boxes)[:, None, :].cuda()

        # Extract only low resolution masks
        masks_list = []
        masks = predict_torch(sam, boxes=prompt_boxes, multimask_output=False)[0].cpu()
        masks = (masks > sam.model.mask_threshold)

        assert len(masks) == len(target_boxes)
        
        # Get DINO features
        dino_features = sam.dino_feats

        # Cache the image embeddings, DINO features, target boxes, image dimensions, and masks
        cache.append([sam.get_image_embedding().cuda(), dino_features.cuda(), target_boxes, (img_height, img_width), masks.cuda()])

    return cache



def predict_torch(
        predictor,
        point_coords = None,
        point_labels = None,
        boxes = None,
        multimask_output = True,
    ) :
    '''
    This function is copied from segment anything predictor and modified for 
    '''
        #we modify the definition of point_labels here to define pos point point label = 1 , neg point label = 0
    if point_coords is not None:
        assert len(point_coords) == len(point_labels)
        points = (point_coords,  point_labels)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=None,
    )

    # Predict masks
    low_res_masks, iou_predictions, cls_scores = predictor.model.mask_decoder(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        dino_feats = predictor.dino_feats,
    )
    #B,C,H,W -> B,H,W,C for MLP to process
    return low_res_masks, iou_predictions, cls_scores


@logger.catch()
def compute_loss(low_res_masks:torch.Tensor, 
                 iou_predictions:torch.Tensor,
                 cls_logits:torch.Tensor,
                 target_masks:torch.Tensor,
                 fg_mask:torch.Tensor,
                 num_pos_sample:int, 
                 debug=False):
    """
    Computes the loss for the given predictions and targets.

    Args:
    - low_res_masks (torch.Tensor): Low-resolution predicted masks.
    - iou_predictions (torch.Tensor): Predicted IoU values.
    - cls_logits (torch.Tensor): Classification logits.
    - target_masks (torch.Tensor): Sampled  GT masks.
    - fg_mask (torch.Tensor): Foreground GT mask (including all GT).
    - num_pos_sample (int): Number of positive samples.
    - debug (bool): If true, enables debugging mode.

    Returns:
    - loss_dict (dict): Dictionary containing various losses.
    """
   
    # Keep masks predicted for positive prompts only
    low_res_masks = low_res_masks[:num_pos_sample]

    # Compute Dice Loss between predicted and target masks
    dc_loss = utils.dice_loss(low_res_masks, target_masks.unsqueeze(1).float())
    iou_pred_target = utils.mIoU(low_res_masks, target_masks.unsqueeze(1).float())

    # Find the index of the mask with the minimum Dice loss
    max_sim_ind = dc_loss.min(dim=1)[1]
    num_masks = low_res_masks.shape[0]

    # Compute classification loss (Dice Loss) for the foreground mask
    dice_loss = utils.dice_loss(cls_logits, fg_mask).mean()

    # Prepare IoU target tensor
    iou_target = torch.zeros_like(iou_predictions)
    iou_target[torch.arange(num_masks)] = iou_pred_target
    cls_loss = F.mse_loss(iou_predictions, iou_target, reduction='none').sum(dim=[1])

    # Separate classification loss for positive and negative samples
    pos_cls_loss = cls_loss[:num_pos_sample].mean()
    neg_cls_loss = cls_loss[num_pos_sample:].mean()

    # Build loss dictionary
    loss_dict = {
        'pos_cls_loss': pos_cls_loss,
        'neg_cls_loss': neg_cls_loss,
        'dice_loss': dice_loss
    }
    # Log the loss values if in debug mode
    if debug:
        logger.debug(f"Loss dict: {loss_dict}")

    return loss_dict
def train_loop(data_loader,  predictor, optimizer, max_steps=3000, neg_factor=3, n_shot=10, pos_sample=20, clip_grad=0.1, debug=False):
    neg_sample = int(neg_factor * pos_sample)
    
    cache = cache_feature(data_loader,predictor, n_shot, debug=debug)    
    for step in range(0, max_steps):
        #Extract sample according to step
        sample_idx = step%len(cache)
        features, dino_features, _,  (img_height, img_width), target_masks = cache[sample_idx]
        # num_select_sample = min(pos_sample, len(target_masks))    
            
        #Shuffule and sample the targets to avoid OOM  
        sample_ind = np.random.choice(np.arange(len(target_masks)), pos_sample,replace=True)
        fg_mask = target_masks.any(dim=0)#.cpu()
        target_masks = target_masks[sample_ind,0]
        #Sample positive point prompts
        pos_point_coords = []
        for mask in target_masks:
            coords = mask.nonzero()[:, [1,0]] # convert to xy
            select_point = coords[np.random.randint(0, len(coords))].view(-1,2)
            pos_point_coords.append(select_point)        
        pos_point_coords = torch.cat(pos_point_coords, dim=0)#M,2 : x,y
    
        #Sample negative point prompts
        scale = min(256/ img_height, 256/ img_width)
        neg_coords = (~fg_mask)[0,:int(scale*img_height), :int(scale*img_width)].nonzero()[:,[1,0]]
        neg_point_coords = neg_coords[np.random.choice(np.arange(len(neg_coords)),neg_sample,replace=False)].view(-1,2)
        
        #Cat the prompts and convert variables to cuda
        point_coords = torch.cat([pos_point_coords, neg_point_coords], dim=0)
        point_coords = point_coords / scale
        prompt_coords_trans = predictor.transform.apply_coords(point_coords.unsqueeze(1).cpu().numpy(), (img_height, img_width))
        prompt_coords_trans = torch.from_numpy(prompt_coords_trans).cuda()
        prompt_labels = torch.ones_like(prompt_coords_trans)[:,:,0].cuda()
        target_masks = target_masks.cuda()
        fg_mask = fg_mask[:,:int(scale*img_height), :int(scale*img_width)].float().cuda()

        predictor.features = features.cuda()
        predictor.dino_feats = dino_features.cuda()

        cls_logits = predictor.predict_fg_map((img_height, img_width))[0]
        cls_logits = cls_logits[:,:int(scale*img_height), :int(scale*img_width)]

        low_res_masks, iou_predictions, cls_scores = predict_torch(predictor, prompt_coords_trans, prompt_labels)
        loss_dict = compute_loss(low_res_masks,
                                 iou_predictions * cls_scores.sigmoid()[:,:,0], 
                                 cls_logits,
                                target_masks= target_masks,
                                fg_mask = fg_mask,
                                num_pos_sample=pos_sample,
                                debug = debug)

        total_loss = sum([v for k,v in loss_dict.items()])
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()  
        loss_dict_data = {k:round(float(v.data),3) for k,v in loss_dict.items()}
        
        if step %100 == 0:
            output_str = f"step: {step}/{max_steps} "
            for k,v in loss_dict_data.items():
                output_str += f"{k}: {v} "
            logger.info(output_str, flush=True)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="可写可不写，只是在命令行参数出现错误的时候，随着错误信息打印出来。")
    parser.add_argument('--config_file', default='configs/crowdhuman.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    #fixed config
    model_arch = 'dino'
    mode = 'training'
    #data related
    #===========>fix seed for reproduction
    np.random.seed(config['train']['seed'])
    torch.random.manual_seed(config['train']['seed'])
    #===========>set arguments or options 
    #datasets
    sam = sam_model_registry[config['model']['sam_model']](checkpoint= config['model']['sam_checkpoint'],n_class=config['model']['n_class'])
    sam.cuda()
    
    if model_arch == 'dino':
        print('usning dino as backbone')
        dino_repo = config['model']['dino_repo']
        model = torch.hub.load(dino_repo, config['model']['dino_model'],source='local',pretrained=False).cuda()
        model.load_state_dict(torch.load(config['model']['dino_checkpoint'],weights_only=True))
    predictor = SamPredictor(sam, model)
    learnable_params = itertools.chain(predictor.model.mask_decoder.parallel_iou_head.parameters(),
                                                         predictor.model.mask_decoder.point_classifier.parameters(),
                                                         predictor.model.mask_decoder.dino_proj.parameters(),
                                                         )
    size = 0
    for param in predictor.model.mask_decoder.parameters():
        param.requires_grad = False
    for param in learnable_params:
        param.requires_grad_()
        size += param.numel()
    print('total learnable parameters:', size)
    optimizer = torch.optim.AdamW(params=predictor.model.mask_decoder.parameters()
                                  , lr= config['train']['lr'], weight_decay= config['train']['weight_decay'])
    # prompts = generate_prompts(patch_h, patch_w).cuda()
    if mode == 'training':
        dataset = CrowdHuman(config['data']['dataset_root'],config['data']['train_file'], transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=True, num_workers=0, drop_last=False,collate_fn=collate_fn)
        train_loop(train_dataloader, predictor, optimizer, config['train']['steps'], config['train']['neg_factor'], config['train']['n_shot'], config['train']['samples_per_batch'], debug=args.debug)
        torch.save(predictor.model.mask_decoder.state_dict(), config['train']['save_path'])
        logger.info('done')
   