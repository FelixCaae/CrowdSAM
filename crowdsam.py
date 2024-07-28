import torch
import torchvision.transforms as T
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt
import utils
from PIL import Image
import logging 
import torch.nn as nn
from loguru import logger
import math
from torchvision.ops.boxes import batched_nms, box_area
from segment_anything.utils.amg import (
    MaskData,
    batch_iterator,
    batched_mask_to_box,
    calculate_stability_score,
    generate_crop_boxes,
    remove_small_regions,
    mask_to_rle_pytorch,
    coco_encode_rle,
)
class Resize(nn.Module):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, image):
        # 计算需要添加的padding数量  
        h_p = (self.patch_size - image.size[1] % self.patch_size) % self.patch_size  
        w_p = (self.patch_size - image.size[0] % self.patch_size) % self.patch_size  
        w,h = image.size        
        # 使用numpy的pad函数添加padding  
        # padded_image = np.pad(image, ((0,height_padding), (0,width_padding),(0,0)), mode='constant',constant_values=0)
        # new_image =  F.interpolate(image, (h+height_padding, w+width_padding))
        return image.resize((w_p+w, h_p+h), Image.BILINEAR)
    
class CrowdSAM():
    vis_img_id = 0
    def __init__(self,config,logger):
        self.device = torch.device(config['environ']['device'])
        
        #hard-coded setting        
        legacy_mode = False
        self.train_free=False
        #model 
        dino_model =  torch.hub.load(config['model']['dino_repo'],config['model']['dino_model'], source='local').to(self.device)
        self.predictor =self.load_sam_model(config['model']['sam_model'], 
                                            config['model']['sam_arch'],
                                            config['model']['sam_checkpoint'], 
                                            config['model']['sam_adapter_checkpoint'],
                                            dino_model, 
                                            config['model']['n_class'])
        
        self.mask_selection = config['test']['mask_selection']
        self.apply_box_offsets = config['test']['apply_box_offsets'] #apply_box_offsets
        #eps settings
        self.max_prompts =config['test']['max_prompts']
        self.filter_thresh = config['test']['filter_thresh']
        #other test settings     
        self.max_size =config['test']['max_size']#resize image to this
        self.grid_size =config['test']['grid_size']
        self.pred_iou_thresh =  config['test']['pred_iou_thresh'] #iou_score filter
        # self.score_thresh = kwargs.get('score_thresh')
        self.stability_score_thresh = config['test']['stability_score_thresh']
        self.stability_score_offset = config['test']['stability_score_offset']
        self.box_nms_thresh = config['test']['box_nms_thresh']
        self.points_per_batch = config['test']['points_per_batch']
        self.crop_n_layers = config['test']['crop_n_layers']
        self.crop_nms_thresh = config['test']['crop_nms_thresh']
        self.crop_overlap_ratio = config['test']['crop_overlap_ratio']
        self.min_mask_region_area = config['test']['min_mask_region_area']
        self.pos_sim_thresh = config['test']['pos_sim_thresh']
        self.output_rles = config['test']['output_rles']
        if legacy_mode:
            self.patch_size = config['model']['patch_size'] # vit_l for dino
            self.feat_size = feat_size
            self.feat_dim = feat_dim # vit_l for dino
            self.transform = T.Compose([
                Resize(patch_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        if config['model']['trainfree']:
            self.train_free = True
            self.patch_size = 14
            self.image_encoder = dino_model
            self.ref_feature = torch.load(config['model']['ref_feature'])['f'].mean(dim=0).to(self.device)#.mean(dim=0)
            self.alpha = config['model']['score_fusion']
        #  if not config['model']['trainfree'] else True
            
        #original sam automask generater args
     
        #other parameters


    #load sam model according to specifiedd arguments
    def load_sam_model(self, sam_model, sam_arch, sam_checkpoint, sam_adapter_checkpoint, dino_model, n_class):
        if sam_arch =='crowdsam':
            from segment_anything_cs import sam_model_registry, SamPredictor
            # from per_segment_anything_person_specific import sam_model_registry,SamPredictor
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint,n_class=n_class)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint),strict=False)
            predictor = SamPredictor(sam, dino_model)

        elif sam_arch =='sam_hq':
            from segment_anything_hq import sam_model_registry, SamPredictor
            # from per_segment_anything_person_specific import sam_model_registry,SamPredictor
            sam_model = sam_model[2:]
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint,n_class=n_class)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint), strict=False)
            predictor = SamPredictor(sam, dino_model)
        elif sam_arch == 'mobile_sam':
            from mobile_sam import sam_model_registry, SamPredictor
            sam_model = sam_model[6:]
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint))
            predictor = SamPredictor(sam, dino_model)
        else:
            from segment_anything import sam_model_registry,SamPredictor
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
            predictor = SamPredictor(sam)

        sam = sam.to(self.device)
        return predictor
    


    def crop_image(self, image, crop_box, sim_map=None):
        
        #crop and then resize image
        x0,y0,x1,y1 = crop_box
        #adapt crop region guided by semantic prior
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        self.orig_image = image
        #crop area represents the area of image to operate
        image = image[y0:y1, x0:x1,:]
        image,r = utils.resize_image(image, self.max_size)
        self.image = image
        self.downscale = r

    @torch.no_grad()
    def generate(self, image: np.ndarray):
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
        Returns:
           predictions (MaskData): The dict that stores predictions in np.ndarray format
             boxes: xyxy format
             scores: >0
             masks: only exists when visualize is enabled
        """
        # Generate masks
        mask_data = self._generate_masks(image)
        # Filter small disconnected regions and holes in masks
        return mask_data

    def _generate_masks(self, image):
        img_size = np.array(image).shape[:2]
        #===============> Step 1. Genereate crops         
        crop_boxes, layer_idxs = generate_crop_boxes(
            img_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        layer_idxs = np.ones(len(crop_boxes))    
        data = MaskData()
        #===============> Step 2. Process Crops   
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image,  crop_box)
            if crop_data is not None:
                data.cat(crop_data)
            del crop_data
            logger.debug(f"#{layer_idx} crop area {str(crop_box)}")
        # Remove duplicate masks between crops
        if len(crop_boxes) > 1 and 'crop_boxes' in data._stats and len(data['crop_boxes']) > 0:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
            del data['crop_boxes']
        if len(data._stats.keys()) > 0:
            del data['iou_preds']
        else:
            data['boxes'] = torch.zeros(0, 4)
            data['scores'] = torch.zeros(0, 4)
        if 'rles' in data._stats:
            data["segmentations"] = [coco_encode_rle(rle) for rle in data["rles"]] 
        else:
            data['segmentations'] = []
        # if self.visualize:
        #     self.visualize_inter_results(np.array(image), crop_boxes, self.adapted_boxes, sim_map)
        # del self.adapted_boxes
        data.to_numpy()    
        return data
    
    def _process_crop(self, image, crop_box):
        #===============> Step 5. Re-extract features 
        #Notice: self.image is a cropped image
        # crop_box = self.shrink_crop_box(crop_box, sim_map, margin=100)
        # self.adapted_boxes.append(crop_box)
        self.crop_image(image, crop_box)
        self.predictor.set_image(self.image)
        orig_h, orig_w = self.orig_image.shape[:2]
        img_size = torch.tensor(self.image.shape[:2])
        

        if not self.train_free:
            # dino_feats = self.predictor.dino_feats
            feat_size = (img_size * min(self.grid_size/img_size)).int()         
            sim_map = self.predictor.predict_fg_map(img_size)
            sim_map = torch.nn.functional.interpolate(sim_map, (self.grid_size, self.grid_size), mode='bilinear')
            sim_map = sim_map.sigmoid().max(dim=1)[0]
            sim_map = sim_map[0,:feat_size[0], :feat_size[1]]
            sim_thresh =self.pos_sim_thresh
            visualize = False
            if visualize:
                # plt.imshow(self.orig_image)
                sim_map_vis = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), (orig_h,orig_w),mode='bilinear')[0,0].cpu()
                sim_map_vis = (sim_map_vis*255).int()
                img = utils.draw_mask(self.orig_image, sim_map_vis)
                plt.imshow(img)
                plt.savefig('test.jpg')
        else:
        #used when  dino_feats = self.predictor.dino_feats
            transform = T.Compose([
                    T.Resize((1022, 1022)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            dino_feats = self.extract_features(self.image, transform)
            feat_size = dino_feats.shape[:2]
            feature_sim = F.cosine_similarity(self.ref_feature, dino_feats.flatten(0,1)) 
            feature_sim = feature_sim.reshape(*feat_size)
            sim_map = feature_sim
            sim_thresh =  self.pos_sim_thresh
        coords = self.match_ref(sim_map, sim_thresh).cpu()
        inv_factor = torch.tensor([feat_size[1]/self.image.shape[1], feat_size[0]/self.image.shape[0]])
        coords = (coords ) /  inv_factor
        # prompt_coords = utils.composite_clustering(coords, self.num_prompts, self.device)[0]
        points_for_image = coords.cpu().numpy()
        logger.debug(f'len points {len(points_for_image)}')
        #No change here
        data = MaskData()
        red_mask = torch.zeros(*img_size, dtype=torch.bool)
        
        def efficient_batch_iterator(batch_size: int, points):
            ind = np.arange(len(points))
            rand_ind = np.random.choice(ind, len(ind), replace=False)
            points = points[rand_ind]
            cum = 0
            while len(points)>0 and cum < self.max_prompts:
                batch_size = min(len(points), batch_size)
                sel_pts= points[:batch_size]
                cum += len(sel_pts)
                yield sel_pts
                points = points[batch_size:]
                keep = (~red_mask[points[:,1].astype('int'), points[:,0].astype('int')]).numpy()
                points = points[keep]
                
        for points in efficient_batch_iterator(self.points_per_batch, points_for_image):
            if len(points) ==0:
                continue
            batch_data = self._process_batch(points, self.predictor.original_size, crop_box)

            red_mask = (batch_data['masks'][batch_data['iou_preds']> self.filter_thresh]).any(0).cpu()
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()
        
        # The data maybe empty here since prompts are dynamic
        if len(data.items()) == 0:
            return None
        if len(data['masks']) == 0:
            return None

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        
        if self.min_mask_region_area > 0:
            min_mask_region_area = self.min_mask_region_area  #* (self.downscale)**2
            data = self.postprocess_small_regions(
                data,
                min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        #Implement joint classification scores here 
        if self.train_free:
            sim_map_high_res = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), self.image.shape[:2],mode='bilinear')[0,0].cuda()  
            # cls_scores = self.evaluate_cls_scores(data['masks'], sim_map_high_res, clf)
            cls_scores = []
            for mask in data['masks']:
                if mask.sum() > 0:
                    cls_score = sim_map_high_res[mask].mean()
                else:
                    cls_score = -0.5
                cls_score = torch.clamp(cls_score + 0.5,0, 1)
                cls_scores.append(cls_score)
            cls_scores = torch.tensor(cls_scores).to(self.device)
            data['scores'] = data['iou_preds'] ** (1- self.apha)  * cls_scores ** self.alpha
        
        else:
            data['scores'] = data['iou_preds']
            
        if self.output_rles:
            data["rles"] = mask_to_rle_pytorch((utils.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)))
        del data['masks']
            
        #Implement the box_offsets herehere 
            #apply box offsets here
        data["boxes"] = utils.uncrop_boxes_xyxy(data["boxes"], crop_box, self.downscale)
        data['points'] = utils.uncrop_points(data['points'], crop_box, self.downscale)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["boxes"]))])
        if self.apply_box_offsets:
            ext_boxes = utils.apply_box_offsets(data['boxes'], data['box_offsets'])
            data['fboxes'] = ext_boxes
        else:
            data['fboxes'] = data['boxes']
        return data

    @torch.no_grad()
    def extract_features(self, image, transform):
        t = transform(Image.fromarray(image))
        _,h,w = t.shape
        feat_h, feat_w = h//self.patch_size, w//self.patch_size
        features_dict = self.image_encoder.forward_features(t.unsqueeze(0).to(self.device))
        features = features_dict['x_norm_patchtokens'].flatten(0,1)
        feat_size = torch.tensor((feat_h, feat_w))
        return features.reshape(feat_h, feat_w, -1)
    
    def select_mask(self, masks, iou_preds):
        bin_masks = masks  > self.predictor.model.mask_threshold
        if self.mask_selection == 'max_area':
            ind = bin_masks.sum(dim=[-1,-2]).max(dim=-1)[1] # L
        elif self.mask_selection == 'min_area':
            ind = bin_masks.sum(dim=[-1,-2]).min(dim=-1)[1] # L
        elif self.mask_selection == 'max_iou':
            ind = iou_preds.max(dim=-1)[1]
        elif self.mask_selection == 'all':
            return masks.flatten(0,1)
        else:
            raise NotImplementedError
        indices = torch.arange(len(masks)), ind
        return indices
        # 

    def _process_batch(
        self,
        points: np.ndarray,
        im_size,
        crop_box
    ) -> MaskData:
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        masks, iou_preds, cls_scores = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )[:3]
        indices = self.select_mask(masks, iou_preds)
        if not self.train_free:
            conf, categories = cls_scores.max(dim=-1)
            masks, iou_preds, points, categories = masks[indices], iou_preds[indices], torch.as_tensor(points.repeat(1, axis=0)), categories[indices]
        else:
            masks, iou_preds, points,  = masks[indices], iou_preds[indices], torch.as_tensor(points.repeat(1, axis=0))
            categories = torch.zeros(len(masks)).int()
        # return 
        #Feature: Select proper mask according to IoU 
            
        #, (torch.clamp(iou_preds,0) * conf.sigmoid()), points, categories)
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
            categories = categories
        )
        del masks
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold ,self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        
        orig_h, orig_w = self.orig_image.shape[:2]
        keep_mask = ~utils.is_box_near_crop_edge(data["boxes"], crop_box,  [0, 0, orig_w, orig_h], self.downscale)
        if not torch.all(keep_mask):
            data.filter(keep_mask)


        return data
    


    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["masks"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        # import time
        # t = time.time()
        for mask in mask_data["masks"].cpu().numpy():

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
        # print(time.time() -t)

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_data["boxes"][i_mask] = torch.as_tensor(boxes[i_mask], device=mask_data["masks"].device)  # update res directly
                mask_data['masks'][i_mask] = torch.as_tensor(masks[i_mask], device=mask_data["masks"].device)
        mask_data.filter(keep_by_nms)

        return mask_data

    def match_ref(self, sim_map, pos_sim_thresh):
        #TODO: It remains a question whether it is better to do some sim map fusion here
        fg_mask = sim_map > pos_sim_thresh
        coords = fg_mask.nonzero()[:,[1,0]]
        return coords
    
    