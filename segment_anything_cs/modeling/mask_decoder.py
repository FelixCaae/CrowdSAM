# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import math

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        n_class = 1,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens + 1)
            ]
        )

        self.iou_prediction_head = MLP( transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth )
        # adapter modules
        self.dino_proj = nn.Linear(1024, transformer_dim)
        self.parallel_iou_head = DropMLP(transformer_dim*2, iou_head_hidden_dim,  1, iou_head_depth)
        self.point_classifier = DropMLP(transformer_dim, iou_head_hidden_dim, n_class, 2)
        coco_setting = False
        if coco_setting:
            self.bg_prototypes = torch.load('/irip/caizhi_2019/label_completer/sam_adapter_weights/background_prototypes.vitl14.pth')
            fg_prototypes = torch.load('/irip/caizhi_2019/label_completer/sam_adapter_weights/fs_coco_trainval_novel_5shot.vitl14.pkl')        
            fg_prototypes_base = torch.load('/irip/caizhi_2019/label_completer/sam_adapter_weights/fs_coco14_base_train.vitl14.pkl')
            classes = fg_prototypes['label_names'] + fg_prototypes_base['label_names']
            prototypes = torch.cat([fg_prototypes['prototypes'], fg_prototypes_base['prototypes']],dim=0)
            mapping= {name:k for  k,name in enumerate(classes) }
            from coco_names import coco_classes
            # self.class_names = coco_classes
            map_id = []
            for name in coco_classes.values():
                map_id.append(mapping[name.lower()])
            map_id = torch.tensor(map_id)
            prototypes = prototypes[map_id]
            self.prototypes = prototypes
  
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        attn_sim=None,
        target_embedding=None,
        dino_feats = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
   
        masks, iou_pred, class_scores = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            attn_sim=attn_sim,
            target_embedding=target_embedding,
            dino_feats= dino_feats,
        )
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(0, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        class_scores = class_scores[:, mask_slice]
        # Prepare output
        return  masks, iou_pred, class_scores
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        attn_sim=None,
        dino_feats=None,
        target_embedding=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # iou_token: 1, 256
        # mask_token: 4, 256
        # sparse_prompt_embeddings: N_prompt, N_point, 256
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # output_tokens: N_prompt, 5, 256
        # tokens: N_rpompt, N_point + 5, 256
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # src: N_prompt, 256, 64, 64
        # src_pos: N_prompt, 256, 64, 64

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, attn_sim, target_embedding)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        # import pdb;pdb.set_trace()
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # masked_feat = image_embeddings[masks].mean(dim)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        #Use decoded masks to pool related region of dino features
        dino_feats = self.dino_proj(dino_feats)
        dino_feats = F.interpolate(dino_feats.permute(0,3,1,2), (256,256), mode='bilinear')
        mask_weight = masks.flatten(2).softmax(-1).reshape(b, 4, 256, 256)
        dino_feats = torch.einsum('blhw,chw->blc', mask_weight, dino_feats[0])
        #Predict semantic results with point classifier
        cls_scores = self.point_classifier(dino_feats)
        #Repeat IoU token and concat with mask token
        fused_token = torch.cat([iou_token_out.unsqueeze(1).repeat(1,4,1), mask_tokens_out], dim=-1)
        #Predict Redisudal IoU scores and add to default IoU scores
        res_iou_pred = self.parallel_iou_head(fused_token).squeeze(2)
        #Multiply cls scores with refined IoU scores.
        iou_pred = (iou_pred+res_iou_pred) 
        return masks, iou_pred, cls_scores#cls_scores#


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class DropMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.p = p

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1:  # Optional: Avoid dropout after the last layer
                x = F.dropout(x, self.p, training=self.training)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x