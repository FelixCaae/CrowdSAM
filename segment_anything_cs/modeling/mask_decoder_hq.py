# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoderHQ(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
        n_class=1,
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
                for i in range(self.num_mask_tokens+1)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens , iou_head_depth
        )

        # HQ-SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim) # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # corresponding new MLP layer for HQ-Ouptput-Token
        self.num_mask_tokens = self.num_mask_tokens + 1
        
        # three conv fusion layers for obtaining HQ-Feature
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        
        self.dino_proj = nn.Linear(1024, transformer_dim)
        self.parallel_iou_head = MLP(transformer_dim*2, iou_head_hidden_dim,  1, iou_head_depth)
        self.point_classifier = MLP(transformer_dim, iou_head_hidden_dim, n_class, 2)


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
        dino_feats,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        masks, iou_pred, class_scores = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
            dino_feats=dino_feats,
        )

        # Select the correct mask or masks for output
        # mask with highest score
        if multimask_output:
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_pred = iou_pred[:, mask_slice]
            class_scores = class_scores[:, mask_slice]
            masks_sam = masks[:, mask_slice]
            masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]

        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            class_scores = class_scores[:, mask_slice]
            masks_sam = masks[:, mask_slice]
            masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]
            masks_sam = masks_sam + masks_hq
        return masks_sam, iou_pred, class_scores

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        dino_feats,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_sam_hq],dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
   
   
        dino_feats = F.interpolate(dino_feats.permute(0,3,1,2), (256,256), mode='bilinear')
        mask_weight = masks_sam.flatten(2).softmax(-1).reshape(b, 4, 256, 256)
        pooled_dino_feats = torch.einsum('blhw,chw->blc', mask_weight, dino_feats[0])
        proj_dino_feats = self.dino_proj(pooled_dino_feats)
        cls_scores = self.point_classifier(proj_dino_feats)
        proj_dino_feats = F.normalize(pooled_dino_feats, dim=-1)
        fused_token = torch.cat([iou_token_out.unsqueeze(1).repeat(1,4,1), mask_tokens_out[:,:4]], dim=-1)
        res_iou_pred = self.parallel_iou_head(fused_token).squeeze(2)
        iou_pred = (iou_pred+res_iou_pred) 
        return masks, iou_pred, cls_scores


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
