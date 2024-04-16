# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
# from .transformer import TwoWayTransformer
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type
from segment_anything.modeling.mask_decoder3D import TwoWayTransformer3D, MLP


class MaskDecoder3DMLP(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_outputs: int = 5,
        activation: Type[nn.Module] = nn.GELU,
        pred_head_depth: int = 3,
        pred_head_hidden_dim: int = 256,
        output_dims: List[int] = [1, 1, 1, 1, 1]
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          pred_head_depth (int): the depth of the MLP used to predict
            mask quality
          pred_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        # self.transformer = transformer
        self.transformer = TwoWayTransformer3D(
                depth=2,
                embedding_dim=self.transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )
        assert num_outputs == len(output_dims), ("Number of output channels must be the same as "
                                                 "the the number of output dimensions")
        self.num_outputs = num_outputs
        self.output_dims = output_dims
        self.output_tokens = nn.Embedding(self.num_outputs, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    transformer_dim,
                    pred_head_hidden_dim,
                    output_dim,
                    pred_head_depth
                )
                for output_dim in self.output_dims
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = self.output_tokens.weight.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe

        # Run the transformer
        # import IPython; IPython.embed()
        hs, _ = self.transformer(src, pos_src, tokens)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_outputs):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](hs[:, i, :]))
        # Cannot be stacked if output_dims are different
        # hyper_in = torch.stack(hyper_in_list, dim=1)

        return hyper_in_list

