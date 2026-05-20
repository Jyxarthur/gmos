# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from .position_encoding import apply_rotary_enc, compute_axial_cis
from .basic import MLP


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        rope=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.rope = rope

        for i in range(depth):
            self.layers.append(
                SelfAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=False,
                    rope=rope,
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # B hw C
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # 1 hw C

        # Prepare queries
        queries = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries = layer(
                queries=queries,
                query_pe=image_pe,
            )

        queries = queries.permute(0, 2, 1).view(bs, c, h, w)  # B C h w
        return queries


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rope=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, rope=rope)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.rope = rope

        self.num_heads = num_heads

    def forward(
        self, queries: Tensor, query_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            if self.rope is None:
                q = queries + query_pe
                attn_out = self.self_attn(q=q, k=q, v=queries)
            else:
                attn_out = self.self_attn(q=queries, k=queries, v=queries, q_pos=query_pe, k_pos=query_pe)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        return queries


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        rope=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.rope = rope

        for i in range(depth):
            self.layers.append(
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    rope=rope,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        if self.rope is None:
            k = keys + image_pe
            attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        else:
            attn_out = self.final_attn_token_to_image(q=q, k=keys, v=keys, q_pos=None, k_pos=image_pe)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        keys = keys.permute(0, 2, 1).view(bs, c, h, w)  # B C h w
        return queries, keys


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rope=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.rope = rope

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope is None:
            k = keys + key_pe
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        else:
            attn_out = self.cross_attn_token_to_image(q=q, k=keys, v=keys, q_pos=None, k_pos=key_pe)
        
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


class TwoSourceTransformerDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        rope1=None,
        rope2=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.rope1 = rope1
        self.rope2 = rope2

        for i in range(depth):
            self.layers.append(
                TwoSourceCrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    rope1=rope1,
                    rope2=rope2,
                )
            )

        self.final_attn_token_to_image1 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope1,
        )
        self.final_attn_token_to_image2 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope2,
        )
        self.norm_final_attn1 = nn.LayerNorm(embedding_dim)
        self.norm_final_attn2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image1_embedding: Tensor,
        image1_pe: Tensor,
        image2_embedding: Tensor,
        image2_pe: Tensor,
        point_embedding: Tensor,
    ):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image1_embedding.shape
        image1_embedding = image1_embedding.flatten(2).permute(0, 2, 1)
        image1_pe = image1_pe.flatten(2).permute(0, 2, 1)

        bs, c_, h_, w_ = image2_embedding.shape
        image2_embedding = image2_embedding.flatten(2).permute(0, 2, 1)
        image2_pe = image2_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys1 = image1_embedding
        keys2 = image2_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries_interm, queries = layer(
                queries=queries,
                keys1=keys1,
                keys2=keys2,
                query_pe=point_embedding,
                key1_pe=image1_pe,
                key2_pe=image2_pe,
            )
        

        q = queries_interm + point_embedding
        if self.rope1 is None:
            k = keys1 + image1_pe
            attn_out = self.final_attn_token_to_image1(q=q, k=k, v=keys1)
        else:
            attn_out = self.final_attn_token_to_image1(q=q, k=keys1, v=keys1, q_pos=None, k_pos=image1_pe)
        queries_interm = queries_interm + attn_out
        queries_interm = self.norm_final_attn1(queries_interm)

        keys1 = keys1.permute(0, 2, 1).view(bs, c, h, w)  # B C h w


        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        if self.rope2 is None:
            k = keys2 + image2_pe
            attn_out = self.final_attn_token_to_image2(q=q, k=k, v=keys2)
        else:
            attn_out = self.final_attn_token_to_image2(q=q, k=keys2, v=keys2, q_pos=None, k_pos=image2_pe)
        queries = queries + attn_out
        queries = self.norm_final_attn2(queries)

        keys2 = keys2.permute(0, 2, 1).view(bs, c_, h_, w_)  # B C h w

        
        
        return queries_interm, queries, keys1, keys2


class TwoSourceCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rope1=None,
        rope2=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image1 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope1,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp1 = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image2 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope2,
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.mlp2 = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm5 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.rope1 = rope1
        self.rope2 = rope2

    def forward(
        self, queries: Tensor, keys1: Tensor, keys2: Tensor, query_pe: Tensor, key1_pe: Tensor, key2_pe: Tensor,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope1 is None:
            k = keys1 + key1_pe
            attn_out = self.cross_attn_token_to_image1(q=q, k=k, v=keys1)
        else:
            attn_out = self.cross_attn_token_to_image1(q=q, k=keys1, v=keys1, q_pos=None, k_pos=key1_pe)
        
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp1(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        queries_interm = queries.clone()

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope2 is None:
            k = keys2 + key2_pe
            attn_out = self.cross_attn_token_to_image2(q=q, k=k, v=keys2)
        else:
            attn_out = self.cross_attn_token_to_image2(q=q, k=keys2, v=keys2, q_pos=None, k_pos=key2_pe)
        
        queries = queries + attn_out
        queries = self.norm4(queries)

        # MLP block
        mlp_out = self.mlp2(queries)
        queries = queries + mlp_out
        queries = self.norm5(queries)

        return queries_interm, queries



class TwoSourcePartialTwoWayTransformerDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        rope1=None,
        rope2=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.rope1 = rope1
        self.rope2 = rope2

        for i in range(depth):
            self.layers.append(
                TwoSourcePartialTwoWayCrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    rope1=rope1,
                    rope2=rope2,
                )
            )

        self.final_attn_token_to_image1 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope1,
        )
        self.final_attn_token_to_image2 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope2,
        )
        self.norm_final_attn1 = nn.LayerNorm(embedding_dim)
        self.norm_final_attn2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image1_embedding: Tensor,
        image1_pe: Tensor,
        image2_embedding: Tensor,
        image2_pe: Tensor,
        point_embedding: Tensor,
    ):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image1_embedding.shape
        image1_embedding = image1_embedding.flatten(2).permute(0, 2, 1)
        image1_pe = image1_pe.flatten(2).permute(0, 2, 1)

        bs, c_, h_, w_ = image2_embedding.shape
        image2_embedding = image2_embedding.flatten(2).permute(0, 2, 1)
        image2_pe = image2_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys1 = image1_embedding
        keys2 = image2_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries_interm, queries, keys2 = layer(
                queries=queries,
                keys1=keys1,
                keys2=keys2,
                query_pe=point_embedding,
                key1_pe=image1_pe,
                key2_pe=image2_pe,
            )
        

        q = queries_interm + point_embedding
        if self.rope1 is None:
            k = keys1 + image1_pe
            attn_out = self.final_attn_token_to_image1(q=q, k=k, v=keys1)
        else:
            attn_out = self.final_attn_token_to_image1(q=q, k=keys1, v=keys1, q_pos=None, k_pos=image1_pe)
        queries_interm = queries_interm + attn_out
        queries_interm = self.norm_final_attn1(queries_interm)

        keys1 = keys1.permute(0, 2, 1).view(bs, c, h, w)  # B C h w


        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        if self.rope2 is None:
            k = keys2 + image2_pe
            attn_out = self.final_attn_token_to_image2(q=q, k=k, v=keys2)
        else:
            attn_out = self.final_attn_token_to_image2(q=q, k=keys2, v=keys2, q_pos=None, k_pos=image2_pe)
        queries = queries + attn_out
        queries = self.norm_final_attn2(queries)

        keys2 = keys2.permute(0, 2, 1).view(bs, c_, h_, w_)  # B C h w

        return queries_interm, queries, keys1, keys2


class TwoSourcePartialTwoWayCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rope1=None,
        rope2=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image1 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope1,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp1 = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image2 = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope2,
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.mlp2 = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm5 = nn.LayerNorm(embedding_dim)


        self.cross_attn_image2_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope2,
        )

        self.norm6 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.rope1 = rope1
        self.rope2 = rope2

    def forward(
        self, queries: Tensor, keys1: Tensor, keys2: Tensor, query_pe: Tensor, key1_pe: Tensor, key2_pe: Tensor,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope1 is None:
            k = keys1 + key1_pe
            attn_out = self.cross_attn_token_to_image1(q=q, k=k, v=keys1)
        else:
            attn_out = self.cross_attn_token_to_image1(q=q, k=keys1, v=keys1, q_pos=None, k_pos=key1_pe)
        
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp1(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        queries_interm = queries.clone()

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope2 is None:
            k = keys2 + key2_pe
            attn_out = self.cross_attn_token_to_image2(q=q, k=k, v=keys2)
        else:
            attn_out = self.cross_attn_token_to_image2(q=q, k=keys2, v=keys2, q_pos=None, k_pos=key2_pe)
        
        queries = queries + attn_out
        queries = self.norm4(queries)

        # MLP block
        mlp_out = self.mlp2(queries)
        queries = queries + mlp_out
        queries = self.norm5(queries)

        q = queries + query_pe
        if self.rope2 is None:
            k = keys2 + key2_pe
            attn_out = self.cross_attn_image2_to_token(q=k, k=q, v=queries)
        else:
            attn_out = self.cross_attn_image2_to_token(q=keys2, k=q, v=queries, q_pos=key2_pe, k_pos=None)
        keys2 = keys2 + attn_out
        keys2 = self.norm6(keys2)


        return queries_interm, queries, keys2



class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        rope=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.rope = rope

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    rope=rope,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        if self.rope is None:
            k = keys + image_pe
            attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        else:
            attn_out = self.final_attn_token_to_image(q=q, k=keys, v=keys, q_pos=None, k_pos=image_pe)

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        keys = keys.permute(0, 2, 1).view(bs, c, h, w)  # B C h w
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rope=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rope=rope,
        )

        self.norm4 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.rope = rope

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)


        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if self.rope is None:
            k = keys + key_pe
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        else:
            attn_out = self.cross_attn_token_to_image(q=q, k=keys, v=keys, q_pos=None, k_pos=key_pe)

        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        if self.rope is None:
            k = keys + key_pe
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        else:
            attn_out = self.cross_attn_image_to_token(q=keys, k=q, v=queries, q_pos=key_pe, k_pos=None)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
        rope = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout
        self.rope = rope

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor = None, k_pos: Tensor = None) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        if self.rope is not None and ((q_pos is not None) or (k_pos is not None)):
            if q_pos is not None:
                q = self.rope(q, q_pos)
            if k_pos is not None:
                k = self.rope(k, k_pos)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = (
            freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out