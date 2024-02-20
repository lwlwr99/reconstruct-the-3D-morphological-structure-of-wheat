from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn, einsum
from einops import repeat
from ..ops import queryandgroup

class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, stop_trans, nsample = 10, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        self.stop_trans = stop_trans

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, stop_trans, nsample, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)
        
        if len(nPlanes) > self.stop_trans:
            self.transformerLayer = PointAttentionLayer(in_planes=nPlanes[0], nsample=np.clip(nsample, 12, None))

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        if len(self.nPlanes) > self.stop_trans:
            out_feats = self.transformerLayer(output)
            output = output.replace_feature(out_feats)
        return output
    

class UBlock_Tiny(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock_Tiny(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        in_planes,
        pos_mlp_hidden_dim=16,
        attn_mlp_hidden_mult=2,
        nsample=16
    ):
        super().__init__()

        self.in_planes = in_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, in_planes)
        self.linear_k = nn.Linear(in_planes, in_planes)
        self.linear_v = nn.Linear(in_planes, in_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, in_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes, in_planes * attn_mlp_hidden_mult, bias=False),
            nn.BatchNorm1d(in_planes * attn_mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes * attn_mlp_hidden_mult, in_planes)
        )
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        x_k = queryandgroup(self.nsample, p, p, torch.cat((x_k, x_v), -1), None, o, o, use_xyz=True)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        x_k, x_v = x_k.chunk(2, dim=-1) 

        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)

        w = x_k - x_q.unsqueeze(1) + p_r 

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        w = self.softmax(w) 

        x = einsum('n s c, n s c -> n c', (x_v + p_r), w)

        return x


class PointAttentionLayer(nn.Module):

    def __init__(self, in_planes, nsample):
        super().__init__()
        self.self_attn = PointTransformerLayer(in_planes=in_planes, nsample=nsample)

        self.norm = nn.LayerNorm(in_planes)

        cat_planes = in_planes * 2

        self.catLayer = nn.Sequential(
            nn.Linear(cat_planes, cat_planes, bias=False),
            nn.BatchNorm1d(cat_planes),
            nn.ReLU(),
            nn.Linear(cat_planes, cat_planes, bias=False),
            nn.BatchNorm1d(cat_planes),
            nn.ReLU(),
            nn.Linear(cat_planes, in_planes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        batch_ids = input.indices[:, 0]
        xyz = input.indices[:, 1:].float()
        feats = input.features

        batch_size = batch_ids.max() + 1
        offset, count = [], 0
        for i in range(batch_size):
            # num = list(batch_ids).count(i)
            num = torch.where(batch_ids == i)[0].shape[0]
            if num != 0:
                count += num
                offset.append(count)
        offset = torch.tensor(offset, dtype=torch.int32, device=feats.device)

        output = self.self_attn((xyz, feats, offset))
        output = self.norm(self.catLayer(torch.cat((feats, output), dim=-1)))
        return output