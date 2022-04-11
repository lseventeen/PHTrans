from audioop import bias
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from nnunet.network_architecture.neural_network import SegmentationNetwork
from einops import rearrange
from nnunet.network_architecture.initialization import InitWeights_He
from .swin_3D import *


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(
            input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
                'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class DownOrUpSample(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_op, conv_kwargs,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, nonlinbasic_block=ConvDropoutNormNonlin):
        super(DownOrUpSample, self).__init__()
        self.blocks = nonlinbasic_block(input_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        nonlin, nonlin_kwargs)

    def forward(self, x):
        return self.blocks(x)


class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.proj = nn.Conv3d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, num_stage, num_only_conv_stage, num_pool, base_num_features, input_resolution, depth, num_heads,
                 window_size, image_channels=1, num_conv_per_stage=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, basic_block=ConvDropoutNormNonlin, max_num_features_factor=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, down_or_upsample=None, feat_map_mul_on_downscale=2,
                 use_checkpoint=False, deep_supervision=None, num_classes=None, is_encoder=True):

        super().__init__()
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder
        self.max_num_features_factor = max_num_features_factor if max_num_features_factor is not None else 1000
        self.dim = min((base_num_features * feat_map_mul_on_downscale **
                       num_stage), base_num_features*self.max_num_features_factor)
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = conv_pad_sizes
        if num_stage == 0 and is_encoder:

            self.first_conv = nn.Conv3d(image_channels, self.dim, 1)
        if not is_encoder and num_stage < num_pool:
            self.input_features = 2*self.dim
        else:
            self.input_features = self.dim
        self.output_features = self.dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_stage]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_stage]

        self.input_du_channels = self.dim
        self.output_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage+1 if is_encoder else num_stage-1)),
                                      base_num_features*self.max_num_features_factor)
        self.conv_blocks = nn.Sequential(
            *([basic_block(self.input_features, self.output_features, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(self.output_features, self.output_features, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_conv_per_stage - 1)]))

        # build blocks
        if num_stage >= num_only_conv_stage:
            self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(dim=self.dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=[0, 0, 0] if (i % 2 == 0) else [
                                         window_size[0] // 2, window_size[1] // 2, window_size[2] // 2],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

        # patch merging layer
        if down_or_upsample is not None:
            dowm_stage = num_stage-1 if not is_encoder else num_stage
            self.down_or_upsample = nn.Sequential(down_or_upsample(self.input_du_channels, self.output_du_channels, pool_op_kernel_sizes[dowm_stage],
                                                                   pool_op_kernel_sizes[dowm_stage], bias=False),
                                                  nn.InstanceNorm3d(
                                                      self.output_du_channels, **norm_op_kwargs)
                                                  )
        else:
            self.down_or_upsample = None
        if deep_supervision is not None and is_encoder == False:
            self.deep_supervision = deep_supervision(self.dim, num_classes)
        else:
            self.deep_supervision = None

    def forward(self, x, skip):
        if self.num_stage == 0 and self.is_encoder:
            x = self.first_conv(x)
        s = x
        if not self.is_encoder and self.num_stage < self.num_pool:
            x = torch.cat((x, skip), dim=1)
        x = self.conv_blocks(x)
        if self.num_stage >= self.num_only_conv_stage:
            if not self.is_encoder and self.num_stage < self.num_pool:
                s = s + skip
            for tblk in self.swin_blocks:
                if self.use_checkpoint:
                    s = checkpoint.checkpoint(tblk, s)
                else:
                    s = tblk(s)
            x = x + s
        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)

        if self.deep_supervision is not None:
            ds = self.deep_supervision(x)

        if self.is_encoder:
            return x, du
        elif self.deep_supervision is None and self.down_or_upsample is not None:
            return du, None
        elif self.deep_supervision is not None and self.down_or_upsample is not None:
            return du, ds
        elif self.deep_supervision is not None and self.down_or_upsample is None:
            return None, ds

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PHTrans(SegmentationNetwork):
    def __init__(self, img_size, base_num_features, num_classes, num_pool, image_channels=1, num_only_conv_stage=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, max_num_features_factor=None, depths=[2, 2, 2, 2], num_heads=[4, 8, 16, 32],
                 window_size=[3, 6, 6], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op

        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.num_layers = len(depths) 
        self.base_num_features = base_num_features
        self.num_features = int(base_num_features * 2 **
                                (self.num_layers - 1)) 
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        # stochastic depth decay rule  [0, 1/12,2/12,3/12,...11/12]
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        # build layers
        self.down_layers = nn.ModuleList()
        for i_layer in range(num_pool):  # 0,1,2,3
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                   img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),  # 56,28,14,7
                               depth=depths[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=self.conv_pad_sizes, pool_op_kernel_sizes=self.pool_op_kernel_sizes,
                               max_num_features_factor=max_num_features_factor,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer-num_only_conv_stage]):sum(depths[:i_layer-num_only_conv_stage + 1])] if (
                                   i_layer >= num_only_conv_stage) else None,  
                               norm_layer=norm_layer,
                               down_or_upsample=nn.Conv3d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               deep_supervision=None,
                               is_encoder=True)
            self.down_layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.up_layers = nn.ModuleList()
        for i_layer in range(num_pool+1)[::-1]:
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                   img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)), 
                               depth=depths[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=self.conv_pad_sizes, pool_op_kernel_sizes=self.pool_op_kernel_sizes,
                               max_num_features_factor=max_num_features_factor,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer-num_only_conv_stage]):sum(depths[:i_layer-num_only_conv_stage + 1])] if (
                                   i_layer >= num_only_conv_stage) else None,  
                               norm_layer=norm_layer,
                               down_or_upsample=nn.ConvTranspose3d if (
                                   i_layer > 0) else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               deep_supervision=DeepSupervision if deep_supervision and i_layer < num_pool else None,
                               num_classes=num_classes,
                               is_encoder=False)
            self.up_layers.append(layer)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        x_skip = []
        for layer in self.down_layers:
            s, x = layer(x, None)
            x_skip.append(s)
        out = []
        for inx, layer in enumerate(self.up_layers):
            x, ds = layer(x, x_skip[5-inx]) if inx > 0 else layer(x, None)
            if inx > 0:
                out.append(ds)
        if self._deep_supervision and self.do_ds:
            return out[::-1]
        else:

            return out[-1]
