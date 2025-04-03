import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import profile
from networks.Linear import *

class Stem_1(nn.Module):
    r""" Stem
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2) # B,L,C  L=H*W
        return x


class PatchMerging_1(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """
    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # B,L,C
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
    def forward(self, x):
        # x: B, H*W, C
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    def forward(self, x):
        # x: B, H*W, C
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x.clone()).permute(0,3,1,2)
        return x

class Bridge(nn.Module):
    def __init__(self,reduction=16):
        super(Bridge,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(1440, 1440//reduction, kernel_size=1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(1440//reduction, 1440, kernel_size=1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        # (B,96,1,1)
        avg1 = self.avgpool(x1)
        max1 = self.maxpool(x1)
        # (B,192,1,1)
        avg2 = self.avgpool(x2)
        max2 = self.maxpool(x2)
        # (B,384,1,1)
        avg3 = self.avgpool(x3)
        max3 = self.maxpool(x3)
        # (B,768,1,1)
        avg4 = self.avgpool(x4)
        max4 = self.maxpool(x4)
        # (B,1440,1,1)
        avg_out = torch.cat([avg1, avg2, avg3, avg4],dim=1)
        avg_out = self.fc(avg_out)
        max_out = torch.cat([max1, max2, max3, max4],dim=1)
        max_out = self.fc(max_out)
        out = self.sig(avg_out + max_out)

        out1 = x1 * out[:, :96, :, :]
        out2 = x2 * out[:, 96:288, :, :]
        out3 = x3 * out[:, 288:672, :, :]
        out4 = x4 * out[:, 672:1440, :, :]

        list = []
        list.append(out1)
        list.append(out2)
        list.append(out3)
        list.append(out4)
        return list


class DownLayer(nn.Module):
    """
    Args:
          dim(int)
          input_resolution (tuple[int])
          depth (int)
          num_heads(int)
          mlp_ratio (float)
          qkv_bias (bool, optional)
          drop (float, optional)
          drop_path (float | tuple[float], optional)
          norm_layer (nn.Module, optional)
    """
    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class UpLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, reduction_ratios, input_resolution):
        super().__init__()
        self.block = nn.ModuleList([
            TransformerBlock(dim, num_heads, reduction_ratios, input_resolution)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.H = input_resolution[0]
        self.W = input_resolution[1]

    def forward(self, x):
        for blk in self.block:
            x = blk(x, self.H, self.W)
        x = self.norm(x)
        return x


# --Transformer--
class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio, input_resolution):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape

        q = self.rope(self.q(x).reshape(B, H, W, C))
        q = q.reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1) # (B,N,C)
            x = self.norm(sp_x)  # (B,N,C)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio, input_resolution):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio, input_resolution)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN_skip(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class lamformer(nn.Module):
    def __init__(self, image_size=224, patch_size=4, in_chans=3, num_classes=9,
                 dims=[96, 192, 384, 768], updims=[768, 384, 192, 96],
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 up_depths=[2, 6, 2, 2], up_num_heads=[24, 12, 6, 3],
                 mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.size1 = (image_size // 4, image_size // 4) # (56,56)
        self.size2 = (image_size // 8, image_size // 8) # (28,28)
        self.size3 = (image_size // 16, image_size // 16) # (14,14)
        self.size4 = (image_size // 32, image_size // 32) # (7,7)

        self.embed1 = Stem_1(img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0])
        self.embed2 = PatchMerging_1(input_resolution=self.size1, dim=dims[0])
        self.embed3 = PatchMerging_1(input_resolution=self.size2, dim=dims[1])
        self.embed4 = PatchMerging_1(input_resolution=self.size3, dim=dims[2])

        self.expand1 = PatchExpand(input_resolution=self.size4, dim=updims[0])
        self.expand2 = PatchExpand(input_resolution=self.size3, dim=updims[1])
        self.expand3 = PatchExpand(input_resolution=self.size2, dim=updims[2])
        self.expand4 = FinalPatchExpand_X4(input_resolution=self.size1, dim=updims[3])

        self.fc3 = nn.Linear(updims[1]*2, updims[1])
        self.fc2 = nn.Linear(updims[2]*2, updims[2])
        self.fc1 = nn.Linear(updims[3]*2, updims[3])

        self.proj = nn.Conv2d(updims[3], num_classes, 1, 1, 0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.stage1 = DownLayer(dim=dims[0],
                                input_resolution=self.size1,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint)

        self.stage2 = DownLayer(dim=dims[1],
                                input_resolution=self.size2,
                                depth=depths[1],
                                num_heads=num_heads[1],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint
                                 )

        self.stage3 = DownLayer(dim=dims[2],
                                input_resolution=self.size3,
                                depth=depths[2],
                                num_heads=num_heads[2],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint
                                 )

        self.stage4 = DownLayer(dim=dims[3],
                                input_resolution=self.size4,
                                depth=depths[3],
                                num_heads=num_heads[3],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint
                                 )

        self.bridge = Bridge()
        self.up2 = UpLayer(dim=updims[1],
                           depth=up_depths[1],
                           num_heads=up_num_heads[1],
                           reduction_ratios=1,
                           input_resolution=self.size3
                           )

        self.up3 = UpLayer(dim=updims[2],
                           depth=up_depths[2],
                           num_heads=up_num_heads[2],
                           reduction_ratios=2,
                           input_resolution=self.size2,
                           )

        self.up4 = UpLayer(dim=updims[3],
                           depth=up_depths[3],
                           num_heads=up_num_heads[3],
                           reduction_ratios=4,
                           input_resolution=self.size1
                           )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward(self,x):
        B = x.shape[0]
        list = []
        # ---阶段1---
        H1,W1 = self.size1
        x1 = self.embed1(x)
        x1 = self.stage1(x1) # B,L,C
        list.append(x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        # ---阶段2---
        H2, W2 = self.size2
        x2 = self.embed2(x1)
        x2 = self.stage2(x2)  # B,L,C
        list.append(x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        # ---阶段3---
        H3, W3 = self.size3
        x3 = self.embed3(x2)
        x3 = self.stage3(x3)  # B,L,C
        list.append(x3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        # ---阶段4---
        H4, W4 = self.size4
        x4 = self.embed4(x3)
        x4 = self.stage4(x4)  # B,L,C
        list.append(x4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous())
        # ---跳跃连接---
        skip = self.bridge(list)
        # --上升1--
        _, c4, _, _ = skip[3].shape
        s4 = skip[3].permute(0, 2, 3, 1).view(B, -1, c4)
        p4 = self.expand1(s4)
        # --上升2--
        _, c3, _, _ = skip[2].shape
        s3 = skip[2].permute(0, 2, 3, 1).view(B, -1, c3)
        p3 = torch.cat([p4,s3], dim=-1)
        p3 = self.fc3(p3)
        p3 = p3 + p4  # 残差连接
        p3 = self.up2(p3)
        p3 = self.expand2(p3)
        # --上升3--
        _, c2, _, _ = skip[1].shape
        s2 = skip[1].permute(0, 2, 3, 1).view(B, -1, c2)
        p2 = torch.cat([p3,s2], dim=-1)
        p2 = self.fc2(p2)
        p2 = p2 + p3  # 残差连接
        p2 = self.up3(p2)
        p2 = self.expand3(p2)
        # --上升4--
        _, c1, _, _ = skip[0].shape
        s1 = skip[0].permute(0, 2, 3, 1).view(B, -1, c1)
        p1 = torch.cat([p2,s1], dim=-1)
        p1 = self.fc1(p1)
        p1 = p1 + p2 # 残差连接
        p1 = self.up4(p1)
        p1 = self.expand4(p1)
        # 投影到分类类别
        out = self.proj(p1)
        return out

if __name__ == '__main__':
    x = torch.randn(2, 1, 224, 224).to('cuda:0')
    net = lamformer(num_classes=2).to('cuda:0')
    flops, params = profile(model=net, inputs=(x,))
    # flops, params = clever_format([flops, params], '%.3f')
    print('flops：{}(G)'.format(flops / 1000 ** 3))
    print('params：{}(M)'.format(params / 1000 ** 2))

