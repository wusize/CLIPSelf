import open_clip
from functools import partial
import os
import torch
from torch import nn
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.utils.logging import print_log
from mmcv.cnn import build_norm_layer


def window_partition(x, window_size=16):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    # 1024 -> 64 -> 16x4
    # x: bs, 1 + h*w, c
    x_cls = x[:, :1]   # bs, 1, c
    x = x[:, 1:]
    B, H_W, C = x.shape
    H = W = int(H_W ** 0.5)
    x = x.view(B, H, W, C)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    assert pad_h == 0 and pad_w == 0, f"For now we do not allow additional padding"
    # if pad_h > 0 or pad_w > 0:
    #     x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    x_cls = x_cls.repeat(1, Hp*Wp // int(window_size**2), 1).view(-1, 1, C)
    windows = torch.cat([x_cls, windows], dim=1)

    return windows, (Hp, Wp)

def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, 1+window_size*window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, 1+H*W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw

    windows_cls = windows[:, :1]    # cls tokens
    windows = windows[:, 1:]

    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp*Wp, -1)

    windows_cls = windows_cls.view(B, Hp*Wp // int(window_size**2), 1, -1)

    x_cls = windows_cls.mean(1)   # average all cls tokens
    x = torch.cat([x_cls, x], dim=1)

    assert Hp == H and Wp == W, "For now we do not allow additional padding"
    # if Hp > H or Wp > W:
    #     x = x[:, :H, :W, :].contiguous()
    return x



@BACKBONES.register_module()
class EvaCLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None,
                 window_size=16, window_block_indexes=[]):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
        self.window_size = window_size
        self.window_block_indexes = window_block_indexes
        clip_model = open_clip.create_model(model_name,
                                            pretrained="eva",
                                            cache_dir=pretrained)
        self.embed_dim = embed_dim = clip_model.embed_dim  # output dim
        self.width = width = clip_model.visual.embed_dim
        self.patch_size = patch_size = clip_model.visual.patch_embed.patch_size[0]
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual
        # self.interpolate3 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        # self.interpolate4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained="eva",
                                            cache_dir=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for EVA: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    def forward(self, x):
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]

        with torch.no_grad():
            x = visual.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = visual.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if visual.pos_embed is not None:
                x = x + visual.rescale_positional_embedding(out_size=(h, w))
            x = visual.pos_drop(x)

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            if os.getenv('RoPE') == '1':
                if visual.training and not isinstance(visual.patch_dropout, nn.Identity):
                    x, patch_indices_keep = visual.patch_dropout(x)
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=patch_indices_keep)
                else:
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=None)
                    x = visual.patch_dropout(x)
            else:
                x = visual.patch_dropout(x)

            rel_pos_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None

            outs = []
            for i, blk in enumerate(visual.blocks[:-1]):
                if i in self.window_block_indexes:
                    x_windows, pad_hw = window_partition(x, window_size=self.window_size)
                    x_windows = blk(x_windows, rel_pos_bias=rel_pos_bias)
                    x = window_unpartition(x_windows, window_size=self.window_size,
                                           hw=(h, w), pad_hw=pad_hw)
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                # x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))
            if not self.training:
                x = x[:, 1:]
                x = visual.norm(x)
                x = visual.head(x)
                assert visual.fc_norm is None
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None

        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x
