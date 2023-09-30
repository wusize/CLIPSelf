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


@BACKBONES.register_module()
class EvaCLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
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
                x = blk(x, rel_pos_bias=rel_pos_bias)
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
