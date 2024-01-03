import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower='openai/clip-vit-large-patch14-336', unfreeze_layers=0):
        super().__init__()
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        for param in self.vision_tower.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0:
            for layer in self.vision_tower.vision_model.encoder.layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
        return image_features

    def forward(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
