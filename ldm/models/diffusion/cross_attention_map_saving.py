import math

import torch
from torchvision.transforms.functional import resize as tv_resize, InterpolationMode

from ldm.models.diffusion.cross_attention_control import get_attention_modules, CrossAttentionType


class AttentionMapSaver():

    def __init__(self, token_ids: range, latents_shape: torch.Size):
        self.token_ids = token_ids
        self.latents_shape = latents_shape
        self.collated_maps = torch.zeros([len(token_ids), latents_shape[0], latents_shape[1]])

    def add_attention_maps(self, maps: torch.Tensor):

        # extract desired tokens
        maps = maps[:, :, self.token_ids]

        # sum along dim 0
        maps = torch.sum(maps, 0)

        # summed maps now has shape [(H*W), 77] for 77 tokens
        # but we want [77, H, W]
        scale_factor = math.sqrt(maps.shape[0] / (self.latents_shape[0] * self.latents_shape[1]))
        maps_h = int(float(self.latents_shape[0]) * scale_factor)
        maps_w = int(self.latents_shape[1] * scale_factor)
        # and we need to do some dimension juggling
        maps = torch.reshape(torch.swapdims(maps, 0, 1), [maps.shape[1], maps_h, maps_w])
        if scale_factor != 1:
            maps = tv_resize(maps, self.latents_shape, InterpolationMode.BILINEAR)

        maps = maps.to(self.collated_maps.device)
        # screen blend, f(a, b) = 1 - (1 - a)(1 - b)
        self.collated_maps = 1 - (1 - self.collated_maps)*(1 - maps)

    def write_maps_to_disk(self):
        print('save goes here')
        pass

def setup_attention_map_saving(unet, saver: AttentionMapSaver):
    def callback(slice, dim, offset, slice_size):
        if dim is not None:
            print("sliced tokens attention map saving is not implemented")
            return
        saver.add_attention_maps(slice)

    tokens_cross_attention_modules = get_attention_modules(unet, CrossAttentionType.TOKENS)
    for _, module in tokens_cross_attention_modules:
        module.set_attention_slice_calculated_callback(callback)

def remove_attention_map_saving(unet):
    tokens_cross_attention_modules = get_attention_modules(unet, CrossAttentionType.TOKENS)
    for _, module in tokens_cross_attention_modules:
        module.set_attention_slice_calculated_callback(None)
