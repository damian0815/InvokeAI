import math

import PIL
import torch
from torchvision.transforms.functional import resize as tv_resize, InterpolationMode

from ldm.models.diffusion.cross_attention_control import get_attention_modules, CrossAttentionType


class AttentionMapSaver():

    def __init__(self, token_ids: range, latents_shape: torch.Size):
        self.token_ids = token_ids
        self.latents_shape = latents_shape
        #self.collated_maps = #torch.zeros([len(token_ids), latents_shape[0], latents_shape[1]])
        self.collated_maps = {}

    def add_attention_maps(self, maps: torch.Tensor, key: str):
        key_and_size = f'{key}_{maps.shape[1]}'

        # extract desired tokens
        maps = maps[:, :, self.token_ids]

        # sum along dim 0
        maps = torch.sum(maps, 0)

        # summed maps now has shape [(H*W), N] for N tokens
        # but we want [N, H, W]
        scale_factor = math.sqrt(maps.shape[0] / (self.latents_shape[0] * self.latents_shape[1]))
        maps_h = int(float(self.latents_shape[0]) * scale_factor)
        maps_w = int(self.latents_shape[1] * scale_factor)
        # and we need to do some dimension juggling
        num_tokens = maps.shape[1]
        maps = torch.reshape(torch.swapdims(maps, 0, 1), [num_tokens, maps_h, maps_w])
        if scale_factor != 1:
            maps = tv_resize(maps, self.latents_shape, InterpolationMode.BICUBIC)

        #maps = torch.nn.functional.normalize(maps, dim=0)

        #maps = maps.to(self.collated_maps.device)
        # screen blend, f(a, b) = 1 - (1 - a)(1 - b)
        #self.collated_maps = torch.cat([self.collated_maps, maps.unsqueeze(0)])
        if key_and_size not in self.collated_maps:
            self.collated_maps[key_and_size] = torch.zeros_like(maps, device='cpu')
        self.collated_maps[key_and_size] += maps.cpu()
        #self.collated_maps = self.collated_maps + maps

    def write_maps_to_disk(self):
        #print('save goes here')
        merged = None
        for key, maps in self.collated_maps.items():

            num_tokens = maps.shape[0]
            height = maps.shape[1]
            width = maps.shape[2]

            stacked = torch.reshape(maps, [num_tokens * height, width])

            stacked_min = torch.min(stacked)
            stacked_range = torch.max(stacked) - stacked_min
            stacked_normalized = (stacked - stacked_min) / stacked_range
            if merged is None:
                merged = stacked_normalized
            else:
                # screen blend
                merged = 1 - (1 - stacked_normalized)*(1 - merged)

        def normalize_and_ubyte(x):
            return x.mul(0xff).byte().cpu()
        merged_ubyte = normalize_and_ubyte(merged)
        pil_image = PIL.Image.fromarray(merged_ubyte.numpy(), mode='L')
        path = f'/tmp/attention_maps_merged.png'
        pil_image.save(path, 'PNG')

def setup_attention_map_saving(unet, saver: AttentionMapSaver):
    def callback(slice, dim, offset, slice_size, key):
        if dim is not None:
            print("sliced tokens attention map saving is not implemented")
            return
        saver.add_attention_maps(slice, key)

    tokens_cross_attention_modules = get_attention_modules(unet, CrossAttentionType.TOKENS)
    for identifier, module in tokens_cross_attention_modules:
        key = ('down' if identifier.startswith('down') else
                'up' if identifier.startswith('up') else
                'mid')
        module.set_attention_slice_calculated_callback(lambda slice, dim, offset, slice_size, key=key: callback(slice, dim, offset, slice_size, key))

def remove_attention_map_saving(unet):
    tokens_cross_attention_modules = get_attention_modules(unet, CrossAttentionType.TOKENS)
    for _, module in tokens_cross_attention_modules:
        module.set_attention_slice_calculated_callback(None)
