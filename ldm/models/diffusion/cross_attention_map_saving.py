import torch
from torchvision.transforms.functional import resize as tv_resize, InterpolationMode

class AttentionMapSaver():

    def __init__(self, token_ids: range, latents_size: torch.Size):
        self.token_ids = token_ids
        self.latents_size = latents_size
        self.collated_maps = {i:torch.zeros(latents_size) for i in self.token_ids}

    def add_attention_maps(self, maps: torch.Tensor):
        if maps.size[-2:] != self.latents_size:
            maps = tv_resize(maps, self.latents_size, InterpolationMode.BILINEAR)

        for token_id in self.token_ids:
            this_map = maps[token_id]
            existing_map = self.collated_maps[token_id]
            # screen blend, f(a, b) = 1 - (1 - a)(1 - b)
            screen_blended_maps = 1 - (1 - maps[token_id])*(1 - this_map)
            self.collated_maps[token_id] = screen_blended_maps
