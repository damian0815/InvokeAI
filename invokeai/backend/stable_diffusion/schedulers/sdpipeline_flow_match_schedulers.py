import logging
import os
from typing import Callable, Optional, Union, Dict, Any, Tuple
from typing_extensions import Self

import torch
from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.configuration_utils import register_to_config, FrozenDict


class SDPipelineInferenceFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """ Wrapper to make FlowMatchEulerDiscreteScheduler compatible with diffusers StableDiffusionPipeline """
    @property
    def init_noise_sigma(self):
        return 1

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, latents, noise, timesteps):
        return self.scale_noise(latents, timesteps, noise)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int]=None,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        if self.config.use_dynamic_shifting and 'mu' not in kwargs:
            kwargs['mu'] = self.shift
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        print(f'timesteps ({num_inference_steps}):', self.timesteps)



class SDPipelineInferenceFlowMatchHeunDiscreteScheduler(FlowMatchHeunDiscreteScheduler):
    """ Wrapper to make FlowMatchEulerDiscreteScheduler compatible with diffusers StableDiffusionPipeline """
    @property
    def init_noise_sigma(self):
        return 1

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, latents, noise, timesteps):
        return self.scale_noise(latents, timesteps, noise)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int]=None,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        if self.config.use_dynamic_shifting and 'mu' not in kwargs:
            kwargs['mu'] = self.shift
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        print(f'timesteps ({num_inference_steps}):', self.timesteps)
