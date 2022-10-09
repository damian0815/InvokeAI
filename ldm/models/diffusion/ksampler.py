"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        unconditioned_x, conditioned_x = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)

        # damian0815 thinking out loud notes:
        # b + (a - b)*scale
        # starting at the output that emerges applying the negative prompt (by default ''),
        # (-> this is why the unconditioning feels like hammer)
        # move toward the positive prompt by an amount controlled by cond_scale.
        return unconditioned_x + (conditioned_x - unconditioned_x) * cond_scale



class ProgrammableCFGDenoiser(CFGDenoiser):
    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        deltas = None
        uncond_latents = None
        weights = []
        weighted_cond_list = cond if type(cond) is list else [(cond,1)]
        for this_cond,this_weight in weighted_cond_list:
            #this_cond,this_weight = weighted_cond
            cond_in = torch.cat([uncond, this_cond])
            # always overwrite uncond_latents? is this right?
            uncond_latents, cond_latents = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            delta = cond_latents - uncond_latents
            deltas = delta if deltas is None else torch.cat((deltas, delta))
            weights.append(this_weight)

        # merge the weighted deltas together into a single merged delta
        per_delta_weights = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
        normalize = False
        if normalize:
            per_delta_weights /= torch.sum(per_delta_weights)
        reshaped_weights = per_delta_weights.reshape(per_delta_weights.shape + (1, 1, 1))
        deltas_merged = torch.sum(deltas * reshaped_weights, dim=0, keepdim=True)

        #old_return_value = super().forward(x, sigma, uncond, cond, cond_scale)
        #assert(0 == len(torch.nonzero(old_return_value - (uncond_latents + deltas_merged * cond_scale))))

        return uncond_latents + deltas_merged * cond_scale

class KSampler(object):
    def __init__(self, model, schedule='lms', device=None, **kwargs):
        super().__init__()
        self.model = K.external.CompVisDenoiser(model)
        self.schedule = schedule
        self.device   = device or choose_torch_device()

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(
                x_in, sigma_in, cond=cond_in
            ).chunk(2)
            return uncond + (cond - uncond) * cond_scale

    # most of these arguments are ignored and are only present for compatibility with
    # other samples
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        def route_callback(k_callback_values):
            if img_callback is not None:
                img_callback(k_callback_values['x'], k_callback_values['i'])

        sigmas = self.model.get_sigmas(S)
        if x_T is not None:
            x = x_T * sigmas[0]
        else:
            x = (
                torch.randn([batch_size, *shape], device=self.device)
                * sigmas[0]
            )   # for GPU draw
        model_wrap_cfg = ProgrammableCFGDenoiser(self.model)
        extra_args = {
            # damian: we could insert extra things in here
            'cond': conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        return (
            K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas, extra_args=extra_args,
                callback=route_callback
            ),
            None,
        )
