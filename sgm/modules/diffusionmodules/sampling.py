"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from ...util import load_xt
import torchvision.transforms.functional as F

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None, inversion=False):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        if inversion:
            sigmas = sigmas.flip(0)
            sigmas[0] += 1e-8
        uc = default(uc, cond)

        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc, is_modulate_step=False, is_injected_step=False,
                modulate_params=None):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), 
                            is_modulate_step=is_modulate_step, is_injected_step=is_injected_step,
                            modulate_params=modulate_params)
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn  # by default s_churn = 0
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0,
                     is_modulate_step=False, is_injected_step=False, modulate_params=None,
                     is_smooth_latent=False, model=None, smooth_step_size=None):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        if sigma_hat.mean() < 1e-6:
            denoised = x
        else:
            denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, is_modulate_step=is_modulate_step,
                                is_injected_step=is_injected_step,
                                modulate_params=modulate_params)
        if is_smooth_latent:
            assert model is not None
            x_denoised = model.decode_first_stage(denoised)
            num_frames = x_denoised.shape[0]
            
            for frame_id in range(1, num_frames - 1):
                if (frame_id - smooth_step_size) % 3 == 0:
                    x_denoised[frame_id] = 0.5 * (x_denoised[frame_id - 1] + x_denoised[frame_id + 1])
            denoised = model.encode_first_stage(x_denoised)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x
    
    def add_noise(self, x, cond, uc=None, num_steps=None, noise_level=0):
        _, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        
        sigma = sigmas[noise_level]
        eps = torch.randn_like(x) * sigma
        x = x + eps
        
        x /= torch.sqrt(1.0 + sigmas[0] ** 2.0)  

        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, callback=None, img_callback=None,
                 is_modulate=False, modulate_params=None, uc_list=None, t_start=None, t_end=None, is_latent_blending=False,
                 feature_height=None, feature_width=None, is_smooth_latent=False, model=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        if is_modulate:
            if len(modulate_params["modulate_timestep_frames"]) == 0:
                modulate_timestep = modulate_params["modulate_timestep"]
            else:
                modulate_timestep = modulate_params["modulate_timestep_frames"].keys()
            is_injected_features = modulate_params["is_injected_features"]
        else:
            is_injected_features = False

        all_sigmas = self.get_sigma_gen(num_sigmas)
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = num_sigmas
            
        all_sigmas = list(all_sigmas)[t_start:(t_end + 1)]
        for i in all_sigmas:
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            
            if is_modulate and i in modulate_timestep:  # only modulate at specific timestep
                is_modulate_step = True

            else:
                is_modulate_step = False
                
            if is_modulate and is_injected_features and i >= min(modulate_timestep):
                is_injected_step = True
            else:
                is_injected_step = False
                
            if modulate_params is not None:
                modulate_params["timestep"] = i
                
            if is_modulate and i in modulate_timestep:
                if len(modulate_params["modulate_timestep_frames"]) > 0:
                    modulate_params["modulate_timestep_frames_group"] = modulate_params["modulate_timestep_frames"][i]
                else:
                    modulate_params["modulate_timestep_frames_group"] = list(range(modulate_params["num_frames"]))
                    
            if uc_list is not None:
                uc = uc_list[i]
                
            if is_smooth_latent:
                if i == 23:
                    is_smooth_latent_step = True
                    smooth_step_size = 1
                elif i == 24:
                    is_smooth_latent_step = True
                    smooth_step_size = 2
                else:
                    is_smooth_latent_step = False
                    smooth_step_size = None
            else:
                is_smooth_latent_step = False
                smooth_step_size = None
                
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                is_modulate_step=is_modulate_step,
                is_injected_step=is_injected_step,
                modulate_params=modulate_params,
                is_smooth_latent=is_smooth_latent_step,
                model=model,
                smooth_step_size=smooth_step_size,
            )
            
            if is_latent_blending:
                if i >= modulate_params["latent_mask_start"] and i <= modulate_params["latent_mask_end"]:
                    xh, xw = x.shape[-2], x.shape[-1]
                    ori_xt = load_xt(
                                modulate_params["feature_folder"], 
                                modulate_params["exp_name"], 
                                modulate_params["timestep"], x.device)
                    ori_xt = ori_xt.to(x.dtype)
                    feature_masks = modulate_params["feature_masks"]
                    feature_masks = torch.stack(feature_masks, dim=0)
                    if feature_height is None:
                        feature_height = 28
                    if feature_width is None:
                        feature_width = 52
                    feature_masks = feature_masks.reshape(feature_masks.shape[0], feature_height, feature_width)  # [f, hw]
                    feature_masks = feature_masks.unsqueeze(1)  # [f, 1, h, w]
                    feature_masks = torch.nn.functional.interpolate(feature_masks, size=(xh, xw), mode='nearest')
                    
                    pil_mask = F.to_pil_image(feature_masks[0])
                    
                    x = x * feature_masks + ori_xt * (1 - feature_masks)
                    x = x.float()

            if callback:
                callback(i)
            if img_callback:
                if is_modulate:
                    if i >= min(modulate_timestep):
                        img_callback(x, i)
                else:
                    img_callback(x, i)
                    

        return x
    
    def inversion(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps, inversion=True,
        )
        
        latents_list = [x]
        
        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                is_modulate_step=False,
                is_injected_step=False,
                modulate_params=None,
            )
            
            latents_list.append(x)
            
        
        x /= torch.sqrt(1.0 + sigmas[-1] ** 2.0)    
        

        return x, latents_list
    
    @torch.enable_grad()
    def null_text_optimization(self, denoiser, latents_inv_list, cond, uc, 
                               num_inner_steps=10, epsilon=1e-5, num_steps=25):
        uc_list = []
        latent_cur = latents_inv_list[-1]  # [XT]
        
        latent_cur, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            latent_cur, cond, uc, num_steps, 
        )
        
        import torch.nn as nn
        tmp_network = nn.Linear(1, 1)
        for p in tmp_network.parameters():
            p.requires_grad = False
        tmp_input = torch.randn(1, 1).requires_grad_(True)
        optimizer = Adam([tmp_input], lr=1e-2)
        target = torch.randn(1, 1)
        for i in range(3):
            tmp_output = tmp_network(tmp_input)
            loss = nnf.mse_loss(tmp_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        bar = tqdm(total=num_inner_steps * num_steps)
        for i in range(num_steps):
            uc_tmp = uc.copy()
            uc_tmp["crossattn"].requires_grad = True
            optimizer = Adam([uc_tmp["crossattn"]], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents_inv_list[len(latents_inv_list) - i - 2]
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            
            for j in range(num_inner_steps):
                # The code snippet is calling the `sampler_step` method with the provided arguments.
                # The method is being called with the following arguments:
                
                latent_prev_rec = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    latent_cur,
                    cond,
                    uc_tmp,
                    gamma,
                    is_modulate_step=False,
                    is_injected_step=False,
                    modulate_params=None,
                )  
                loss = nnf.mse_loss(latent_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()

                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
                
            uc_tmp["crossattn"] = uc_tmp["crossattn"].detach()
            uc_list.append(uc_tmp)
            with torch.no_grad():
                latent_cur = self.sampler_step(
                        s_in * sigmas[i],
                        s_in * sigmas[i + 1],
                        denoiser,
                        latent_cur,
                        cond,
                        uc_tmp,
                        gamma,
                        is_modulate_step=False,
                        is_injected_step=False,
                        modulate_params=None,
                    )  
        
        bar.close()
        
        return uc_list    
    
    
    def edit(self, denoiser, x, cond, uc=None, num_steps=None, cond_edit=None, uc_edit=None, 
             edit_start_step=5, edit_end_step=24):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        all_sigmas = self.get_sigma_gen(num_sigmas)

        for i in all_sigmas:
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            
            if i >= edit_start_step and i <= edit_end_step:
                cond_to_use = cond_edit
                uc_to_use = uc_edit
            else:
                cond_to_use = cond
                uc_to_use = uc
                
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond_to_use,
                uc_to_use,
                gamma,
                is_modulate_step=False,
                is_injected_step=False,
                modulate_params=False,
            )
 
        return x

class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x
