import torch

from ..modules.attention import *
from ..modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding
from functools import partial
from typing import Dict
from ..modules.diffusionmodules.util import get_modulate_lambda


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
            
        self.attn2_out = None  # newly added
        
    ''' 
    def enable_hook(self, masks, modulate_lambda):
        # Register a forward hook on the weight tensor if not already registered
        if self.attn2_hook_handle is None:
            self.attn2_hook_handle = self.attn2_out.register_forward_hook(partial(self.hook, masks=masks, modulate_lambda=modulate_lambda))

    def disable_hook(self):
        # Remove the hook if it is registered
        if self.attn2_hook_handle is not None:
            self.attn2_hook_handle.remove()
            self.attn2_hook_handle = None

    def hook(self, tensor, input, output, masks, modulate_lambda):
        # Modify the weight tensor during the forward pass
        # masks: list
        
        for i, mask in enumerate(masks):
            tensor[i * 2: (i + 1) * 2] = tensor[i * 2: (i + 1) * 2] + modulate_lambda * mask[None, :, None]
    '''        

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None,
        is_modulate_step: bool = False, is_injected_step: bool = False,
        modulate_params: Dict = None
    ) -> torch.Tensor:
        if self.checkpoint:
            def _forward_wrapper(x):
                return self._forward(x, context=context, timesteps=timesteps, 
                                    is_modulate_step=is_modulate_step, 
                                    is_injected_step=is_injected_step,
                                    modulate_params=modulate_params)
            return checkpoint(_forward_wrapper, x)
                # return checkpoint(self._forward, x, context, timesteps)
        else:
            return self._forward(x, context, timesteps=timesteps, 
                                 is_modulate_step=is_modulate_step, 
                                 is_injected_step=is_injected_step,
                                 modulate_params=modulate_params)

    def _forward(self, x, context=None, timesteps=None,
                 is_modulate_step: bool = False, is_injected_step: bool = False,
                 modulate_params: Dict = None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)


        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip
        
        injected_k = None
        injected_q = None
        injected_v = None
        
        
        if self.disable_self_attn:
            if is_injected_step:
                for k, v in modulate_params["injected_features_group"].items():
                    if "temporal_self_attn_k" in k:
                        injected_k = v
                    elif "temporal_self_attn_q" in k:
                        injected_q = v
                    elif "temporal_self_attn_v" in k:
                        injected_v = v
                self.attn1_out = self.attn1(self.norm1(x), context=context, 
                                            injected_k=injected_k, 
                                            injected_q=injected_q,
                                            injected_v=injected_v)
            else:
                self.attn1_out = self.attn1(self.norm1(x), context=context, )
        else:
            if is_injected_step:
                for k, v in modulate_params["injected_features_group"].items():
                    if "temporal_self_attn_k" in k:
                        injected_k = v
                    elif "temporal_self_attn_q" in k:
                        injected_q = v
                    elif "temporal_self_attn_v" in k:
                        injected_v = v
                self.attn1_out = self.attn1(self.norm1(x), 
                                            injected_k=injected_k, 
                                            injected_q=injected_q,
                                            injected_v=injected_v)
            else:
                self.attn1_out = self.attn1(self.norm1(x))
        
        if is_modulate_step and "self_attn" in modulate_params["modulate_attn_type"]:
            masks = modulate_params["feature_masks"]
            num_masks = len(masks)
            half_hw = self.attn1_out.shape[0] // 2
            for i, mask in enumerate(masks):
                if i in modulate_params["modulate_block_frames_group"] and i in modulate_params["modulate_layer_frames_group"] and \
                    i in modulate_params["modulate_timestep_frames_group"]:  
                    modulate_lambda = get_modulate_lambda(
                    modulate_params["modulate_lambda_start"],
                    modulate_params["modulate_lambda_end"],
                    modulate_params["modulate_schedule"],
                    total_steps=modulate_params["num_frames"],
                    current_step=i,
                    )
                    
                    self.attn1_out[half_hw:, i] = self.attn1_out[half_hw:, i] + \
                        modulate_lambda * mask[:, None]
                    if modulate_params["modulate_uc"]:
                        self.attn1_out[:half_hw, i] = self.attn1_out[:half_hw, i] + \
                        modulate_lambda * mask[:, None]
        
        x = self.attn1_out + x

        injected_k = None
        injected_q = None    
        injected_v = None
            
        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                self.attn2_out = self.attn2(self.norm2(x))
            else:
                self.attn2_out = self.attn2(self.norm2(x), context=context,)
                
            if is_modulate_step and "cross_attn" in modulate_params["modulate_attn_type"]:
                masks = modulate_params["feature_masks"]
                num_masks = len(masks)
                half_hw = self.attn2_out.shape[0] // 2
                for i, mask in enumerate(masks):
                    if i in modulate_params["modulate_block_frames_group"] and i in modulate_params["modulate_layer_frames_group"] and \
                    i in modulate_params["modulate_timestep_frames_group"]:       
                        modulate_lambda = get_modulate_lambda(
                            modulate_params["modulate_lambda_start"],
                            modulate_params["modulate_lambda_end"],
                            modulate_params["modulate_schedule"],
                            total_steps=modulate_params["num_frames"],
                            current_step=i,
                            )
                        self.attn2_out[half_hw:, i] = self.attn2_out[half_hw:, i] + \
                            modulate_lambda * mask[:, None]
                        if modulate_params["modulate_uc"]:
                            self.attn2_out[:half_hw, i] = self.attn2_out[:half_hw, i] + \
                            modulate_lambda * mask[:, None]
                                
            x = self.attn2_out + x

                
        
        x_skip = x
        
        self.ff_out = self.ff(self.norm3(x))
        if is_modulate_step and "ff_out" in modulate_params["modulate_attn_type"]:
            masks = modulate_params["feature_masks"]
            num_masks = len(masks)
            half_hw = self.ff_out.shape[0] // 2
            for i, mask in enumerate(masks):
                if i in modulate_params["modulate_block_frames_group"] and i in modulate_params["modulate_layer_frames_group"] and \
                i in modulate_params["modulate_timestep_frames_group"]:       
                    modulate_lambda = get_modulate_lambda(
                        modulate_params["modulate_lambda_start"],
                        modulate_params["modulate_lambda_end"],
                        modulate_params["modulate_schedule"],
                        total_steps=modulate_params["num_frames"],
                        current_step=i,
                        )
                    self.ff_out[half_hw:, i] = self.ff_out[half_hw:, i] + \
                        modulate_lambda * mask[:, None]
                    if modulate_params["modulate_uc"]:
                        self.ff_out[:half_hw, i] = self.ff_out[:half_hw, i] + \
                        modulate_lambda * mask[:, None]
        
        
        
        if self.is_res:
            x = x_skip + self.ff_out

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim
            

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )
        
        self.features_after_temporal = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        is_modulate_step: bool = False,
        is_injected_step: bool = False,
        modulate_params: Dict = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            if is_modulate_step and "spatial" in modulate_params["modulate_layer_type"]:
                is_modulate_step_spatial = True
                if "spatial" in modulate_params["modulate_layer_frames"].keys():
                    modulate_params["modulate_layer_frames_group"] = modulate_params["modulate_layer_frames"]["spatial"]
                else:
                    modulate_params["modulate_layer_frames_group"] = list(range(modulate_params["num_frames"]))
            else:
                is_modulate_step_spatial = False
            

            x = block(
                x,
                context=spatial_context,
                is_modulate_step=is_modulate_step_spatial,
                is_injected_step=is_injected_step,
                modulate_params=modulate_params,
            )

            x_mix = x
            x_mix = x_mix + emb

            if is_modulate_step and "temporal" in modulate_params["modulate_layer_type"]:
                is_modulate_step_temporal = True
                if "temporal" in modulate_params["modulate_layer_frames"].keys():
                    modulate_params["modulate_layer_frames_group"] = modulate_params["modulate_layer_frames"]["temporal"]
                else:
                    modulate_params["modulate_layer_frames_group"] = list(range(modulate_params["num_frames"]))
            else:
                is_modulate_step_temporal = False
                
            x_mix = mix_block(
                x_mix,
                context=time_context,
                timesteps=timesteps,
                is_modulate_step=is_modulate_step_temporal,
                is_injected_step=is_injected_step,
                modulate_params=modulate_params,
            )

         
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )
            
            
                
        if self.use_linear:
            x = self.proj_out(x)
        
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        
        self.features_after_temporal = x
        return out
