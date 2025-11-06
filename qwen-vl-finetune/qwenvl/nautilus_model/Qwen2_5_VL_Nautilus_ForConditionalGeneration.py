from cmath import isnan
from heapq import merge
from pyexpat import model
from typing_extensions import Self
from transformers import Qwen2_5_VLForConditionalGeneration, VisualBertConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (Qwen2_5_VLCausalLMOutputWithPast,QWEN2_5_VL_INPUTS_DOCSTRING,_CONFIG_FOR_DOC,
                                                                add_start_docstrings_to_model_forward,replace_return_docstrings,
                                                                Qwen2_5_VisionTransformerPretrainedModel,)

from typing import Any, Dict, List, Optional, Tuple, Union
from .Nautilus_layers import CrossAttentionNetwork, MLP, GlobalQueries 
from .dinov2 import DINOv2
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F

class Qwen2_5_Nautilus_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.nautilus_pool_layer = torch.nn.AdaptiveAvgPool2d(16)
        dino_vit_dim = 1024
        qwen_vit_dim = 1280
        self.nautilus_w1_mlp = MLP(dino_vit_dim,[qwen_vit_dim],qwen_vit_dim)
        self.nautilus_dark_mlp = MLP(qwen_vit_dim,[qwen_vit_dim],qwen_vit_dim)
        self.nautilus_encoder = DINOv2("vitl")
        self.nautilus_dark_attn = CrossAttentionNetwork(qwen_vit_dim,8)
        self.nautilus_global_queries = GlobalQueries(nn.Parameter(torch.randn(1, 1, qwen_vit_dim)))
    # ehance forward
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        pixel_values = hidden_states
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        ehanced_hidden_states = self.ehance_embeds(hidden_states, pixel_values, grid_thw, cu_seqlens, window_index)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        ehanced_hidden_states = self.merger(ehanced_hidden_states)
        ehanced_hidden_states = ehanced_hidden_states[reverse_indices, :]

        return torch.stack((hidden_states, ehanced_hidden_states), dim=1).reshape(-1, hidden_states.shape[1])

    def ehance_embeds(self, image_embeds, pixel_values, grid_thw, cu_seqlens, window_index):
        '''
        image_embeds: 经过qwen vit处理但seq“无序”tensor
        pixel_values: 原始图片pixel values经过Preprocess后的值，但reshape到了patch维度，有序
        grid_thw: grid info
        cu_seqlens: every sample seqlen in a batch
        '''
        # remove copies
        pixel_single_values = pixel_values.reshape(-1, 3, 2, 14, 14)[:,:,0,:,:].squeeze(2)
        pixel_image_list, nautilus_embeds_list = [], []
        for idx,(grid_t,grid_h,grid_w) in zip(range(len(cu_seqlens) - 1),grid_thw):
            pixel_image = pixel_single_values[cu_seqlens[idx]:cu_seqlens[idx+1]]
            pixel_image = self.restore_image_from_patches(pixel_image,grid_h,grid_w)
            nautilus_embeds = self.nautilus_encoder.get_intermediate_layers(pixel_image.unsqueeze(0), [23], return_class_token=False)[0]
            pixel_image_list.append(pixel_image)
            nautilus_embeds_list.append(nautilus_embeds.squeeze(0))
        
        # shuffle nautilus_embeds pixel_image
        batch_nautilus_embeds = torch.cat(nautilus_embeds_list, dim=0)
        batch_pixel_image = pixel_single_values.mean(dim=(1, 2, 3))

        seq_len,_ = batch_nautilus_embeds.shape
        batch_nautilus_embeds = batch_nautilus_embeds.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        batch_nautilus_embeds = batch_nautilus_embeds[window_index, :, :]
        batch_nautilus_embeds = batch_nautilus_embeds.reshape(seq_len, -1)

        batch_pixel_image = batch_pixel_image.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        batch_pixel_image = batch_pixel_image[window_index, :, :]
        batch_pixel_image = batch_pixel_image.reshape(seq_len, -1).squeeze()

        weight_1 = self.nautilus_w1_mlp(batch_nautilus_embeds)
        weight_2 = 1 / torch.exp(- weight_1)

        # split batch
        filted_embeds_list = []
        for idx in range(len(cu_seqlens) - 1):
            pixel_image = batch_pixel_image[cu_seqlens[idx]:cu_seqlens[idx+1]]
            single_image_embeds = image_embeds[cu_seqlens[idx]:cu_seqlens[idx+1]]
            min_index = torch.argmin(pixel_image)
            dark_mask = torch.zeros(1,single_image_embeds.shape[0]).to(single_image_embeds.device)
            dark_mask[0,min_index] = 1
            dark_mask = dark_mask.to(single_image_embeds.dtype)
            dark_embeds = dark_mask @ single_image_embeds
            # min_index_list.append(min_index)

            mean_featuers = torch.mean(single_image_embeds, dim=0).unsqueeze(0)
            global_embeds = self.nautilus_dark_attn(self.nautilus_global_queries() + mean_featuers.unsqueeze(1),single_image_embeds.unsqueeze(1),single_image_embeds.unsqueeze(1))[0]            
            dark_embeds = dark_embeds - global_embeds
            dark_embeds = self.nautilus_dark_mlp(dark_embeds.squeeze(1))
            
            filted_embeds = single_image_embeds - dark_embeds
            filted_embeds_list.append(filted_embeds)
        batch_filted_embeds = torch.cat(filted_embeds_list, dim=0)
        ehanced_hidden_states = weight_2 * batch_filted_embeds

        return ehanced_hidden_states

    
    def restore_image_from_patches(self, patches: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:


        N, C, patch_size, _ = patches.shape
        assert N == h_patches * w_patches, "patch not match，make sure that h_patches * w_patches == N"

        patches = patches.view(h_patches, w_patches, C, patch_size, patch_size)
        patches = patches.permute(2, 0, 3, 1, 4).contiguous()
        image = patches.view(C, h_patches * patch_size, w_patches * patch_size)

        return image

class Qwen2_5_VL_Nautilus_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_Nautilus_VisionTransformerPretrainedModel._from_config(config.vision_config)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d, nn.LayerNorm)): 
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def init_nautilus_model(self, dino_path, dino_only=False):

        if not dino_only:
            # print(f"nautilus model successfuly inited")
            try:
                self.visual.nautilus_w1_mlp.weight_init()
                self.visual.nautilus_dark_mlp.weight_init()
                self.visual.nautilus_dark_attn.linear_init()
                self.visual.nautilus_global_queries.weight_init()
            except:
                self.visual.nautilus_w1_mlp.modules_to_save.default.weight_init()
                self.visual.nautilus_dark_mlp.modules_to_save.default.weight_init()
                self.visual.nautilus_dark_attn.modules_to_save.default.linear_init()
                self.visual.nautilus_global_queries.modules_to_save.default.weight_init()
        self.visual.nautilus_encoder.load_model(dino_path)     
        
    
    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                # image_embeds = image_embeds + self.nautilus_debug_mlp_2(image_embeds)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
